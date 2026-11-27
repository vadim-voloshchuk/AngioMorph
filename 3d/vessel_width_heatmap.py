from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize


# ==============================================================================
# 1. АРХИТЕКТУРА DEEPLABV3+
# ==============================================================================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: list[int]):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.aspp = ASPP(in_channels=256, out_channels=256, atrous_rates=[6, 12, 18])

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.size()
        x_flat = x.view(B * T, C, H, W)

        f1 = self.block1(x_flat)
        p1 = self.pool1(f1)
        f2 = self.block2(p1)
        p2 = self.pool2(f2)
        f3 = self.block3(p2)
        p3 = self.pool3(f3)

        aspp_out = self.aspp(p3)

        up_1 = self.up1(aspp_out)
        dec_1 = self.dec_conv1(torch.cat([up_1, f3], dim=1))
        up_2 = self.up2(dec_1)
        dec_2 = self.dec_conv2(torch.cat([up_2, f2], dim=1))
        up_3 = self.up3(dec_2)
        dec_3 = self.dec_conv3(torch.cat([up_3, f1], dim=1))

        logits = self.classifier(dec_3)
        return logits.view(B, T, -1, H, W)


# ==============================================================================
# 2. РАСЧЕТ ТЕПЛОВОЙ КАРТЫ ШИРИНЫ (EDT + ВОРОНОЙ)
# ==============================================================================
def clean_binary_mask(mask: np.ndarray, min_area: int = 40) -> np.ndarray:
    """Закрывает микроотверстия и удаляет мелкие шумовые пятна, сохраняя ветви."""
    binary = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 1
    return cleaned


def compute_width_map_edt_voronoi(
    mask: np.ndarray,
    pixel_spacing: float = 1.0,
    smooth_sigma: float = 1.5,
    min_area: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет 2D-карту диаметров сосуда (в мм) методом EDT + проекция Вороного.
    Возвращает:
        width_map_mm: float32 (H, W) — диаметр в мм внутри сосуда, np.nan снаружи.
        skeleton: bool (H, W) — центральная линия сосуда.
    """
    clean_mask = clean_binary_mask(mask, min_area=min_area)
    if np.sum(clean_mask) == 0:
        return np.full(mask.shape, np.nan, dtype=np.float32), np.zeros(mask.shape, dtype=bool)

    # 1. Карта радиусов (расстояние до ближайшей границы)
    radius_map = distance_transform_edt(clean_mask)

    # 2. Скелет (центральная линия толщиной 1 пиксель)
    skeleton = skeletonize(clean_mask > 0)
    if not np.any(skeleton):
        return np.full(mask.shape, np.nan, dtype=np.float32), skeleton

    # 3. Проекция Вороного: для каждого пикселя находим ближайший пиксель скелета
    _, nearest_idx = distance_transform_edt(~skeleton, return_indices=True)
    raw_diameter_px = 2.0 * radius_map[nearest_idx[0], nearest_idx[1]]
    raw_diameter_px = np.where(clean_mask > 0, raw_diameter_px, 0.0)

    # 4. Нормализованное гауссово сглаживание строго внутри маски (без затекания фона)
    if smooth_sigma > 0:
        mask_f = clean_mask.astype(np.float64)
        numerator = gaussian_filter(raw_diameter_px * mask_f, sigma=smooth_sigma)
        denominator = gaussian_filter(mask_f, sigma=smooth_sigma)
        smooth_diameter_px = np.where(clean_mask > 0, numerator / np.maximum(denominator, 1e-6), np.nan)
    else:
        smooth_diameter_px = np.where(clean_mask > 0, raw_diameter_px, np.nan)

    width_map_mm = (smooth_diameter_px * pixel_spacing).astype(np.float32)
    return width_map_mm, skeleton


def estimate_catheter_width_px(catheter_mask: np.ndarray) -> float | None:
    """Оценивает медианный диаметр катетера в пикселях для возможной калибровки."""
    clean_cath = clean_binary_mask(catheter_mask, min_area=30)
    if np.sum(clean_cath) == 0:
        return None
    radius_map = distance_transform_edt(clean_cath)
    skel = skeletonize(clean_cath > 0)
    if not np.any(skel):
        return None
    return float(np.median(radius_map[skel]) * 2.0)


# ==============================================================================
# 3. ИНФЕРЕНС И DICOMDIR
# ==============================================================================
def load_deeplab_model(weights_path: Path, device: torch.device) -> DeepLabV3Plus:
    model = DeepLabV3Plus(in_channels=1, num_classes=3)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def predict_masks(model: DeepLabV3Plus, frame: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    orig_h, orig_w = frame.shape
    resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
    arr = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5

    input_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)

    preds = torch.argmax(output.view(1, 3, 512, 512), dim=1).squeeze().cpu().numpy()
    artery_mask = cv2.resize((preds == 1).astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    catheter_mask = cv2.resize((preds == 2).astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return artery_mask, catheter_mask


def parse_dicomdir_and_select(patient_folder: Path) -> Path:
    dicomdir_path = patient_folder / "DICOMDIR"
    if not dicomdir_path.exists():
        dicomdir_path = patient_folder / "dicomdir"
        if not dicomdir_path.exists():
            raise FileNotFoundError(f"Файл DICOMDIR не найден в {patient_folder}")

    dicomdir = pydicom.dcmread(dicomdir_path)
    runs_list: list[tuple[str, Path]] = []
    current_series_desc = "Unknown Series"

    for record in dicomdir.DirectoryRecordSequence:
        if record.DirectoryRecordType == "SERIES":
            current_series_desc = getattr(record, "SeriesDescription", f"Серия {getattr(record, 'SeriesNumber', '?')}")
        elif record.DirectoryRecordType == "IMAGE":
            rel_parts = record.ReferencedFileID
            rel_path = Path(rel_parts) if isinstance(rel_parts, str) else Path(*rel_parts)
            full_path = patient_folder / rel_path
            instance_num = getattr(record, "InstanceNumber", "?")
            runs_list.append((f"{current_series_desc} (Файл: {full_path.name}, Запись: {instance_num})", full_path))

    if not runs_list:
        raise ValueError("В DICOMDIR не найдено ни одной проекции.")

    print("\nДоступные проекции:")
    for i, (desc, _) in enumerate(runs_list, start=1):
        print(f"[{i}] {desc}")

    while True:
        try:
            choice = int(input("\nВыберите номер проекции: "))
            if 1 <= choice <= len(runs_list):
                return runs_list[choice - 1][1]
        except ValueError:
            pass
        print("Неверный номер. Попробуйте еще раз.")


# ==============================================================================
# 4. ВИЗУАЛИЗАЦИЯ (PNG + MP4)
# ==============================================================================
def save_peak_heatmap_figure(
    frame: np.ndarray,
    width_map_mm: np.ndarray,
    skeleton: np.ndarray,
    frame_idx: int,
    vmax_mm: float,
    output_path: Path,
    cmap_name: str = "turbo",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(frame, cmap="gray")
    axes[0].set_title(f"Исходная ангиограмма (кадр {frame_idx + 1})", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(frame, cmap="gray")
    im = axes[1].imshow(width_map_mm, cmap=cmap_name, vmin=0.0, vmax=vmax_mm, alpha=0.82)
    # Тонкая центральная линия для контроля корректности скелета
    skel_overlay = np.where(skeleton, 1.0, np.nan)
    axes[1].imshow(skel_overlay, cmap="gray", vmin=0, vmax=1, alpha=0.35)

    axes[1].set_title("Тепловая карта ширины артерий (EDT + Вороной)", fontsize=12)
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Диаметр сосуда (мм)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    print(f"Изображение пикового кадра сохранено: {output_path}")
    plt.show()


def save_heatmap_video(
    frames: list[np.ndarray],
    width_maps: list[np.ndarray],
    vmax_mm: float,
    fps: float,
    output_path: Path,
) -> None:
    h, w = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w * 2, h))

    for frame, wmap in zip(frames, width_maps):
        base_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        overlay = base_bgr.copy()

        valid = ~np.isnan(wmap)
        if np.any(valid):
            norm = np.clip(np.nan_to_num(wmap, nan=0.0) / vmax_mm, 0.0, 1.0)
            norm_u8 = (norm * 255).astype(np.uint8)
            colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
            overlay[valid] = cv2.addWeighted(base_bgr[valid], 0.25, colored[valid], 0.75, 0)

        combined = np.hstack([base_bgr, overlay])
        writer.write(combined)

    writer.release()
    print(f"Видео с тепловой картой сохранено: {output_path}")


# ==============================================================================
# 5. ТОЧКА ВХОДА
# ==============================================================================
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    weights_path = Path(input("Путь к весам DeepLabV3+: ").strip())
    patient_folder = Path(input("Путь к папке пациента (с DICOMDIR): ").strip())

    model = load_deeplab_model(weights_path, device)
    dicom_path = parse_dicomdir_and_select(patient_folder)

    dcm = pydicom.dcmread(dicom_path)
    pixel_data = dcm.pixel_array
    if pixel_data.ndim == 2:
        pixel_data = np.expand_dims(pixel_data, axis=0)

    imager_spacing = float(dcm.ImagerPixelSpacing[0]) if "ImagerPixelSpacing" in dcm else 0.2
    fps = float(dcm.CineRate) if "CineRate" in dcm else 15.0

    frames: list[np.ndarray] = []
    artery_masks: list[np.ndarray] = []
    cath_widths_px: list[float] = []

    print(f"\nОбработка {pixel_data.shape[0]} кадров...")
    for i in range(pixel_data.shape[0]):
        frame = cv2.normalize(pixel_data[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        art_mask, cath_mask = predict_masks(model, frame, device)

        frames.append(frame)
        artery_masks.append(art_mask)

        cath_w = estimate_catheter_width_px(cath_mask)
        if cath_w is not None and cath_w > 2.0:
            cath_widths_px.append(cath_w)
        print(f"Сегментация: {i + 1}/{pixel_data.shape[0]}", end="\r")

    print("\n")
    # Выбор калибровки: DICOM ImagerPixelSpacing или по катетеру (например, 6F = 2.0 мм)
    pixel_spacing = imager_spacing
    if cath_widths_px:
        med_cath_px = float(np.median(cath_widths_px))
        cath_spacing_6f = 2.0 / med_cath_px
        print(f"ImagerPixelSpacing: {imager_spacing:.4f} мм/пикс")
        print(f"Медианная ширина катетера: {med_cath_px:.2f} пикс (для 6F=2.0 мм -> {cath_spacing_6f:.4f} мм/пикс)")
        use_cath = input("Использовать калибровку по катетеру 6F вместо ImagerPixelSpacing? [y/N]: ").strip().lower()
        if use_cath == "y":
            pixel_spacing = cath_spacing_6f

    # Построение карт ширины для всех кадров
    width_maps: list[np.ndarray] = []
    skeletons: list[np.ndarray] = []
    areas: list[int] = []

    for art_mask in artery_masks:
        wmap, skel = compute_width_map_edt_voronoi(
            art_mask,
            pixel_spacing=pixel_spacing,
            smooth_sigma=1.5,
            min_area=40,
        )
        width_maps.append(wmap)
        skeletons.append(skel)
        areas.append(int(np.sum(~np.isnan(wmap))))

    peak_idx = int(np.argmax(areas))
    peak_wmap = width_maps[peak_idx]

    valid_vals = peak_wmap[~np.isnan(peak_wmap)]
    if valid_vals.size == 0:
        print("Артерии не обнаружены ни на одном кадре.")
        return

    # Верхний предел шкалы берем по 99-му перцентилю, чтобы выбросы в бифуркациях не сжимали палитру
    vmax_mm = float(np.percentile(valid_vals, 99))
    print(f"Пиковый кадр: {peak_idx + 1} | Медианный диаметр: {np.median(valid_vals):.2f} мм | 99-й перц.: {vmax_mm:.2f} мм")

    out_dir = Path("heatmap_results")
    out_dir.mkdir(exist_ok=True)

    save_heatmap_video(frames, width_maps, vmax_mm, fps, out_dir / "width_heatmap_video.mp4")
    save_peak_heatmap_figure(
        frames[peak_idx],
        peak_wmap,
        skeletons[peak_idx],
        peak_idx,
        vmax_mm,
        out_dir / "width_heatmap_peak.png",
        cmap_name="turbo",  # Замените на "turbo_r", если нужно подсвечивать сужения красным
    )


if __name__ == "__main__":
    main()