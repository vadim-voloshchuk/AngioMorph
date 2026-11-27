from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
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
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.size()
        x_flat = x.view(B * T, C, H, W)
        f1 = self.block1(x_flat)
        f2 = self.block2(self.pool1(f1))
        f3 = self.block3(self.pool2(f2))
        aspp_out = self.aspp(self.pool3(f3))
        dec_1 = self.dec_conv1(torch.cat([self.up1(aspp_out), f3], dim=1))
        dec_2 = self.dec_conv2(torch.cat([self.up2(dec_1), f2], dim=1))
        dec_3 = self.dec_conv3(torch.cat([self.up3(dec_2), f1], dim=1))
        return self.classifier(dec_3).view(B, T, -1, H, W)


# ==============================================================================
# 2. РЕШАТЕЛЬ УРАВНЕНИЯ ЛАПЛАСА (YEZZI–PRINCE) ДЛЯ ОДНОЙ ВЕТВИ
# ==============================================================================
def _order_skeleton_branch(skel_branch: np.ndarray) -> np.ndarray:
    """Упорядочивает пиксели неветвящегося скелета от одного конца к другому."""
    pts = np.argwhere(skel_branch)
    if len(pts) < 3:
        return pts

    pt_set = {tuple(p) for p in pts}
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in pt_set:
        nbs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy != 0 or dx != 0) and (y + dy, x + dx) in pt_set:
                    nbs.append((y + dy, x + dx))
        adj[(y, x)] = nbs

    # Стартуем с концевой точки (степень 1), если есть
    endpoints = [p for p, nbs in adj.items() if len(nbs) == 1]
    start = endpoints[0] if endpoints else next(iter(pt_set))

    ordered = []
    visited = {start}
    q = deque([start])
    while q:
        curr = q.popleft()
        ordered.append(curr)
        for nb in adj[curr]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return np.array(ordered, dtype=np.int32)


def _solve_laplace_2d(interior_mask: np.ndarray, wall0: np.ndarray, wall1: np.ndarray) -> np.ndarray:
    """Решает nabla^2 phi = 0 с Дирихле (wall0=0, wall1=1) и Нейманом на торцах."""
    h, w = interior_mask.shape
    phi = np.full((h, w), np.nan, dtype=np.float64)
    phi[wall0] = 0.0
    phi[wall1] = 1.0

    ys, xs = np.where(interior_mask)
    n = len(ys)
    if n == 0:
        return phi

    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[ys, xs] = np.arange(n)

    rows, cols, vals = [], [], []
    rhs = np.zeros(n, dtype=np.float64)

    for k in range(n):
        y, x = ys[k], xs[k]
        diag = 0.0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if interior_mask[ny, nx]:
                    diag += 1.0
                    rows.append(k)
                    cols.append(idx_map[ny, nx])
                    vals.append(-1.0)
                elif wall0[ny, nx]:
                    diag += 1.0
                elif wall1[ny, nx]:
                    diag += 1.0
                    rhs[k] += 1.0
                # Иначе — торец ветви (условие Неймана: нулевой поток, слагаемое не добавляется)
        rows.append(k)
        cols.append(k)
        vals.append(max(diag, 1.0))

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    sol = spla.spsolve(A, rhs)
    phi[ys, xs] = np.clip(sol, 0.0, 1.0)
    return phi


def _solve_upwind_length(
    interior_mask: np.ndarray,
    zero_wall: np.ndarray,
    ty: np.ndarray,
    tx: np.ndarray,
) -> np.ndarray:
    """
    Решает (T * nabla L) = 1 методом upwind-разностей с L = 0 на zero_wall.
    """
    h, w = interior_mask.shape
    L = np.full((h, w), np.nan, dtype=np.float64)
    L[zero_wall] = 0.0

    ys, xs = np.where(interior_mask)
    n = len(ys)
    if n == 0:
        return L

    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[ys, xs] = np.arange(n)

    rows, cols, vals = [], [], []
    rhs = np.ones(n, dtype=np.float64)

    for k in range(n):
        y, x = ys[k], xs[k]
        vy, vx = float(ty[y, x]), float(tx[y, x])
        diag = 0.0

        # Компонента по Y (upwind: смотрим назад против потока)
        if abs(vy) > 1e-6:
            ny = y - (1 if vy > 0 else -1)
            if 0 <= ny < h and (interior_mask[ny, x] or zero_wall[ny, x]):
                wy = abs(vy)
                diag += wy
                if interior_mask[ny, x]:
                    rows.append(k)
                    cols.append(idx_map[ny, x])
                    vals.append(-wy)

        # Компонента по X (upwind)
        if abs(vx) > 1e-6:
            nx = x - (1 if vx > 0 else -1)
            if 0 <= nx < w and (interior_mask[y, nx] or zero_wall[y, nx]):
                wx = abs(vx)
                diag += wx
                if interior_mask[y, nx]:
                    rows.append(k)
                    cols.append(idx_map[y, nx])
                    vals.append(-wx)

        if diag < 1e-6:
            # Редкий краевой пиксель у торца
            diag = 1.0
            rhs[k] = 0.0

        rows.append(k)
        cols.append(k)
        vals.append(diag)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    sol = spla.spsolve(A, rhs)
    L[ys, xs] = np.maximum(sol, 0.0)
    return L


def solve_branch_yezzi_prince(
    branch_mask: np.ndarray,
    skel_branch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет потенциал phi и толщину W = L0 + L1 (в пикселях) для одной ветви.
    """
    ordered_pts = _order_skeleton_branch(skel_branch)
    if len(ordered_pts) < 6:
        return np.full(branch_mask.shape, np.nan), np.full(branch_mask.shape, np.nan)

    # Сглаженные касательные вдоль скелета
    pts_f = ordered_pts.astype(np.float64)
    tangents = np.gradient(pts_f, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-6)

    # Внешняя граница (1-пиксельное кольцо вокруг маски ветви)
    kernel4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    dilated = cv2.dilate(branch_mask.astype(np.uint8), kernel4, iterations=1) > 0
    outer_ring = dilated & (~branch_mask)

    # Привязка каждой точки кольца к ближайшей точке скелета
    skel_idx_map = np.full(branch_mask.shape, -1, dtype=np.int32)
    skel_idx_map[ordered_pts[:, 0], ordered_pts[:, 1]] = np.arange(len(ordered_pts))
    _, nearest = distance_transform_edt(~skel_branch, return_indices=True)

    ry, rx = np.where(outer_ring)
    sy = nearest[0, ry, rx]
    sx = nearest[1, ry, rx]
    k_idx = skel_idx_map[sy, sx]

    # Исключаем торцы ветви (ближайшие к первым/последним 2 пикселям скелета) -> условие Неймана
    is_cap = (k_idx <= 1) | (k_idx >= len(ordered_pts) - 2)
    valid = ~is_cap

    # Знак 2D векторного произведения: (tangent x vec_to_wall) делит границу на левую и правую стенки
    ty_s = tangents[k_idx, 0]
    tx_s = tangents[k_idx, 1]
    vy_w = ry - sy
    vx_w = rx - sx
    cross = tx_s * vy_w - ty_s * vx_w

    wall0 = np.zeros_like(branch_mask, dtype=bool)
    wall1 = np.zeros_like(branch_mask, dtype=bool)
    wall0[ry[valid & (cross < 0)], rx[valid & (cross < 0)]] = True
    wall1[ry[valid & (cross >= 0)], rx[valid & (cross >= 0)]] = True

    if not np.any(wall0) or not np.any(wall1):
        return np.full(branch_mask.shape, np.nan), np.full(branch_mask.shape, np.nan)

    # 1. Решаем уравнение Лапласа
    phi = _solve_laplace_2d(branch_mask, wall0, wall1)

    # 2. Вычисляем нормированный градиент T = nabla(phi) / |nabla(phi)|
    phi_filled = np.nan_to_num(phi, nan=0.5)
    gy, gx = np.gradient(phi_filled)
    gnorm = np.hypot(gy, gx)
    ty = gy / np.maximum(gnorm, 1e-6)
    tx = gx / np.maximum(gnorm, 1e-6)

    # 3. Решаем УЧП для L0 (от wall0) и L1 (от wall1)
    L0 = _solve_upwind_length(branch_mask, wall0, ty, tx)
    L1 = _solve_upwind_length(branch_mask, wall1, -ty, -tx)

    width_px = L0 + L1
    # Вычитаем 1 пиксель, т.к. wall0 и wall1 лежат на внешнем кольце (на +0.5 пикс с каждой стороны)
    width_px = np.where(branch_mask, np.maximum(width_px - 1.0, 0.5), np.nan)
    return phi, width_px


# ==============================================================================
# 3. ДЕКОМПОЗИЦИЯ ДЕРЕВА НА ВЕТВИ + СБОРКА КАРТЫ
# ==============================================================================
def compute_width_map_laplace(
    prob_map: np.ndarray,
    pixel_spacing: float,
    prob_thresh: float = 0.5,
    min_area: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Разбивает маску сосудистого дерева на ветви вне узлов бифуркаций,
    решает уравнение Лапласа (Yezzi–Prince) на каждой ветви и собирает общую карту (в мм).
    """
    binary = (prob_map >= prob_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask = np.zeros_like(binary, dtype=bool)
    for lid in range(1, num_labels):
        if stats[lid, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == lid] = True

    h, w = clean_mask.shape
    full_width_px = np.full((h, w), np.nan, dtype=np.float32)
    full_phi = np.full((h, w), np.nan, dtype=np.float32)
    junction_zone = np.zeros((h, w), dtype=bool)

    if not np.any(clean_mask):
        return full_width_px, full_phi, junction_zone

    radius_map = distance_transform_edt(clean_mask)
    skeleton = skeletonize(clean_mask)

    # Поиск узлов бифуркаций/пересечений (число 8-связных соседей >= 3)
    nb_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nb_count = cv2.filter2D(skeleton.astype(np.uint8), -1, nb_kernel, borderType=cv2.BORDER_CONSTANT)
    junction_pts = np.argwhere(skeleton & (nb_count >= 3))

    # Вырезаем зоны слияния (POC) радиусом 1.3 * R вокруг каждого узла
    for jy, jx in junction_pts:
        r_poc = max(int(np.ceil(radius_map[jy, jx] * 1.3)), 3)
        cv2.circle(junction_zone.view(np.uint8), (int(jx), int(jy)), r_poc, 1, -1)

    junction_zone &= clean_mask
    skel_no_junc = skeleton & (~junction_zone)

    # Нумеруем отдельные ветви скелета
    n_branches, branch_skel_labels = cv2.connectedComponents(skel_no_junc.astype(np.uint8), connectivity=8)
    if n_branches <= 1:
        return full_width_px, full_phi, junction_zone

    # Распределяем пиксели маски (вне зон слияния) по ближайшей ветви скелета
    _, nearest = distance_transform_edt(branch_skel_labels == 0, return_indices=True)
    branch_mask_labels = np.where(clean_mask & (~junction_zone), branch_skel_labels[nearest[0], nearest[1]], 0)

    # Решаем Yezzi–Prince в компактном bounding box каждой ветви
    pad = 3
    for bid in range(1, n_branches):
        b_mask = branch_mask_labels == bid
        b_skel = branch_skel_labels == bid
        if np.sum(b_skel) < 6 or np.sum(b_mask) < 15:
            continue

        ys, xs = np.where(b_mask)
        y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)

        roi_phi, roi_w = solve_branch_yezzi_prince(b_mask[y0:y1, x0:x1], b_skel[y0:y1, x0:x1])

        valid_roi = ~np.isnan(roi_w)
        full_width_px[y0:y1, x0:x1][valid_roi] = roi_w[valid_roi]
        full_phi[y0:y1, x0:x1][valid_roi] = roi_phi[valid_roi]

    # Для зон бифуркаций (POC) можно оставить NaN или заполнить ближайшими валидными значениями ветвей
    valid_w = ~np.isnan(full_width_px)
    if np.any(valid_w) and np.any(junction_zone):
        _, idx = distance_transform_edt(~valid_w, return_indices=True)
        full_width_px[junction_zone] = full_width_px[idx[0][junction_zone], idx[1][junction_zone]]

    return full_width_px * pixel_spacing, full_phi, junction_zone


# ==============================================================================
# 4. ИЗОЦЕНТРОВАЯ КАЛИБРОВКА И ИНФЕРЕНС
# ==============================================================================
def get_isocentric_pixel_spacing(dcm: pydicom.Dataset) -> tuple[float, str]:
    """Вычисляет масштаб в плоскости изоцентра сердца по DICOM-тегам C-дуги."""
    imager_spacing = float(dcm.ImagerPixelSpacing[0]) if "ImagerPixelSpacing" in dcm else 0.2

    if "EstimatedRadiographicMagnificationFactor" in dcm:
        ermf = float(dcm.EstimatedRadiographicMagnificationFactor)
        if ermf > 0.5:
            return imager_spacing / ermf, f"Изоцентр (ImagerPixelSpacing / ERMF={ermf:.2f})"

    if "DistanceSourceToDetector" in dcm and "DistanceSourceToPatient" in dcm:
        sid = float(dcm.DistanceSourceToDetector)
        sod = float(dcm.DistanceSourceToPatient)
        if sid > 0 and sod > 0:
            ermf = sid / sod
            return imager_spacing / ermf, f"Изоцентр (SID/SOD={ermf:.2f})"

    return imager_spacing, "Внимание: теги SID/SOD/ERMF не найдены, использован ImagerPixelSpacing"


def predict_artery_prob(model: DeepLabV3Plus, frame: np.ndarray, device: torch.device) -> np.ndarray:
    """Возвращает непрерывную карту вероятностей артерии в исходном разрешении (INTER_LINEAR)."""
    orig_h, orig_w = frame.shape
    resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
    arr = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5

    inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp).view(1, 3, 512, 512)
        prob_art = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

    return cv2.resize(prob_art, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def parse_dicomdir_and_select(patient_folder: Path) -> Path:
    dicomdir_path = patient_folder / "DICOMDIR"
    if not dicomdir_path.exists():
        dicomdir_path = patient_folder / "dicomdir"
        if not dicomdir_path.exists():
            raise FileNotFoundError(f"Файл DICOMDIR не найден в {patient_folder}")

    dicomdir = pydicom.dcmread(dicomdir_path)
    runs_list: list[tuple[str, Path]] = []
    desc = "Unknown Series"

    for rec in dicomdir.DirectoryRecordSequence:
        if rec.DirectoryRecordType == "SERIES":
            desc = getattr(rec, "SeriesDescription", f"Серия {getattr(rec, 'SeriesNumber', '?')}")
        elif rec.DirectoryRecordType == "IMAGE":
            rel = rec.ReferencedFileID
            rel_path = Path(rel) if isinstance(rel, str) else Path(*rel)
            full_path = patient_folder / rel_path
            inst = getattr(rec, "InstanceNumber", "?")
            runs_list.append((f"{desc} (Файл: {full_path.name}, Запись: {inst})", full_path))

    print("\nДоступные проекции:")
    for i, (name, _) in enumerate(runs_list, start=1):
        print(f"[{i}] {name}")

    while True:
        try:
            choice = int(input("\nВыберите номер проекции: "))
            if 1 <= choice <= len(runs_list):
                return runs_list[choice - 1][1]
        except ValueError:
            pass


# ==============================================================================
# 5. ТОЧКА ВХОДА И ВИЗУАЛИЗАЦИЯ
# ==============================================================================
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path(input("Путь к весам DeepLabV3+: ").strip())
    patient_folder = Path(input("Путь к папке пациента (с DICOMDIR): ").strip())

    model = DeepLabV3Plus(in_channels=1, num_classes=3)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device).eval()

    dicom_path = parse_dicomdir_and_select(patient_folder)
    dcm = pydicom.dcmread(dicom_path)
    pixel_data = dcm.pixel_array
    if pixel_data.ndim == 2:
        pixel_data = np.expand_dims(pixel_data, axis=0)

    pixel_spacing, calib_info = get_isocentric_pixel_spacing(dcm)
    print(f"\nМасштаб: {pixel_spacing:.4f} мм/пикс ({calib_info})")

    # Находим пиковый кадр по площади вероятностной маски
    probs, frames, areas = [], [], []
    for i in range(pixel_data.shape[0]):
        frame = cv2.normalize(pixel_data[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        prob = predict_artery_prob(model, frame, device)
        frames.append(frame)
        probs.append(prob)
        areas.append(int(np.sum(prob >= 0.5)))
        print(f"Сегментация: {i + 1}/{pixel_data.shape[0]}", end="\r")

    peak_idx = int(np.argmax(areas))
    print(f"\nРасчет уравнения Лапласа (Yezzi–Prince) для пикового кадра {peak_idx + 1}...")

    width_mm, phi_map, junc_zone = compute_width_map_laplace(probs[peak_idx], pixel_spacing)
    valid = width_mm[~np.isnan(width_mm)]
    if valid.size == 0:
        print("Ветви достаточной длины не найдены.")
        return

    vmax = float(np.percentile(valid, 99))
    print(f"Медианный диаметр: {np.median(valid):.2f} мм | 99-й перц.: {vmax:.2f} мм")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(frames[peak_idx], cmap="gray")
    axes[0].set_title(f"Ангиограмма (кадр {peak_idx + 1})")
    axes[0].axis("off")

    axes[1].imshow(frames[peak_idx], cmap="gray")
    im_phi = axes[1].imshow(phi_map, cmap="coolwarm", vmin=0.0, vmax=1.0, alpha=0.85)
    axes[1].set_title("Потенциал Лапласа φ (0=левая, 1=правая стенка)")
    axes[1].axis("off")
    fig.colorbar(im_phi, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(frames[peak_idx], cmap="gray")
    im_w = axes[2].imshow(width_mm, cmap="turbo", vmin=0.0, vmax=vmax, alpha=0.85)
    # Контур зон бифуркаций (POC)
    axes[2].contour(junc_zone.astype(float), levels=[0.5], colors="white", linewidths=0.8)
    axes[2].set_title("Ширина артерий (Yezzi–Prince, мм)")
    axes[2].axis("off")
    cbar = fig.colorbar(im_w, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Диаметр (мм)")

    out_dir = Path("heatmap_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "laplace_width_peak.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    print(f"Сохранено в {out_path}")
    plt.show()


if __name__ == "__main__":
    main()