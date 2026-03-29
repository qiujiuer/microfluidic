import os
import glob
import numpy as np
import pandas as pd
import cv2
import tifffile
from sklearn.cluster import KMeans

# ===================== 0) 数据集路径 =====================
dataset_root = r"E:\qiujiuer_data\pycharm_file\patchs\dataset_bright"
images_dir = os.path.join(dataset_root, "images")

geom_dir  = os.path.join(dataset_root, "labels", "geom")
state_dir = os.path.join(dataset_root, "labels", "state")
vis_dir   = os.path.join(dataset_root, "labels", "vis")      # 只保存少量 overlay
debug_dir = os.path.join(dataset_root, "labels", "debug")
ae_patch_root = os.path.join(dataset_root, "labels", "AE_patches")

os.makedirs(geom_dir, exist_ok=True)
os.makedirs(state_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)
os.makedirs(ae_patch_root, exist_ok=True)

# 试跑：先限制数量；确认OK后改成 None
max_images = None   # 例如 100；确认没问题再设 None 跑全部

# =========== vis 输出策略 ===========
save_overlay_every = 50       # 每隔多少张保存一张 overlay
save_first_overlay = True     # 是否保存第一张
save_overlay_if_AE = True     # 只要该图出现 A 或 E，就额外保存 overlay（强烈建议开）

# ===================== 1) 参数（与你单张版一致） =====================
N_ROWS, N_COLS = 20, 20
patch_size = 42

well_radius_um = 50.0
droplet_min_radius_um = 28.0

# 微井 Hough
h_dp = 1.2
h_param1 = 120
h_param2 = 20
h_minRadius = 10
h_maxRadius = 22
h_minDist = 35

# 标签：0 Droplet / 1 Artifact / 2 Empty
LABELS = {
    0: ("Droplet", ""),
    1: ("Artifact", "A"),
    2: ("Empty/Invalid", "E"),
}

# Empty/Invalid（严格）
E_mean_th = 45.0
E_center_mean_th = 35.0
E_std_th = 10.0
E_edge_ratio_th = 0.010

# Artifact（核心连通域 >=2）
center_mask_factor = 0.75
core_factor = 0.85
min_core_area = 10

# 可视化
purple = (180, 0, 180)      # BGR
blue   = (255, 0, 0)        # BGR
text_white = (255, 255, 255)
text_black = (0, 0, 0)
circle_r_factor_of_pitch = 0.30
circle_thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

# ===================== 2) 工具函数 =====================
def to_uint8(x):
    if x.dtype == np.uint8:
        return x
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def read_tif_bgr(path):
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        g8 = to_uint8(arr)
        return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        rgba8 = to_uint8(arr)
        return cv2.cvtColor(rgba8, cv2.COLOR_RGBA2BGR)
    rgb = arr[..., :3]
    rgb8 = to_uint8(rgb)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)

def assign_grid_20x20(pts_xy):
    y = pts_xy[:, 1].reshape(-1, 1)
    km = KMeans(n_clusters=N_ROWS, n_init=10, random_state=0).fit(y)
    row_labels = km.labels_
    row_centers = km.cluster_centers_.flatten()

    order = np.argsort(row_centers)
    label_to_row = {int(old): int(new) for new, old in enumerate(order)}
    row_idx = np.array([label_to_row[int(l)] for l in row_labels])

    grid = [[None] * N_COLS for _ in range(N_ROWS)]
    used = np.zeros(len(pts_xy), dtype=bool)

    for r in range(N_ROWS):
        idxs = np.where(row_idx == r)[0]
        cy = row_centers[order[r]]
        if len(idxs) > N_COLS:
            idxs = idxs[np.argsort(np.abs(pts_xy[idxs, 1] - cy))[:N_COLS]]
        idxs = idxs[np.argsort(pts_xy[idxs, 0])]
        for c, idx in enumerate(idxs[:N_COLS]):
            grid[r][c] = int(idx)
            used[idx] = True

    remaining = np.where(~used)[0]
    for r in range(N_ROWS):
        filled = [i for i in grid[r] if i is not None]
        if len(filled) == N_COLS:
            continue
        cy = row_centers[order[r]]
        cand = remaining[np.argsort(np.abs(pts_xy[remaining, 1] - cy))]
        need = N_COLS - len(filled)
        take = cand[:need]
        merged = np.array(filled + take.tolist(), dtype=int)
        merged = merged[np.argsort(pts_xy[merged, 0])][:N_COLS]
        for c in range(N_COLS):
            grid[r][c] = int(merged[c])
        used[merged] = True
        remaining = np.where(~used)[0]

    uniq = len(set([grid[r][c] for r in range(N_ROWS) for c in range(N_COLS)]))
    if uniq != N_ROWS * N_COLS:
        raise RuntimeError("Grid assignment failed (not 400 unique).")

    centers = np.zeros((N_ROWS, N_COLS, 2), dtype=np.float32)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            centers[r, c] = pts_xy[grid[r][c]]
    return centers

def crop_with_padding(arr, x0, y0, w, h, fill=0):
    H, W = arr.shape[:2]
    x1, y1 = x0 + w, y0 + h
    if arr.ndim == 2:
        out = np.full((h, w), fill, dtype=arr.dtype)
    else:
        out = np.full((h, w, arr.shape[2]), fill, dtype=arr.dtype)

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
    out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return out

def build_global_distance_transform(gray8):
    blur = cv2.GaussianBlur(gray8, (5, 5), 1.2)
    edges = cv2.Canny(blur, 60, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    non_edge = (edges == 0).astype(np.uint8) * 255
    return cv2.distanceTransform(non_edge, cv2.DIST_L2, 5).astype(np.float32)

def empty_score(patch_gray8):
    blur = cv2.GaussianBlur(patch_gray8, (5, 5), 1.2)
    mean = float(patch_gray8.mean())
    std = float(patch_gray8.std())
    edges = cv2.Canny(blur, 60, 150)
    edge_ratio = float((edges > 0).mean())

    mask = np.zeros_like(patch_gray8, dtype=np.uint8)
    rr = int(round((patch_size / 2) * 0.45))
    cv2.circle(mask, (patch_size // 2, patch_size // 2), rr, 1, -1)
    center_mean = float(patch_gray8[mask == 1].mean()) if np.any(mask == 1) else mean

    is_E = (mean <= E_mean_th) or (center_mean <= E_center_mean_th) or ((std <= E_std_th) and (edge_ratio <= E_edge_ratio_th))
    return is_E, {"mean": mean, "std": std, "edge_ratio": edge_ratio, "center_mean": center_mean}

def count_core_components(dist_patch, r_min_px):
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    rr = int(round((patch_size / 2.0) * center_mask_factor))
    cv2.circle(mask, (patch_size // 2, patch_size // 2), rr, 1, -1)

    d = dist_patch * mask.astype(np.float32)
    thr = float(r_min_px) * core_factor
    core = (d >= thr).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.array([])
    return int(np.sum(areas >= min_core_area)) if areas.size else 0

# ===================== 3) 单张处理函数 =====================
def process_one_tif(tif_path, save_overlay_requested: bool):
    image_id = os.path.splitext(os.path.basename(tif_path))[0]

    out_geom   = os.path.join(geom_dir,  f"{image_id}_well_geometry.csv")
    out_labels = os.path.join(state_dir, f"{image_id}_well_labels.csv")
    out_debug  = os.path.join(debug_dir, f"{image_id}_well_debug.csv")
    out_overlay = os.path.join(vis_dir, f"{image_id}_overlay.png")
    this_ae_dir = os.path.join(ae_patch_root, image_id)
    os.makedirs(this_ae_dir, exist_ok=True)

    img_bgr = read_tif_bgr(tif_path)
    gray8 = to_uint8(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    blur = cv2.GaussianBlur(gray8, (5, 5), 1.2)

    well_circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=h_dp, minDist=h_minDist,
        param1=h_param1, param2=h_param2,
        minRadius=h_minRadius, maxRadius=h_maxRadius
    )
    if well_circles is None:
        raise RuntimeError("Well Hough failed")

    well_circles = np.squeeze(well_circles, axis=0)
    if well_circles.ndim == 1:
        well_circles = well_circles.reshape(1, 3)

    pts = well_circles[:, :2].astype(np.float32)
    rads = well_circles[:, 2].astype(np.float32)

    centers = assign_grid_20x20(pts)

    pitch_x = float(np.median(np.diff(np.sort(centers[N_ROWS//2, :, 0]))))
    pitch_y = float(np.median(np.diff(np.sort(centers[:, N_COLS//2, 1]))))
    pitch = float((pitch_x + pitch_y) / 2.0)
    margin = int(round(pitch * 0.6))

    out_w = int(round(margin * 2 + pitch * (N_COLS - 1)))
    out_h = int(round(margin * 2 + pitch * (N_ROWS - 1)))

    src = np.array([centers[0,0], centers[0,N_COLS-1], centers[N_ROWS-1,0], centers[N_ROWS-1,N_COLS-1]], dtype=np.float32)
    dst = np.array([
        [margin, margin],
        [margin + pitch*(N_COLS-1), margin],
        [margin, margin + pitch*(N_ROWS-1)],
        [margin + pitch*(N_COLS-1), margin + pitch*(N_ROWS-1)]
    ], dtype=np.float32)

    Hmat = cv2.getPerspectiveTransform(src, dst)
    aligned = cv2.warpPerspective(img_bgr, Hmat, (out_w, out_h), flags=cv2.INTER_LINEAR)
    aligned_gray8 = to_uint8(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))

    well_r_px = float(np.median(rads))
    um_per_px = well_radius_um / well_r_px
    r_min_px = max(3.0, droplet_min_radius_um / um_per_px)

    dist_global = build_global_distance_transform(aligned_gray8)

    # 只有需要保存 overlay 时才画（省时间）
    overlay = aligned.copy() if save_overlay_requested else None

    half = patch_size // 2
    circle_r = max(3, int(round(pitch * circle_r_factor_of_pitch)))
    font_scale = max(0.35, float(pitch) / 120.0)
    font_th = 2

    geom_rows, label_rows, debug_rows = [], [], []
    A_wells, E_wells = [], []

    for r in range(N_ROWS):
        for c in range(N_COLS):
            cx = int(round(margin + pitch * c))
            cy = int(round(margin + pitch * r))
            wid = f"r{r:02d}c{c:02d}"

            if overlay is not None:
                cv2.rectangle(overlay, (cx-half, cy-half), (cx+half, cy+half), purple, 2)

            x0, y0 = cx - half, cy - half
            patch_g = crop_with_padding(aligned_gray8, x0, y0, patch_size, patch_size, fill=0)
            patch_d = crop_with_padding(dist_global,   x0, y0, patch_size, patch_size, fill=0.0)

            is_E, E_feats = empty_score(patch_g)
            if is_E:
                label_id = 2
                label_name, letter = LABELS[label_id]
                note = "empty"
                E_wells.append(wid)

                if overlay is not None:
                    (tw, thh), _ = cv2.getTextSize(letter, font, font_scale, font_th)
                    org = (cx - tw//2, cy + thh//2)
                    cv2.putText(overlay, letter, org, font, font_scale, text_black, font_th+2, cv2.LINE_AA)
                    cv2.putText(overlay, letter, org, font, font_scale, text_white, font_th, cv2.LINE_AA)

                core_cnt = 0
            else:
                core_cnt = count_core_components(patch_d, r_min_px)
                if core_cnt >= 2:
                    label_id = 1
                    label_name, letter = LABELS[label_id]
                    note = "merged"
                    A_wells.append(wid)

                    if overlay is not None:
                        (tw, thh), _ = cv2.getTextSize(letter, font, font_scale, font_th)
                        org = (cx - tw//2, cy + thh//2)
                        cv2.putText(overlay, letter, org, font, font_scale, text_black, font_th+2, cv2.LINE_AA)
                        cv2.putText(overlay, letter, org, font, font_scale, text_white, font_th, cv2.LINE_AA)
                else:
                    label_id = 0
                    label_name, _ = LABELS[label_id]
                    note = ""
                    if overlay is not None:
                        cv2.circle(overlay, (cx, cy), circle_r, blue, circle_thickness)

            geom_rows.append({
                "well_id": wid, "row": r, "col": c,
                "center_x_px_aligned": float(cx),
                "center_y_px_aligned": float(cy),
                "pitch_px": float(pitch),
                "margin_px": int(margin),
                "patch_size": int(patch_size),
                "well_r_px_median": float(well_r_px),
                "um_per_px": float(um_per_px),
                "droplet_min_r_px": float(r_min_px),
            })

            label_rows.append({
                "well_id": wid,
                "label_id": int(label_id),
                "label_name": label_name,
                "review_flag": 0,
                "note": note
            })

            debug_rows.append({
                "well_id": wid,
                "label_id": int(label_id),
                "core_cnt": int(core_cnt),
                **E_feats
            })

    pd.DataFrame(geom_rows).to_csv(out_geom, index=False, encoding="utf-8-sig")
    pd.DataFrame(label_rows).to_csv(out_labels, index=False, encoding="utf-8-sig")
    pd.DataFrame(debug_rows).to_csv(out_debug, index=False, encoding="utf-8-sig")

    # 如果出现A/E，且你希望强制保存 overlay，则这里再画一次并保存
    need_force_overlay = save_overlay_if_AE and (len(A_wells) + len(E_wells) > 0)
    if overlay is None and need_force_overlay:
        # 重新绘制 overlay（只为这张图）
        overlay = aligned.copy()
        for r in range(N_ROWS):
            for c in range(N_COLS):
                cx = int(round(margin + pitch * c))
                cy = int(round(margin + pitch * r))
                wid = f"r{r:02d}c{c:02d}"
                cv2.rectangle(overlay, (cx-half, cy-half), (cx+half, cy+half), purple, 2)

                # 读回标签（用 label_rows 更快）
                lid = label_rows[r*N_COLS + c]["label_id"]
                if lid == 0:
                    cv2.circle(overlay, (cx, cy), circle_r, blue, circle_thickness)
                else:
                    letter = LABELS[lid][1]
                    (tw, thh), _ = cv2.getTextSize(letter, font, font_scale, font_th)
                    org = (cx - tw//2, cy + thh//2)
                    cv2.putText(overlay, letter, org, font, font_scale, text_black, font_th+2, cv2.LINE_AA)
                    cv2.putText(overlay, letter, org, font, font_scale, text_white, font_th, cv2.LINE_AA)

    if overlay is not None:
        cv2.imwrite(out_overlay, overlay)

    # 导出 A/E patch 供复核
    for wid in A_wells + E_wells:
        rr = int(wid[1:3]); cc = int(wid[4:6])
        cx = int(round(margin + pitch * cc))
        cy = int(round(margin + pitch * rr))
        x0, y0 = cx - half, cy - half
        patch_rgb = crop_with_padding(aligned, x0, y0, patch_size, patch_size, fill=0)
        cv2.imwrite(os.path.join(this_ae_dir, f"{wid}.png"), patch_rgb)

    return image_id, len(A_wells), len(E_wells)

# ===================== 4) 批量跑 =====================
tif_files = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
if max_images is not None:
    tif_files = tif_files[:max_images]

print(f"[INFO] Found {len(tif_files)} tif files to process.")

summary_rows = []
ok, fail = 0, 0

for i, tif_path in enumerate(tif_files, 1):
    image_id = os.path.splitext(os.path.basename(tif_path))[0]

    # 每隔 N 张保存一张 overlay；可选保存第一张
    save_overlay_requested = (save_first_overlay and i == 1) or (save_overlay_every > 0 and i % save_overlay_every == 0)

    try:
        img_id, A_cnt, E_cnt = process_one_tif(tif_path, save_overlay_requested=save_overlay_requested)
        summary_rows.append({"image_id": img_id, "A_count": A_cnt, "E_count": E_cnt, "status": "ok"})
        ok += 1
        print(f"[{i}/{len(tif_files)}] {img_id}: A={A_cnt}, E={E_cnt}, overlay={'YES' if save_overlay_requested else 'no'}")
    except Exception as e:
        summary_rows.append({"image_id": image_id, "A_count": "", "E_count": "", "status": f"fail: {e}"})
        fail += 1
        print(f"[{i}/{len(tif_files)}] {image_id}: FAIL -> {e}")

summary_csv = os.path.join(dataset_root, "labels", "batch_summary.csv")
pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

print("==== BATCH DONE ====")
print("OK:", ok, "FAIL:", fail)
print("Summary:", summary_csv)
