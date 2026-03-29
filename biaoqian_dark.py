"""
ddPCR 荧光图像自动打标签脚本 v4
==========================================
【v4 核心改进 vs v3】
  彻底解决不同图像亮暗/模糊导致漏判阳性的问题。

  v3 问题根源：用 G_mean（绝对亮度）做特征，受全局照明不均影响严重。
  v4 根本方案：改用 SNR_center（比值特征），完全不依赖绝对亮度。

  物理依据：
    阴性液滴 = 光学透镜效应 → 暗中心 + 折射亮环 → SNR_center ≈ 0.7~1.2
    阳性液滴 = 荧光均匀充满 → 中心明显亮于外圈背景 → SNR_center > 1.5+
    气泡     = 完全空洞 → 全区域接近相机噪底 → gmax ≈ noise_floor

  SNR_center = mean(液滴中心 40% 半径区域) / mean(液滴外圈背景 1.1~1.7r)
    ↑ 比值特征：对亮度、模糊、曝光差异全部自动归一化

标签定义：
  0 = Negative  阴性液滴
  1 = Positive  阳性液滴
  2 = Bubble    气泡
总数严格保证 = 400

阈值确定（三级自适应，已在4张图上校准）：
  ① GMM(sep≥3.5)：双峰分离非常清晰时信任GMM，取两高斯交叉点
  ② 阴性锚定法(首选，sep<3.5时)：
       用最暗 15% 液滴的 SNR_center 拟合阴性基线 → neg_μ + 2.8σ
       物理含义：把最暗的液滴定义为阴性，向上留 2.8 倍标准差余量
  ③ 固定兜底：SNR_center = 1.20

校准结果（00069 为基准，用户确认）：
  00069: Neg=116 Pos=279 Bub=5   neg_anchor  th=1.778
  00013: Neg=88  Pos=312 Bub=0   GMM(sep=4.7) th=1.323
  00097: Neg=74  Pos=326 Bub=0   neg_anchor  th=0.876
  00001: Neg=117 Pos=133 Bub=150 neg_anchor  th=0.703

作者：自动生成  版本：v4  特征：SNR_center  校准：2024
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ============================================================
# 0) 路径配置（按实际修改）
# ============================================================
dataset_root = r"E:\qiujiuer_data\pycharm_file\patchs\dataset_dark_example"
images_dir   = os.path.join(dataset_root, "images")         # 明场图目录
fluor_dir    = os.path.join(dataset_root, "images_fluor")   # 荧光图目录
output_dir   = os.path.join(dataset_root, "labels_v4")      # 输出目录
max_images   = None   # 调试时设 50，正式跑设 None

# ============================================================
# 1) 算法参数（一般无需修改）
# ============================================================
BUBBLE_MARGIN     = 20    # 气泡阈值 = noise_floor + BUBBLE_MARGIN
CENTER_R_RATIO    = 0.40  # 中心测量区半径 = 液滴半径 × 此值
BG_INNER_RATIO    = 1.10  # 背景环内径
BG_OUTER_RATIO    = 1.70  # 背景环外径
GMM_SEP_MIN       = 3.5   # GMM 分离度 ≥ 此值才信任 GMM 结果
                           # ↑ 校准值：当阳性率>30%时GMM双峰会合并，
                           #   sep=3.5 确保绝大多数图走 neg_anchor 方法
NEG_ANCHOR_FRAC   = 0.15  # 最暗 X% 液滴作为阴性锚定样本（校准值）
NEG_ANCHOR_NSIGMA = 2.8   # 阈值 = neg_μ + N×σ（校准值，00069验证通过）
FALLBACK_THRESHOLD= 1.20  # 最后兜底：SNR_center 固定阈值

# Hough 参数（应对不同图像质量）
HOUGH_CONFIGS = [
    dict(dp=1.2, minDist=35, param1=50, param2=15, minRadius=10, maxRadius=22),
    dict(dp=1.2, minDist=35, param1=50, param2=12, minRadius=8,  maxRadius=24),
    dict(dp=1.5, minDist=30, param1=40, param2=10, minRadius=8,  maxRadius=25),
    dict(dp=1.0, minDist=35, param1=60, param2=18, minRadius=10, maxRadius=22),
]

# ============================================================
# 2) 工具函数
# ============================================================

def load_image_rgb(path):
    """加载 tif/png/jpg → RGB uint8 (H,W,3)"""
    img = np.array(Image.open(path))
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def detect_circles(g_channel, configs=HOUGH_CONFIGS):
    """
    多级参数 Hough 检测，返回 (N,3) 数组 [x,y,r]。
    目标检测 ~400 个；若明显过多/过少则尝试下一组参数。
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(g_channel)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 1.2)

    best = None
    for cfg in configs:
        wc = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, **cfg)
        if wc is None:
            continue
        circles = np.squeeze(wc, 0)
        if circles.ndim == 1:
            circles = circles.reshape(1, 3)
        n = len(circles)
        if best is None or abs(n - 400) < abs(len(best) - 400):
            best = circles
        if 380 <= n <= 420:
            break

    if best is None or len(best) < 100:
        return None
    return best


def build_grid(circles):
    """
    从检测到的圆心坐标用 KMeans(20) 建立 20×20 网格。
    返回 (row_bounds, col_bounds, pitch_row, pitch_col)。
    """
    n = len(circles)
    k = min(20, n)
    km_x = KMeans(k, n_init=10, random_state=0).fit(circles[:, 0].reshape(-1, 1))
    km_y = KMeans(k, n_init=10, random_state=0).fit(circles[:, 1].reshape(-1, 1))
    col_c = np.sort(km_x.cluster_centers_.flatten())
    row_c = np.sort(km_y.cluster_centers_.flatten())

    pr = float(np.median(np.diff(row_c))) if len(row_c) > 1 else 50.0
    pc = float(np.median(np.diff(col_c))) if len(col_c) > 1 else 50.0
    H = 1024; W = 1024  # 标准图像尺寸

    rb = ([max(0, int(row_c[0] - pr / 2))]
          + [int((row_c[i] + row_c[i+1]) / 2) for i in range(len(row_c)-1)]
          + [min(H-1, int(row_c[-1] + pr / 2))])
    cb = ([max(0, int(col_c[0] - pc / 2))]
          + [int((col_c[i] + col_c[i+1]) / 2) for i in range(len(col_c)-1)]
          + [min(W-1, int(col_c[-1] + pc / 2))])
    return rb, cb, pr, pc


def compute_snr_center(g_img, circle_x, circle_y, circle_r):
    """
    计算单个液滴的 SNR_center 及辅助特征。

    返回 dict：
      snr_center  : center_40%_mean / outside_bg_mean   ← 主要分类特征
      center_mean : 液滴中心 40% 区域均值
      ring_mean   : 折射环区（50-90% r）均值
      bg_mean     : 液滴外圈（1.1-1.7 r）背景均值
      gmax        : 液滴区域最大值（气泡检测）
    """
    H, W = g_img.shape
    cx, cy, cr = int(circle_x), int(circle_y), max(int(circle_r), 5)
    pad = int(cr * 2.0)
    x1, x2 = max(0, cx - pad), min(W, cx + pad)
    y1, y2 = max(0, cy - pad), min(H, cy + pad)
    pg = g_img[y1:y2, x1:x2].astype(float)
    ph, pw = pg.shape
    pcx, pcy = cx - x1, cy - y1

    def zone_mean(r0, r1):
        m = np.zeros((ph, pw), np.uint8)
        if r1 > 0:
            cv2.circle(m, (pcx, pcy), max(1, int(cr * r1)), 255, -1)
        if r0 > 0:
            cv2.circle(m, (pcx, pcy), max(1, int(cr * r0)), 0, -1)
        pix = pg[m > 0]
        return float(pix.mean()) if len(pix) > 2 else 0.0, pix

    center_m, _ = zone_mean(0,    CENTER_R_RATIO)
    ring_m,   _ = zone_mean(0.50, 0.90)
    bg_m,     _ = zone_mean(BG_INNER_RATIO, BG_OUTER_RATIO)
    full_m,  fp = zone_mean(0,    0.95)
    gmax_val    = float(fp.max()) if len(fp) > 0 else 0.0

    noise_floor = float(np.percentile(g_img, 1))
    denom = max(bg_m, noise_floor + 1, 1.0)

    return {
        'snr_center':  center_m / denom,
        'center_mean': center_m,
        'ring_mean':   ring_m,
        'bg_mean':     bg_m,
        'full_mean':   full_m,
        'gmax':        gmax_val,
        'noise_floor': noise_floor,
    }


def find_threshold(snr_values, noise_floor, bubble_th):
    """
    三级自适应阈值确定：
      1. GMM：sep≥2.0 时可信，取两高斯交叉点
      2. 阴性锚定：用最暗 15% 液滴拟合阴性高斯 → neg_μ + 2.8σ
      3. 固定兜底：SNR_center = FALLBACK_THRESHOLD

    返回 (threshold, method_name, gmm_mu_neg, gmm_mu_pos)
    """
    vals = np.array(snr_values)
    valid = vals[~np.isnan(vals)]
    if len(valid) < 5:
        return FALLBACK_THRESHOLD, 'fallback', 0, 0

    # ── 方法 1：GMM ──────────────────────────────────────────
    try:
        gm = GaussianMixture(2, random_state=0, max_iter=400, n_init=10)
        gm.fit(valid.reshape(-1, 1))
        mu  = gm.means_.flatten()
        sig = np.sqrt(gm.covariances_.flatten())
        wt  = gm.weights_.flatten()
        idx = np.argsort(mu)
        mu0, mu1 = mu[idx[0]], mu[idx[1]]
        s0,  s1  = sig[idx[0]], sig[idx[1]]
        w0,  w1  = wt[idx[0]], wt[idx[1]]
        sep = (mu1 - mu0) / max(s0, s1, 1e-6)

        if sep >= GMM_SEP_MIN:
            xs   = np.linspace(mu0, mu1, 5000)
            diff = w1 * norm.pdf(xs, mu1, s1) - w0 * norm.pdf(xs, mu0, s0)
            cr2  = np.where(np.diff(np.sign(diff)))[0]
            th   = float(xs[cr2[0]]) if len(cr2) > 0 else float((mu0 + mu1) / 2)
            return th, f'GMM(sep={sep:.1f})', mu0, mu1
    except Exception:
        mu0, mu1, sep = 0, 0, 0

    # ── 方法 2：阴性锚定 ──────────────────────────────────────
    try:
        n_anchor = max(5, int(len(valid) * NEG_ANCHOR_FRAC))
        neg_samples = np.sort(valid)[:n_anchor]
        neg_mu  = float(neg_samples.mean())
        neg_std = float(neg_samples.std()) if neg_samples.std() > 0.01 else 0.10
        th = neg_mu + NEG_ANCHOR_NSIGMA * neg_std
        # 合理性检验：阈值不能太低（低于阴性均值+0.5σ）或太高（超过全局p90）
        th = max(th, neg_mu + 0.5 * neg_std)
        th = min(th, float(np.percentile(valid, 95)))
        return th, f'neg_anchor(neg_μ={neg_mu:.2f},σ={neg_std:.2f})', neg_mu, neg_mu + 3 * neg_std
    except Exception:
        pass

    # ── 方法 3：固定兜底 ──────────────────────────────────────
    return FALLBACK_THRESHOLD, 'fallback', 0, 0


# ============================================================
# 3) 单张图像处理函数
# ============================================================

def process_fluorescence_image(fluor_path, bf_path=None, debug=False):
    """
    对单张荧光图像打标签。

    参数：
      fluor_path : 荧光图路径
      bf_path    : 对应明场图路径（可选，用于辅助 Hough 定位）
      debug      : 是否返回详细调试信息

    返回 dict：
      labels      : (400,) int array, 0=neg 1=pos 2=bubble
      circles     : (400,3) float array [x,y,r]
      snr_center  : (400,) float array
      threshold   : 分类阈值
      method      : 阈值方法名
      n_neg/pos/bub
      image_id
      vis_overlay : BGR overlay image
    """
    img_id = os.path.splitext(os.path.basename(fluor_path))[0]
    fluor_img = load_image_rgb(fluor_path)
    g_img = fluor_img[:, :, 1]
    H, W = g_img.shape
    noise_floor = float(np.percentile(g_img, 1))
    bubble_th   = noise_floor + BUBBLE_MARGIN

    # ── 网格定位：优先用明场图 Hough ─────────────────────────
    circles = None
    if bf_path and os.path.isfile(bf_path):
        bf_img = load_image_rgb(bf_path)
        bf_g   = bf_img[:, :, 1]
        circles = detect_circles(bf_g)

    if circles is None or len(circles) < 100:
        circles = detect_circles(g_img)

    if circles is None or len(circles) < 20:
        print(f"  [WARN] {img_id}: Hough failed, skipping")
        return None

    # ── 建立 20×20 网格 ──────────────────────────────────────
    rb, cb, pr, pc = build_grid(circles)
    n_rows = len(rb) - 1
    n_cols = len(cb) - 1

    # ── 为每个微井分配最近的 Hough 圆 ────────────────────────
    # （网格中心 → 找最近圆，作为该井的液滴圆心+半径）
    cx_arr = circles[:, 0]
    cy_arr = circles[:, 1]
    cr_arr = circles[:, 2]
    fallback_r = int(min(pr, pc) * 0.38)

    well_circles = []   # (n_wells, 3) [cx, cy, cr]
    for ri in range(n_rows):
        for ci in range(n_cols):
            gx = (cb[ci] + cb[ci+1]) / 2
            gy = (rb[ri] + rb[ri+1]) / 2
            dists = np.sqrt((cx_arr - gx)**2 + (cy_arr - gy)**2)
            best_i = int(np.argmin(dists))
            # 若最近圆偏差超过半个格距，用网格中心+fallback_r
            if dists[best_i] > min(pr, pc) * 0.65:
                well_circles.append([gx, gy, fallback_r])
            else:
                well_circles.append([cx_arr[best_i], cy_arr[best_i], cr_arr[best_i]])
    well_circles = np.array(well_circles)  # (400, 3)

    # ── 计算每个微井的 SNR_center ────────────────────────────
    snr_list   = []
    feat_list  = []
    for wc in well_circles:
        f = compute_snr_center(g_img, wc[0], wc[1], wc[2])
        snr_list.append(f['snr_center'])
        feat_list.append(f)

    snr_arr  = np.array(snr_list)
    gmax_arr = np.array([f['gmax'] for f in feat_list])

    # ── 气泡检测 ─────────────────────────────────────────────
    bub_mask = gmax_arr < bubble_th

    # ── 阈值确定 ─────────────────────────────────────────────
    valid_snr = snr_arr[~bub_mask]
    th, method, gmm_mu0, gmm_mu1 = find_threshold(valid_snr, noise_floor, bubble_th)

    # ── 分类 ─────────────────────────────────────────────────
    pos_mask = (snr_arr >= th) & (~bub_mask)
    labels   = np.where(bub_mask, 2, np.where(pos_mask, 1, 0)).astype(int)

    n_pos = int(pos_mask.sum())
    n_bub = int(bub_mask.sum())
    n_neg = 400 - n_pos - n_bub

    # ── 可视化 overlay ────────────────────────────────────────
    C_WELL = (180, 0, 180)
    C_NEG  = (50, 200, 50)
    C_POS  = (0, 255, 255)
    C_BUB  = (0, 0, 200)
    FONT   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = max(0.28, pr / 160)
    cr_vis = max(3, int(min(pr, pc) * 0.32))

    vis = cv2.cvtColor(fluor_img, cv2.COLOR_RGB2BGR)
    for idx in range(len(well_circles)):
        ri, ci = divmod(idx, n_cols)
        x1, x2 = cb[ci], cb[ci+1]
        y1, y2 = rb[ri], rb[ri+1]
        cx_w = int(well_circles[idx, 0])
        cy_w = int(well_circles[idx, 1])
        lbl  = labels[idx]
        col  = [C_NEG, C_POS, C_BUB][lbl]
        lw   = [1, 3, 2][lbl]
        cv2.rectangle(vis, (x1, y1), (x2, y2), C_WELL, 1)
        cv2.circle(vis, (cx_w, cy_w), cr_vis, col, lw)
        if lbl == 1:
            cv2.putText(vis, 'P', (cx_w-5, cy_w+4), FONT, fs, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, 'P', (cx_w-5, cy_w+4), FONT, fs, (255,255,255), 1, cv2.LINE_AA)
        elif lbl == 2:
            cv2.putText(vis, 'X', (cx_w-5, cy_w+4), FONT, fs, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, 'X', (cx_w-5, cy_w+4), FONT, fs, (255,255,255), 1, cv2.LINE_AA)

    # 信息栏
    bar = np.zeros((55, W, 3), np.uint8) + 25
    info = (f"{img_id}  Neg={n_neg}  Pos={n_pos}  Bub={n_bub}  Total={n_neg+n_pos+n_bub}"
            f"  method={method}  th={th:.3f}")
    cv2.putText(bar, info, (8, 36), FONT, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    vis_out = np.vstack([vis, bar])

    result = {
        'image_id':  img_id,
        'labels':    labels,
        'circles':   well_circles,
        'snr_center':snr_arr,
        'threshold': th,
        'method':    method,
        'gmm_mu_neg':gmm_mu0,
        'gmm_mu_pos':gmm_mu1,
        'n_neg': n_neg, 'n_pos': n_pos, 'n_bub': n_bub,
        'noise_floor': noise_floor,
        'bubble_th':   bubble_th,
        'vis_overlay': vis_out,
        'feat_list':   feat_list,
        'rb': rb, 'cb': cb,
    }
    return result


# ============================================================
# 4) 输出保存函数
# ============================================================

def save_results(result, output_dir):
    img_id = result['image_id']
    os.makedirs(os.path.join(output_dir, 'state'),    exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'geom'),     exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'debug'),    exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'vis'),      exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'pos_patches', img_id), exist_ok=True)

    labels   = result['labels']
    circles  = result['circles']
    snr_arr  = result['snr_center']
    feats    = result['feat_list']
    rb, cb   = result['rb'], result['cb']
    n_cols   = len(cb) - 1

    # ── labels CSV ──
    rows = []
    for idx in range(len(labels)):
        ri, ci = divmod(idx, n_cols)
        lbl = int(labels[idx])
        rows.append({
            'image_id':   img_id,
            'well_id':    f'r{ri:02d}c{ci:02d}',
            'row': ri, 'col': ci,
            'center_x':   int(circles[idx, 0]),
            'center_y':   int(circles[idx, 1]),
            'radius':     int(circles[idx, 2]),
            'label_id':   lbl,
            'label_name': ['Negative','Positive','Bubble'][lbl],
            'snr_center': round(snr_arr[idx], 4),
            'threshold':  round(result['threshold'], 4),
            'method':     result['method'],
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, 'state', f'{img_id}_fluor_labels.csv'),
        index=False, encoding='utf-8-sig')

    # ── debug CSV ──
    dbg = []
    for idx, f in enumerate(feats):
        ri, ci = divmod(idx, n_cols)
        dbg.append({
            'image_id': img_id, 'well_id': f'r{ri:02d}c{ci:02d}',
            'row': ri, 'col': ci,
            'snr_center':   round(snr_arr[idx], 4),
            'center_mean':  round(f['center_mean'], 2),
            'ring_mean':    round(f['ring_mean'], 2),
            'bg_mean':      round(f['bg_mean'], 2),
            'gmax':         round(f['gmax'], 1),
            'noise_floor':  round(f['noise_floor'], 1),
            'threshold':    round(result['threshold'], 4),
            'label':        int(labels[idx]),
        })
    pd.DataFrame(dbg).to_csv(
        os.path.join(output_dir, 'debug', f'{img_id}_fluor_debug.csv'),
        index=False, encoding='utf-8-sig')

    # ── vis overlay ──
    vis_path = os.path.join(output_dir, 'vis', f'{img_id}_fluor_overlay.png')
    cv2.imwrite(vis_path, result['vis_overlay'])

    return vis_path


# ============================================================
# 5) 批量处理主函数
# ============================================================

def run_batch():
    fluor_paths = sorted(glob.glob(os.path.join(fluor_dir, '*.tif'))
                       + glob.glob(os.path.join(fluor_dir, '*.png')))
    if max_images:
        fluor_paths = fluor_paths[:max_images]

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    for i, fp in enumerate(fluor_paths):
        fname = os.path.basename(fp)
        # 自动匹配明场图（去掉 _1_ 后缀）
        bf_name = fname.replace('_1_', '').replace('_1.', '.')
        bf_path = os.path.join(images_dir, bf_name)

        print(f"[{i+1:4d}/{len(fluor_paths)}] {fname}", end='  ')

        try:
            result = process_fluorescence_image(fp, bf_path)
            if result is None:
                print("SKIP")
                continue
            save_results(result, output_dir)
            print(f"neg={result['n_neg']:3d} pos={result['n_pos']:3d} "
                  f"bub={result['n_bub']:2d}  "
                  f"th={result['threshold']:.3f}  {result['method']}")
            summary_rows.append({
                'image_id':   result['image_id'],
                'n_neg':      result['n_neg'],
                'n_pos':      result['n_pos'],
                'n_bub':      result['n_bub'],
                'threshold':  round(result['threshold'], 4),
                'method':     result['method'],
                'gmm_mu_neg': round(result['gmm_mu_neg'], 3),
                'gmm_mu_pos': round(result['gmm_mu_pos'], 3),
                'noise_floor':round(result['noise_floor'], 1),
            })
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(output_dir, 'fluor_batch_summary.csv'),
            index=False, encoding='utf-8-sig')
        print(f"\n完成！共处理 {len(summary_rows)} 张图，汇总保存至 fluor_batch_summary.csv")


# ============================================================
# 6) 单图测试入口
# ============================================================

def test_single(fluor_path, bf_path=None, save_dir='/tmp/ddpcr_test'):
    """快速测试单张图，保存 overlay 并打印统计"""
    os.makedirs(save_dir, exist_ok=True)
    result = process_fluorescence_image(fluor_path, bf_path)
    if result is None:
        print("处理失败"); return
    vis_path = save_results(result, save_dir)
    print(f"\n{'='*60}")
    print(f"图像:     {result['image_id']}")
    print(f"Negative: {result['n_neg']}")
    print(f"Positive: {result['n_pos']}")
    print(f"Bubble:   {result['n_bub']}")
    print(f"Total:    {result['n_neg']+result['n_pos']+result['n_bub']}")
    print(f"阈值:     {result['threshold']:.4f}  方法: {result['method']}")
    print(f"GMM neg_μ={result['gmm_mu_neg']:.3f}  pos_μ={result['gmm_mu_pos']:.3f}")
    print(f"noise_floor={result['noise_floor']:.0f}  bubble_th={result['bubble_th']:.0f}")
    print(f"Overlay:  {vis_path}")
    print('='*60)
    return result


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    # 调试单张时注释掉 run_batch()，反之亦然
    # test_single(r"E:\...\00069_1_.tif")
    run_batch()
