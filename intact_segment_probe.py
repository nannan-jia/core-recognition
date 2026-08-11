"""
完整岩段探针（新脚本）

灰度三档 + 块上色 + 缝画线：
1. 黑 / 灰 / 白 分档（仅作描述与可视化）
2. 行级碎裂纹理：整行密实碎块则短路，不再找主岩块/裂缝
3. 只有「贯穿岩柱高度的更黑带」才当作断口切开（避免岩内深色花纹误切）
4. 切开后的每一段：灰+白为主、外形像圆柱面、且纹理不像碎堆 → 完整段彩色覆盖
5. 段间暗点拟合可倾斜直线 → 裂缝红线（限制在段间局部，避免飞线）

OpenCV + NumPy
输入：input/<箱号>/row_*.jpg 或 input/<箱号>_row_*.jpg
输出：output/<孔号>/<箱号>/<row>/*_intact.jpg、*_levels.jpg
      例如 ZK1031-06/row_01 → output/ZK1031/ZK1031-06/row_01/
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 箱号形如 ZK1031-06 / ZK1024A-1；孔号取其连字符前一段 ZK1031 / ZK1024A
SAMPLE_RE = re.compile(r"^(ZK\d+[A-Za-z]?-\d+)", re.IGNORECASE)
ROW_RE = re.compile(r"(row_\d+)", re.IGNORECASE)

EDGE_MARGIN = 20
MIN_SEG_WIDTH_FRAC = 0.035
MIN_GAP_WIDTH = 4
MIN_GAP_SEP = 80
# 贯穿：该列暗像素占岩柱高度的比例（切开完整段用）
SPAN_MIN = 0.38
# 相对邻域更暗
PROM_MIN = 10.0
# 段内连通域：明显切开后只留最大主岩块（丢掉断口后游离小尖）
MIN_CC_AREA_FRAC = 0.25
MIN_CC_X_OVERLAP = 0.35

# 行级/段级碎裂：纹理 + 红棕土状色调门闩
# 仅纹理会把「深灰+白脉柱面」误判成全碎，故必须像土红碎屑才短路
ROW_FRAG_LAP_MIN = 95.0
ROW_FRAG_TEX_MIN = 20.5
ROW_FRAG_EDGE_MIN = 0.22
ROW_FRAG_GRAINY_MIN = 0.75
ROW_FRAG_SMOOTH_MAX = 0.10
ROW_FRAG_RED_MIN = 0.32
ROW_FRAG_WHITE_MAX = 0.22  # 白脉过多时不当全碎短路
SEG_FRAG_LAP_MIN = 105.0
SEG_FRAG_TEX_MIN = 22.0
SEG_FRAG_EDGE_MIN = 0.24
SEG_FRAG_RED_MIN = 0.30


def distractor_mask(bgr: np.ndarray) -> np.ndarray:
    """只屏蔽红油漆字和较大白标签，避免把岩体高光/色斑整片抹掉。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 红油漆：高饱和、够亮
    red = ((h <= 8) | (h >= 172)) & (s >= 120) & (v >= 90)
    red = red.astype(np.uint8) * 255
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # 白标签：很亮且低饱和；再按连通域面积过滤，去掉岩表高光碎点
    white = ((s <= 35) & (v >= 210)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    white_keep = np.zeros_like(white)
    min_label_area = int(bgr.shape[0] * bgr.shape[1] * 0.0008)
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= min_label_area:
            white_keep[labels == lab] = 255

    mask = cv2.bitwise_or(red, white_keep)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def estimate_core_band(gray: np.ndarray, distract: np.ndarray) -> tuple[int, int]:
    h, w = gray.shape
    score = np.zeros(h, np.float32)
    for y in range(h):
        valid = distract[y] == 0
        if valid.sum() < w * 0.25:
            continue
        vals = gray[y, valid]
        score[y] = float(vals.std() + 0.35 * vals.mean())
    if not np.any(score > 0):
        return int(h * 0.1), int(h * 0.9)
    thr = float(np.percentile(score[score > 0], 35))
    band = score >= thr
    best = (int(h * 0.1), int(h * 0.9))
    y = 0
    while y < h:
        if not band[y]:
            y += 1
            continue
        y0 = y
        while y < h and band[y]:
            y += 1
        if y - y0 > best[1] - best[0]:
            best = (y0, y)
    pad = max(1, (best[1] - best[0]) // 25)
    return best[0] + pad, best[1] - pad


def moving_mean(x: np.ndarray, k: int) -> np.ndarray:
    k = max(3, k | 1)
    pad = k // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid").astype(np.float32)


def analyze_columns(
    gray: np.ndarray, distract: np.ndarray, y0: int, y1: int
) -> dict[str, np.ndarray | float]:
    band = gray[y0:y1, :].astype(np.float32)
    label_col = distract[y0:y1].mean(axis=0) > 30
    col = band.mean(axis=0)
    col_min = band.min(axis=0)

    # 先用粗阈值估“非空列”，再用这些列估岩体中位，避免断口把中位拉低
    rough_dark = float(np.percentile(col[~label_col], 15)) if np.any(~label_col) else 40.0
    solid = (~label_col) & (col > rough_dark + 10)
    rock_med = float(np.median(col[solid])) if np.any(solid) else float(np.median(col))

    # 断口阈值：明显暗于岩体中位；白脉用岩体区高分位
    black_thr = max(8.0, min(rock_med * 0.68, rock_med - 22.0))
    white_thr = float(np.percentile(band[:, solid], 80)) if np.any(solid) else rock_med + 40.0
    white_thr = min(250.0, max(white_thr, rock_med + 25.0))

    span = (band < black_thr).mean(axis=0)
    span[label_col] = 0

    local_bg = moving_mean(col, 91)
    prom = local_bg - col
    prom[label_col] = 0

    # 贯穿黑缝得分（宽而黑）
    gap_score = np.clip((black_thr - col) / (black_thr + 1e-6), 0, 1) * span
    gap_score = gap_score * np.clip(prom / 25.0, 0.35, 1.5)
    gap_score[label_col] = 0
    gap_score[:EDGE_MARGIN] = 0
    gap_score[-EDGE_MARGIN:] = 0

    # 细缝/闭合缝：黑帽 + 竖向边缘 + 列最小很暗
    blackhat = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
    )
    bh_col = blackhat[y0:y1].mean(axis=0).astype(np.float32)
    sob = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))[y0:y1].mean(axis=0)
    sob = sob.astype(np.float32)
    bh_n = bh_col / (np.percentile(bh_col, 95) + 1e-6)
    sob_n = sob / (np.percentile(sob, 95) + 1e-6)
    thin = np.clip(bh_n, 0, 1.5) * 0.55 + np.clip(sob_n, 0, 1.5) * 0.45
    # 需要至少有一点“更暗”的迹象，避免纯岩性条带
    thin = thin * np.clip((rock_med - col_min) / 40.0, 0.2, 1.3)
    thin[label_col] = 0
    thin[:EDGE_MARGIN] = 0
    thin[-EDGE_MARGIN:] = 0

    return {
        "col": col.astype(np.float32),
        "col_min": col_min.astype(np.float32),
        "span": span.astype(np.float32),
        "prom": prom.astype(np.float32),
        "gap_score": gap_score.astype(np.float32),
        "thin_score": thin.astype(np.float32),
        "label_col": label_col.astype(np.uint8),
        "rock_med": rock_med,
        "black_thr": black_thr,
        "white_thr": white_thr,
    }


def _merge_centers(centers: list[int], scores: np.ndarray, min_sep: int) -> list[int]:
    centers = sorted(centers, key=lambda x: scores[x], reverse=True)
    kept: list[int] = []
    for x in centers:
        if all(abs(x - k) >= min_sep for k in kept):
            kept.append(x)
    return sorted(kept)


def find_gap_intervals(sig: dict) -> list[tuple[int, int]]:
    """只返回用于切开完整段的强贯穿黑缝；细缝另算，避免岩性条带切碎。"""
    gap_score = sig["gap_score"]
    thin_score = sig["thin_score"]
    span = sig["span"]
    prom = sig["prom"]
    col_min = sig["col_min"]
    black_thr = float(sig["black_thr"])
    w = len(gap_score)

    strong_thr = max(
        0.08,
        float(np.percentile(gap_score[gap_score > 0], 65)) if np.any(gap_score > 0) else 0.12,
    )
    binary = (gap_score >= strong_thr) & (span >= SPAN_MIN) & (prom >= PROM_MIN)

    gaps: list[tuple[int, int]] = []
    x = 0
    while x < w:
        if not binary[x]:
            x += 1
            continue
        x0 = x
        while x < w and binary[x]:
            x += 1
        if x - x0 >= 3:
            gaps.append((x0, x))
        elif gap_score[x0:x].max() >= strong_thr * 1.25:
            peak = int(x0 + np.argmax(gap_score[x0:x]))
            gaps.append((max(0, peak - 4), min(w, peak + 4)))
        x = max(x, x0 + 1)

    # 仅提升“几乎贯穿 + 很暗”的细缝峰为切开点
    thin_thr = max(0.65, float(np.percentile(thin_score, 98)))
    thin_peaks = []
    for i in range(2, w - 2):
        if thin_score[i] < thin_thr:
            continue
        if thin_score[i] >= thin_score[i - 1] and thin_score[i] >= thin_score[i + 1]:
            thin_peaks.append(i)
    for p in _merge_centers(thin_peaks, thin_score, MIN_GAP_SEP):
        if any(abs(p - (a + b) // 2) < MIN_GAP_SEP for a, b in gaps):
            continue
        if span[p] >= 0.32 and (col_min[p] <= black_thr or prom[p] >= 12):
            gaps.append((max(0, p - 4), min(w, p + 4)))

    if not gaps:
        return []

    gaps = sorted(gaps)
    merged = [gaps[0]]
    for a, b in gaps[1:]:
        pa, pb = merged[-1]
        if a - pb < MIN_GAP_SEP:
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))

    refined = []
    for a, b in merged:
        if b - a <= 100:
            refined.append((a, b))
            continue
        local = gap_score[a:b] + 0.25 * thin_score[a:b]
        peak = int(a + np.argmax(local))
        refined.append((max(a, peak - 10), min(b, peak + 10)))
    return refined


def intervals_between(width: int, gaps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    cur = 0
    for a, b in gaps:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < width:
        out.append((cur, width))
    return out


def keep_primary_components(
    mask: np.ndarray,
    min_area_frac: float = MIN_CC_AREA_FRAC,
    min_x_overlap: float = MIN_CC_X_OVERLAP,
) -> np.ndarray:
    """只保留面积最大的主岩块连通域。

    铺色 mask 上若已明显切开成多块，丢掉其余小块
   （例如断口后侧向粘连/投影略有重叠的尾部碎尖）。
    min_area_frac / min_x_overlap 保留兼容调用方，现不再用重叠回填次块。
    """
    del min_area_frac, min_x_overlap
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    main_i = int(np.argmax(areas)) + 1
    keep = np.zeros_like(mask)
    keep[labels == main_i] = 255
    return keep


def trim_to_solid_span(
    mask: np.ndarray,
    ref: np.ndarray | None = None,
    min_cov: float = 0.15,
    min_run: int = 20,
    min_valley: int = 4,
) -> np.ndarray:
    """按列覆盖保留主实心跨度。

    只在当前 mask 有岩体的水平范围内操作：保留最长的一段高覆盖跨度，
    丢掉其后（或之前）被低谷隔开的小尖。不用整段 raw 从左盲目切断，
    避免同段左侧碎区把右侧主块剪空。
    """
    del ref, min_run, min_valley  # 兼容旧调用；跨度以 mask 自身为准
    if mask.size == 0 or not np.any(mask):
        return mask

    mask_cov = (mask > 0).mean(axis=0).astype(np.float32)
    w = len(mask_cov)
    mcols = np.where(mask_cov > 0)[0]
    if mcols.size == 0:
        return mask
    m0 = int(mcols.min())
    m1 = int(mcols.max()) + 1

    def longest_run(thr: float) -> tuple[int, int] | None:
        best = None
        j = m0
        while j < m1:
            if mask_cov[j] < thr:
                j += 1
                continue
            a = j
            while j < m1 and mask_cov[j] >= thr:
                j += 1
            if best is None or j - a > best[0]:
                best = (j - a, a, j)
        if best is None:
            return None
        return best[1], best[2]

    span = longest_run(min_cov)
    if span is None:
        span = longest_run(0.05)
    if span is None:
        return keep_primary_components(mask)

    solid_start, cut = span
    out = np.zeros_like(mask)
    out[:, solid_start:cut] = mask[:, solid_start:cut]
    if not np.any(out):
        return keep_primary_components(mask)
    return keep_primary_components(out)


def segment_mask_in_interval(
    gray: np.ndarray,
    distract: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    black_thr: float,
) -> np.ndarray:
    """段内灰+白（非成片黑）像素构成的柱面 mask。"""
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    roi = gray[y0:y1, x0:x1]
    dist = distract[y0:y1, x0:x1]
    raw = ((roi >= black_thr) & (dist == 0)).astype(np.uint8) * 255
    # 填小洞，成近似圆柱面
    rock = cv2.morphologyEx(
        raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2
    )
    rock = cv2.morphologyEx(
        rock, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    # 腐蚀打断细桥，只留主连通域再有限恢复；再用原始列覆盖剪掉断口后小尖
    neck_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(rock, neck_k, iterations=1)
    primary = keep_primary_components(eroded)
    restored = cv2.bitwise_and(cv2.dilate(primary, neck_k, iterations=1), rock)
    restored = cv2.morphologyEx(
        restored, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    restored = keep_primary_components(restored)
    restored = trim_to_solid_span(restored, ref=raw)
    mask[y0:y1, x0:x1] = restored
    return mask


def fragment_texture_metrics(
    gray: np.ndarray,
    distract: np.ndarray,
    y0: int,
    y1: int,
    x0: int | None = None,
    x1: int | None = None,
    bgr: np.ndarray | None = None,
    black_thr: float | None = None,
) -> dict[str, float]:
    """边缘密度 + 局部方差 + 红棕/白脉色调。

    密贴碎块连通域也会很大，故纹理看碎；白脉柱面也有高纹理，
    需用红棕土状占比压住误伤。
    """
    h, w = gray.shape
    xa = 0 if x0 is None else max(0, x0)
    xb = w if x1 is None else min(w, x1)
    empty = {
        "edge": 0.0,
        "lap": 0.0,
        "tex": 0.0,
        "smooth_frac": 1.0,
        "grainy_frac": 0.0,
        "red_frac": 0.0,
        "white_frac": 0.0,
    }
    if xb <= xa + 2 or y1 <= y0 + 2:
        return empty

    band = gray[y0:y1, xa:xb]
    valid = distract[y0:y1, xa:xb] == 0
    if not np.any(valid):
        return empty

    blur = cv2.GaussianBlur(band, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    edge = float(edges[valid].mean() / 255.0)
    lap = cv2.Laplacian(band, cv2.CV_32F, ksize=3)
    lap_std = float(lap[valid].std())
    k = 9
    bf = band.astype(np.float32)
    mu = cv2.blur(bf, (k, k))
    mu2 = cv2.blur(bf * bf, (k, k))
    local_std = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    tex = float(local_std[valid].mean())

    bh, bw = band.shape
    win = max(48, bw // 16)
    step = max(1, win // 3)
    smooth_n = 0
    grainy_n = 0
    nwin = 0
    for x in range(0, max(1, bw - win + 1), step):
        m = valid[:, x : x + win]
        if m.sum() < win * bh * 0.35:
            continue
        e = float(edges[:, x : x + win][m].mean() / 255.0)
        ls = float(local_std[:, x : x + win][m].mean())
        nwin += 1
        if e < 0.14 and ls < 16.0:
            smooth_n += 1
        if e > 0.20 or ls > 20.0:
            grainy_n += 1

    red_frac = 0.0
    white_frac = 0.0
    if bgr is not None:
        hsv = cv2.cvtColor(bgr[y0:y1, xa:xb], cv2.COLOR_BGR2HSV)
        hh, ss, vv = cv2.split(hsv)
        rock = valid.copy()
        if black_thr is not None:
            rock &= band >= float(black_thr)
        if np.any(rock):
            hf = hh[rock].astype(np.float32)
            sf = ss[rock].astype(np.float32)
            vf = vv[rock].astype(np.float32)
            red = ((hf <= 20) | (hf >= 160)) & (sf >= 40) & (sf <= 200) & (vf >= 40) & (vf <= 210)
            white = (sf <= 40) & (vf >= 160)
            red_frac = float(red.mean())
            white_frac = float(white.mean())

    return {
        "edge": edge,
        "lap": lap_std,
        "tex": tex,
        "smooth_frac": smooth_n / max(nwin, 1),
        "grainy_frac": grainy_n / max(nwin, 1),
        "red_frac": red_frac,
        "white_frac": white_frac,
    }


def row_is_all_fragment(metrics: dict[str, float]) -> bool:
    """整行土红密实碎屑才短路；白脉柱面即使纹理碎也不短路。"""
    if metrics.get("red_frac", 0.0) < ROW_FRAG_RED_MIN:
        return False
    if metrics.get("white_frac", 0.0) > ROW_FRAG_WHITE_MAX:
        return False
    grainy = (
        metrics["lap"] >= ROW_FRAG_LAP_MIN
        and metrics["tex"] >= ROW_FRAG_TEX_MIN
        and metrics["grainy_frac"] >= ROW_FRAG_GRAINY_MIN
        and metrics["smooth_frac"] <= ROW_FRAG_SMOOTH_MAX
    )
    edge_heavy = (
        metrics["edge"] >= ROW_FRAG_EDGE_MIN
        and metrics["tex"] >= ROW_FRAG_TEX_MIN
        and metrics["grainy_frac"] >= ROW_FRAG_GRAINY_MIN
        and metrics["smooth_frac"] <= ROW_FRAG_SMOOTH_MAX
    )
    return bool(grainy or edge_heavy)


def segment_texture_is_fragment(metrics: dict[str, float]) -> bool:
    """段级碎裂否决：仅土红碎堆纹理才强制 fragment，避免白脉误伤。"""
    if metrics.get("red_frac", 0.0) < SEG_FRAG_RED_MIN:
        return False
    return bool(
        (metrics["lap"] >= SEG_FRAG_LAP_MIN and metrics["tex"] >= SEG_FRAG_TEX_MIN)
        or (metrics["edge"] >= SEG_FRAG_EDGE_MIN and metrics["tex"] >= SEG_FRAG_TEX_MIN)
    )


def full_band_fragment_seg(
    gray: np.ndarray,
    distract: np.ndarray,
    y0: int,
    y1: int,
    black_thr: float,
) -> dict:
    """全碎行的整带 fragment 段，仅用于可视化轮廓。"""
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    roi = gray[y0:y1]
    dist = distract[y0:y1]
    raw = ((roi >= black_thr) & (dist == 0)).astype(np.uint8) * 255
    raw = cv2.morphologyEx(
        raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    mask[y0:y1] = raw
    return {
        "kind": "fragment",
        "x0": 0,
        "x1": w,
        "y0": y0,
        "y1": y1,
        "width": w,
        "height": y1 - y0,
        "black_frac": 0.0,
        "white_frac": 0.0,
        "cover": float((raw > 0).mean()),
        "mask": mask,
        "all_fragment": True,
    }


def classify_interval(
    gray: np.ndarray,
    distract: np.ndarray,
    mask: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    black_thr: float,
    white_thr: float,
    img_w: int,
    bgr: np.ndarray | None = None,
) -> dict | None:
    core_h = y1 - y0
    # 用实际岩体 mask 收紧左右界，避免把断口后游离小块算进完整段宽度
    cols = np.where(mask[y0:y1, x0:x1].any(axis=0))[0]
    if cols.size == 0:
        return None
    x0k = x0 + int(cols.min())
    x1k = x0 + int(cols.max()) + 1
    width = x1k - x0k
    if width < int(img_w * MIN_SEG_WIDTH_FRAC):
        return None

    roi_gray = gray[y0:y1, x0k:x1k]
    roi_mask = mask[y0:y1, x0k:x1k] > 0
    if roi_mask.mean() < 0.25:
        return {"kind": "fragment", "x0": x0k, "x1": x1k, "y0": y0, "y1": y1, "mask": mask}

    black_frac = float(((roi_gray < black_thr) & ~roi_mask).mean())
    white_frac = float(((roi_gray >= white_thr) & roi_mask).sum() / max(roi_mask.sum(), 1))
    gray_white_frac = float(roi_mask.mean())  # 段内岩体覆盖
    # 高度：mask 在竖直方向的跨度
    rows = np.where(roi_mask.any(axis=1))[0]
    if rows.size == 0:
        return None
    height = int(rows.max() - rows.min() + 1)

    tex = fragment_texture_metrics(
        gray, distract, y0, y1, x0k, x1k, bgr=bgr, black_thr=black_thr
    )
    looks_intact = (
        height >= int(core_h * 0.50)
        and gray_white_frac >= 0.35
        and black_frac <= 0.28  # 非岩体暗缝（托盘/断口），不含已入 mask 的柱面
        and white_frac < 0.60  # 过高多为标签/大片高光，不当完整岩块
        and width >= int(img_w * MIN_SEG_WIDTH_FRAC)
        and not segment_texture_is_fragment(tex)
    )
    kind = "intact" if looks_intact else "fragment"
    return {
        "kind": kind,
        "x0": x0k,
        "x1": x1k,
        "y0": y0,
        "y1": y1,
        "width": width,
        "height": height,
        "black_frac": black_frac,
        "white_frac": white_frac,
        "cover": gray_white_frac,
        "edge": tex["edge"],
        "lap": tex["lap"],
        "tex": tex["tex"],
        "red_frac": tex["red_frac"],
        "mask": mask,
    }


def fit_crack(
    gray: np.ndarray,
    y0: int,
    y1: int,
    gap: tuple[int, int],
    black_thr: float,
) -> np.ndarray:
    """在缝区间内拟合可倾斜线；端点限制在缝的 x 范围内。"""
    ga, gb = gap
    roi = gray[y0:y1, ga:gb]
    ys, xs = np.where(roi < black_thr)
    if ys.size < 20:
        mx = (ga + gb) // 2
        return np.array([[mx, y0], [mx, y1]], dtype=np.int32)

    xs = xs + ga
    ys = ys + y0
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    vx, vy, xf, yf = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    if abs(vy) < 1e-6:
        mx = int(np.clip(xf, ga, gb - 1))
        return np.array([[mx, y0], [mx, y1]], dtype=np.int32)

    t0 = (y0 - yf) / vy
    t1 = (y1 - yf) / vy
    x_at_y0 = float(xf + t0 * vx)
    x_at_y1 = float(xf + t1 * vx)

    # 限制在缝附近，防止斜线扫过整段岩芯
    mid = 0.5 * (ga + gb)
    max_dev = max(18, (gb - ga) + 12)
    x_at_y0 = float(np.clip(x_at_y0, mid - max_dev, mid + max_dev))
    x_at_y1 = float(np.clip(x_at_y1, mid - max_dev, mid + max_dev))
    x_at_y0 = float(np.clip(x_at_y0, 0, gray.shape[1] - 1))
    x_at_y1 = float(np.clip(x_at_y1, 0, gray.shape[1] - 1))
    return np.array([[int(x_at_y0), y0], [int(x_at_y1), y1]], dtype=np.int32)


def palette(n: int) -> list[tuple[int, int, int]]:
    base = [
        (50, 190, 70),
        (40, 180, 220),
        (220, 160, 40),
        (200, 90, 200),
        (60, 90, 230),
        (40, 200, 170),
        (180, 110, 50),
        (140, 200, 70),
    ]
    out = list(base)
    rng = np.random.default_rng(1)
    while len(out) < n:
        out.append(tuple(int(v) for v in rng.integers(40, 230, size=3)))
    return out[:n]


def draw_result(
    bgr: np.ndarray,
    segs: list[dict],
    cracks: list[np.ndarray],
    all_fragment: bool = False,
) -> np.ndarray:
    vis = bgr.copy()
    overlay = bgr.copy()
    intact = [s for s in segs if s["kind"] == "intact"]
    frags = [s for s in segs if s["kind"] == "fragment"]
    colors = palette(len(intact))

    for i, seg in enumerate(intact):
        color = colors[i]
        m = seg["mask"] > 0
        overlay[m] = color
        cnts, _ = cv2.findContours(seg["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 2)
        cv2.putText(
            vis,
            f"I{i+1}",
            (seg["x0"] + 6, max(20, seg["y0"] + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for seg in frags:
        m = seg["mask"] > 0
        overlay[m] = (70, 70, 70)
        cnts, _ = cv2.findContours(seg["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (100, 100, 100), 1)

    vis = cv2.addWeighted(overlay, 0.38, vis, 0.62, 0)

    for crack in cracks:
        cv2.polylines(vis, [crack.reshape(-1, 1, 2)], False, (0, 0, 255), 2, cv2.LINE_AA)
        for p in crack:
            cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)

    if all_fragment:
        msg = "all_fragment=1 intact=0 cracks=0"
    else:
        msg = f"intact={len(intact)} fragments={len(frags)} cracks={len(cracks)}"
    cv2.putText(vis, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_levels(
    gray: np.ndarray,
    distract: np.ndarray,
    y0: int,
    y1: int,
    black_thr: float,
    white_thr: float,
    gaps: list[tuple[int, int]],
) -> np.ndarray:
    h, w = gray.shape
    dbg = np.zeros((h, w, 3), np.uint8)
    band = np.zeros((h, w), bool)
    band[y0:y1] = True
    black = band & (gray < black_thr) & (distract == 0)
    white = band & (gray >= white_thr) & (distract == 0)
    mid = band & ~black & ~white & (distract == 0)
    dbg[black] = (30, 30, 30)
    dbg[mid] = (140, 140, 140)
    dbg[white] = (230, 230, 230)
    dbg[distract > 0] = (40, 40, 200)  # 标签：偏红，仅小区域
    for a, b in gaps:
        cv2.rectangle(dbg, (a, y0), (b, y1), (0, 0, 255), 1)
    cv2.line(dbg, (0, y0), (w - 1, y0), (0, 255, 0), 1)
    cv2.line(dbg, (0, y1), (w - 1, y1), (0, 255, 0), 1)
    return dbg


def resolve_sample_id(path: Path) -> str:
    """从子目录名或文件名解析箱号，如 ZK1031-06。"""
    if path.parent.resolve() != INPUT_DIR.resolve():
        m = SAMPLE_RE.match(path.parent.name)
        if m:
            return m.group(1)
    m = SAMPLE_RE.match(path.stem)
    if m:
        return m.group(1)
    return path.stem


def resolve_row_id(path: Path) -> str | None:
    """从文件名解析 row_01 这类行号。"""
    m = ROW_RE.search(path.stem)
    return m.group(1).lower() if m else None


def out_dir_for(sample_id: str, row_id: str | None = None) -> Path:
    """output/<孔号>/<箱号>/<row>/，如 ZK1031/ZK1031-06/row_01/。"""
    well = sample_id.rsplit("-", 1)[0]
    d = OUT_DIR / well / sample_id
    if row_id:
        d = d / row_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def iter_input_images() -> list[Path]:
    files: list[Path] = []
    for p in sorted(INPUT_DIR.rglob("*.jpg")) + sorted(INPUT_DIR.rglob("*.png")):
        name = p.name.lower()
        if name.startswith("overlay") or name == "crop.json":
            continue
        files.append(p)
    return files


def process_one(path: Path) -> None:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    h, w = bgr.shape[:2]
    distract = distractor_mask(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(gray, 11)
    gray_work = gray.copy()
    gray_work[distract > 0] = med[distract > 0]

    y0, y1 = estimate_core_band(gray_work, distract)
    sig = analyze_columns(gray_work, distract, y0, y1)
    black_thr = float(sig["black_thr"])
    white_thr = float(sig["white_thr"])

    row_tex = fragment_texture_metrics(
        gray_work, distract, y0, y1, bgr=bgr, black_thr=black_thr
    )
    all_frag = row_is_all_fragment(row_tex)

    if all_frag:
        gaps: list[tuple[int, int]] = []
        segs = [full_band_fragment_seg(gray_work, distract, y0, y1, black_thr)]
        cracks: list[np.ndarray] = []
        intact: list[dict] = []
    else:
        gaps = find_gap_intervals(sig)
        segs = []
        for x0, x1 in intervals_between(w, gaps):
            mask = segment_mask_in_interval(gray_work, distract, y0, y1, x0, x1, black_thr)
            info = classify_interval(
                gray_work,
                distract,
                mask,
                y0,
                y1,
                x0,
                x1,
                black_thr,
                white_thr,
                w,
                bgr=bgr,
            )
            if info is not None:
                segs.append(info)

        intact = [s for s in segs if s["kind"] == "intact"]
        # 裂缝：只在完整段之间的 gap 上拟合（完整-破碎之间也可标）
        cracks = []
        for gap in gaps:
            left = any(s["x1"] <= gap[0] + 2 and s["x1"] >= gap[0] - 30 for s in segs)
            right = any(s["x0"] >= gap[1] - 2 and s["x0"] <= gap[1] + 30 for s in segs)
            if left or right or True:
                cracks.append(fit_crack(gray_work, y0, y1, gap, black_thr))

    vis = draw_result(bgr, segs, cracks, all_fragment=all_frag)
    levels = draw_levels(gray_work, distract, y0, y1, black_thr, white_thr, gaps)

    sample_id = resolve_sample_id(path)
    row_id = resolve_row_id(path)
    sample_out = out_dir_for(sample_id, row_id)
    out_vis = sample_out / f"{path.stem}_intact.jpg"
    out_lvl = sample_out / f"{path.stem}_levels.jpg"
    cv2.imwrite(str(out_vis), vis)
    cv2.imwrite(str(out_lvl), levels)

    label = f"{sample_id}/{row_id}" if row_id else sample_id
    print(f"\n=== {label}/{path.name} ===")
    print(
        f"band y=[{y0},{y1}) rock_med={sig['rock_med']:.1f} "
        f"black_thr={black_thr:.1f} white_thr={white_thr:.1f}"
    )
    print(
        f"row_tex edge={row_tex['edge']:.3f} lap={row_tex['lap']:.1f} "
        f"tex={row_tex['tex']:.1f} grainy={row_tex['grainy_frac']:.2f} "
        f"smooth={row_tex['smooth_frac']:.2f} red={row_tex['red_frac']:.2f} "
        f"white={row_tex['white_frac']:.2f} all_fragment={int(all_frag)}"
    )
    print(f"gaps={gaps}")
    print(
        f"intact={len(intact)} fragments={sum(s['kind']=='fragment' for s in segs)} "
        f"cracks={len(cracks)}"
    )
    for i, s in enumerate(intact, 1):
        extra = ""
        if "edge" in s:
            extra = f" edge={s['edge']:.3f} lap={s['lap']:.1f} tex={s['tex']:.1f}"
        print(
            f"  I{i}: x=[{s['x0']},{s['x1']}) w={s['width']} "
            f"cover={s['cover']:.2f} black={s['black_frac']:.2f} white={s['white_frac']:.2f}"
            f"{extra}"
        )
    print(f"saved: {out_vis}")


def main() -> None:
    files = iter_input_images()
    if not files:
        raise SystemExit(f"no images in {INPUT_DIR}")
    for p in files:
        process_one(p)


if __name__ == "__main__":
    main()
