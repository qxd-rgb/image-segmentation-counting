"""分割流水线（M3~M7 模块）

从报告 notebook（第6组.ipynb）中抽出的同名实现，函数体与报告完全一致。
"""
import numpy as np
import cv2



def preprocess(bgr, sigma=25):
    '''灰度化 + 高斯背景估计 + 除法归一化'''
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    base = cv2.GaussianBlur(gray, (0, 0), sigma).astype(np.float32)
    norm = cv2.divide(gray.astype(np.float32) + 1.0, base + 1.0)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    return gray, base, norm, blur


def binarize(blur, open_it=1, close_it=2):
    '''Otsu 二值化 + 极性判定 + 形态学 + 孔洞填充'''
    thr, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    flipped = False
    if np.count_nonzero(th) > th.size * 0.5:      # 前景应为少数类
        th = cv2.bitwise_not(th); flipped = True
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=open_it)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=close_it)
    # 孔洞填充
    ff = th.copy()
    mask = np.zeros((th.shape[0] + 2, th.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 255)
    th = th | cv2.bitwise_not(ff)
    return th, thr, flipped


def _reconstruct(marker, mask, kernel):
    '''灰度形态学重建（膨胀 + 截断），用于 h-maxima'''
    prev = marker.copy()
    while True:
        nxt = np.minimum(cv2.dilate(prev, kernel), mask)
        if np.array_equal(nxt, prev):
            return nxt
        prev = nxt


def distance_and_seeds(binary, h_rel=0.25):
    '''距离变换 + h-maxima 重建 + 种子标记'''
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() < 5:
        z = np.zeros_like(binary, np.int32)
        return dist, z, z, np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    h = max(1.5, h_rel * float(dist.max()))
    rec = _reconstruct(np.clip(dist - h, 0, None), dist, kernel)
    peaks = ((dist - rec) > 1e-3).astype(np.uint8) * 255
    peaks = cv2.morphologyEx(peaks, cv2.MORPH_CLOSE, kernel)
    n_seed, markers = cv2.connectedComponents(peaks)
    return dist, markers.astype(np.int32), peaks, h


def watershed_instances(binary, h_rel=0.25, min_area=250):
    '''分水岭分割：返回实例掩膜、标签图、区域属性列表与中间结果'''
    dist, markers, peaks, h = distance_and_seeds(binary, h_rel)
    if markers.max() <= 0:
        return dict(mask=np.zeros_like(binary), labels=np.zeros_like(binary, np.int32),
                    regions=[], dist=dist, peaks=peaks, h=h, markers=markers)
    markers = markers.astype(np.int32)
    bg_label = int(markers.max()) + 1
    markers[binary == 0] = bg_label
    ws = cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers.copy())
    inst = np.zeros_like(binary, np.uint8)
    regions = []
    for lb in range(1, int(ws.max()) + 1):
        if lb == bg_label:
            continue
        m = (ws == lb).astype(np.uint8)
        area = int(m.sum())
        if area < min_area:
            continue
        mom = cv2.moments(m)
        cx = mom["m10"] / max(mom["m00"], 1e-6)
        cy = mom["m01"] / max(mom["m00"], 1e-6)
        regions.append(dict(label=lb, area=area, cx=cx, cy=cy))
        inst[m > 0] = 255
    return dict(mask=inst, labels=ws, regions=regions, dist=dist, peaks=peaks, h=h, markers=markers)


def baseline_cc(binary, min_area=250):
    '''基线：连通域直接计数，不做粘连拆分'''
    n_lab, lab, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    pts = [(cents[i][0], cents[i][1], int(stats[i, cv2.CC_STAT_AREA]))
           for i in range(1, n_lab) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    mask = np.zeros_like(binary)
    for (x, y, a) in pts:
        mask[lab == lab[int(y), int(x)]] = 255
    return dict(mask=mask, regions=pts, labels=lab)



def count_image(path, h_rel=0.25, min_area=250, save=None):
    """真实图像入口：读图 → 预处理 → 二值化 → 分水岭计数。

    path 用 np.fromfile + cv2.imdecode 读取，避免中文路径下 imread 失败。
    save 给一个输出路径时，把实例掩膜写成 PNG。
    """
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    gray, base, norm, blur = preprocess(bgr)
    th, thr, flipped = binarize(blur)
    res = watershed_instances(th, h_rel=h_rel, min_area=min_area)
    if save:
        cv2.imencode(".png", res["mask"])[1].tofile(save)
    return len(res["regions"]), res
