"""可视化工具

从报告 notebook（第6组.ipynb）中抽出的同名实现，函数体与报告完全一致。
"""
import colorsys

import numpy as np
import cv2



cc = baseline_cc(th)
vis_cc = bgr.copy(); vis_ws = bgr.copy()

import colorsys


def _palette(n=20, sat=0.55, val=0.90):
    out = []
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        out.append((int(b * 255), int(g * 255), int(r * 255)))     # BGR
    return out


PALETTE = _palette()


def colorize(labels, n_lab, base_img, offset=0):
    out = base_img.copy()
    color = np.zeros_like(out)
    for lb in range(1, n_lab + 1):
        mask_lb = (labels == lb)
        if mask_lb.sum() == 0:
            continue
        color[mask_lb] = np.array(PALETTE[(lb + offset) % len(PALETTE)], dtype=np.uint8)
    return cv2.addWeighted(out, 0.45, color, 0.55, 0)

_, lab_cc, stats_cc, cents_cc = cv2.connectedComponentsWithStats(th, connectivity=8)
vis_cc = colorize(lab_cc, int(lab_cc.max()), bgr)
vis_ws = colorize(res["labels"], int(res["labels"].max()) - 1, bgr)   # 最大标签是分水岭的背景标记，不参与着色
pred_pts = [(r["cx"], r["cy"]) for r in res["regions"]]
for (x, y) in pred_pts:
    cv2.circle(vis_ws, (int(x), int(y)), 2, (255, 255, 255), -1)

