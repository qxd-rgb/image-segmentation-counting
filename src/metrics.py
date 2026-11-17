"""精度评估（M8 模块）

从报告 notebook（第6组.ipynb）中抽出的同名实现，函数体与报告完全一致。
"""
import math

import numpy as np
import pandas as pd

from .pipeline import preprocess, binarize, baseline_cc, watershed_instances
from .scene import SCENES, build_scene



def match_centers(pred, gt, tol=20.0):
    '''质心最近邻贪心匹配，返回 (TP, FP, FN)'''
    used = [False] * len(pred)
    tp = 0
    for (gx, gy) in gt:
        best, bd = -1, 1e9
        for i, (px, py) in enumerate(pred):
            if used[i]:
                continue
            d = math.hypot(px - gx, py - gy)
            if d < bd:
                bd, best = d, i
        if best >= 0 and bd <= tol:
            used[best] = True
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def evaluate_one(bgr, gt_mask, gt_pts, h_rel=0.25, min_area=250):
    gray, base, norm, blur = preprocess(bgr)
    th, thr, flipped = binarize(blur)
    ws = watershed_instances(th, h_rel=h_rel, min_area=min_area)
    cc = baseline_cc(th, min_area=min_area)
    pred_ws = [(r["cx"], r["cy"]) for r in ws["regions"]]
    pred_cc = [(x, y) for (x, y, a) in cc["regions"]]
    inter = np.count_nonzero((ws["mask"] > 0) & (gt_mask > 0))
    union = np.count_nonzero((ws["mask"] > 0) | (gt_mask > 0))
    tp, fp, fn = match_centers(pred_ws, gt_pts)
    tp_cc, fp_cc, fn_cc = match_centers(pred_cc, gt_pts)
    return dict(n_gt=len(gt_pts), n_ws=len(pred_ws), n_cc=len(pred_cc),
                iou=inter / max(union, 1), tp=tp, fp=fp, fn=fn,
                tp_cc=tp_cc, fp_cc=fp_cc, fn_cc=fn_cc,
                precision=tp / max(tp + fp, 1e-9), recall=tp / max(tp + fn, 1e-9),
                thr=thr, ws=ws, th=th, blur=blur, norm=norm)



def evaluate_dataset(data_dict, h_rel=0.25, min_area=250):
    """在一组（场景 -> 图像列表）上逐图评估，返回逐图指标的 DataFrame。"""
    rows = []
    for scene, items in data_dict.items():
        for i, (bgr, gt_mask, coins, cfg) in enumerate(items):
            m = evaluate_one(bgr, gt_mask, [(c[0], c[1]) for c in coins], h_rel, min_area)
            rows.append(dict(scene=scene, idx=i, n_gt=m["n_gt"], n_ws=m["n_ws"], n_cc=m["n_cc"],
                             iou=m["iou"], tp=m["tp"], fp=m["fp"], fn=m["fn"],
                             tp_cc=m["tp_cc"], fp_cc=m["fp_cc"], fn_cc=m["fn_cc"],
                             precision=m["precision"], recall=m["recall"]))
    df = pd.DataFrame(rows)
    # 逐图偏差与其绝对值：MAE 用绝对值，带符号平均偏差用原始差值
    df["err"] = df["n_ws"] - df["n_gt"]
    df["err_cc"] = df["n_cc"] - df["n_gt"]
    df["abs_err"] = df["err"].abs()
    df["abs_err_cc"] = df["err_cc"].abs()
    df["acc"] = 1 - df["abs_err"] / df["n_gt"]
    df["acc_cc"] = 1 - df["abs_err_cc"] / df["n_gt"]
    return df


def summarize(df):
    """按场景汇总指标，返回 DataFrame（每行一个场景）。"""
    out = []
    for scene, g in df.groupby("scene", sort=False):
        tp, fp, fn = int(g.tp.sum()), int(g.fp.sum()), int(g.fn.sum())
        out.append({
            "场景": scene,
            "图像数": len(g),
            "真实目标均值": round(g.n_gt.mean(), 2),
            "分水岭计数均值": round(g.n_ws.mean(), 2),
            "连通域计数均值": round(g.n_cc.mean(), 2),
            "计数准确率(分水岭)": "%.2f%%" % (100 * g.acc.mean()),
            "计数准确率(连通域)": "%.2f%%" % (100 * g.acc_cc.mean()),
            "完全正确率(分水岭)": "%.1f%%" % (100 * (g.abs_err == 0).mean()),
            "MAE(分水岭)": round(g.abs_err.mean(), 3),
            "MAE(连通域)": round(g.abs_err_cc.mean(), 3),
            "带符号平均偏差(分水岭)": round(g.err.mean(), 2),
            "带符号平均偏差(连通域)": round(g.err_cc.mean(), 2),
            "Precision": round(tp / max(tp + fp, 1e-9), 4),
            "Recall": round(tp / max(tp + fn, 1e-9), 4),
            "F1": round(2 * tp / max(2 * tp + fp + fn, 1e-9), 4),
            "IoU": round(g.iou.mean(), 4),
        })
    return pd.DataFrame(out).set_index("场景")


def sweep_overlap(gaps=(1.00, 0.85, 0.70, 0.55, 0.40), n_img=8, seed0=3000):
    rows = []
    for gap in gaps:
        cfg = dict(SCENES["dense"]); cfg.update(gap=gap, n=12)
        accs_ws, accs_cc, ious = [], [], []
        for i in range(n_img):
            bgr, gm, coins, _ = build_scene(cfg, seed0 + i)
            m = evaluate_one(bgr, gm, [(c[0], c[1]) for c in coins])
            accs_ws.append(1 - abs(m["n_ws"] - len(coins)) / len(coins))
            accs_cc.append(1 - abs(m["n_cc"] - len(coins)) / len(coins))
            ious.append(m["iou"])
        rows.append(dict(gap=gap, acc_ws=np.mean(accs_ws), acc_cc=np.mean(accs_cc), iou=np.mean(ious)))
    return pd.DataFrame(rows)


def sweep_noise(sigmas=(0, 4, 8, 12, 16, 20), n_img=8, seed0=4000):
    rows = []
    for sg in sigmas:
        cfg = dict(SCENES["dense"]); cfg.update(noise=float(sg), n=12)
        accs_ws, accs_cc, ious = [], [], []
        for i in range(n_img):
            bgr, gm, coins, _ = build_scene(cfg, seed0 + i)
            m = evaluate_one(bgr, gm, [(c[0], c[1]) for c in coins])
            accs_ws.append(1 - abs(m["n_ws"] - len(coins)) / len(coins))
            accs_cc.append(1 - abs(m["n_cc"] - len(coins)) / len(coins))
            ious.append(m["iou"])
        rows.append(dict(sigma=sg, acc_ws=np.mean(accs_ws), acc_cc=np.mean(accs_cc), iou=np.mean(ious)))
    return pd.DataFrame(rows)


