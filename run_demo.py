# -*- coding: utf-8 -*-
"""命令行入口：生成仿真数据、跑完整分割计数流程、打印指标。

用法：
    python run_demo.py                     # 默认每场景 15 张（与报告一致，共 45 张）
    python run_demo.py --n 5               # 每场景 5 张，快速看结果
    python run_demo.py --sweeps            # 额外跑重叠度/噪声两组扫描
    python run_demo.py --image coins.jpg   # 对单张真实图像计数，并保存实例掩膜
"""
import argparse
import os
import time

import numpy as np

from src import (build_dataset, build_hard, count_image, evaluate_dataset,
                 preprocess, binarize, watershed_instances, summarize,
                 sweep_noise, sweep_overlap)


def run_dataset(n_img, out_dir, do_sweeps):
    t_gen = time.time()
    data = {s: build_dataset(s, n_img=n_img, seed0=1000) for s in ("sparse", "dense", "dark")}
    gen_cost = time.time() - t_gen
    t0 = time.time()
    df = evaluate_dataset(data)
    elapsed = time.time() - t0

    print("\n=== 逐场景汇总 ===")
    summary = summarize(df)
    print(summary.to_string())

    n_img_all = len(df)
    acc = df["acc"].mean() * 100
    acc_cc = df["acc_cc"].mean() * 100
    exact = (df["abs_err"] == 0).mean() * 100
    mae = df["abs_err"].mean()
    bias = df["err"].mean()
    iou = df["iou"].mean()
    tp, fp, fn = int(df.tp.sum()), int(df.fp.sum()), int(df.fn.sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
    print("\n=== 整体（%d 张图 / %d 个目标） ===" % (n_img_all, int(df.n_gt.sum())))
    print("计数准确率        : %.2f%%  (连通域基线 %.2f%%)" % (acc, acc_cc))
    print("完全正确率        : %.1f%%" % exact)
    print("MAE               : %.2f 个/图" % mae)
    print("带符号平均偏差    : %+.2f 个/图" % bias)
    print("F1 / 前景并集 IoU : %.4f / %.4f" % (f1, iou))
    # 单独计时一遍流水线本身，口径与报告 5.1 节一致
    t_ws, k = time.time(), 0
    for scene, items in data.items():
        for (bgr, gt, coins, cfg) in items:
            gray, base, norm, blur = preprocess(bgr)
            th, thr, flipped = binarize(blur)
            watershed_instances(th, min_area=250)
            k += 1
    print("流水线耗时        : %.1f ms/张（preprocess + binarize + watershed，共 %d 张；"
          "随机器负载浮动，报告在同一台机器上测得 35.3 ms）"
          % ((time.time() - t_ws) / max(k, 1) * 1000, k))

    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), encoding="utf-8-sig")
    print("\n汇总已写入 %s" % os.path.join(out_dir, "summary.csv"))

    if do_sweeps:
        print("\n=== 重叠度扫描（gap 越小重叠越严重） ===")
        print(sweep_overlap().round(4).to_string(index=False))
        print("\n=== 噪声扫描 ===")
        print(sweep_noise().round(4).to_string(index=False))
    return df


def run_image(path, out_dir):
    t0 = time.time()
    n, res = count_image(path, save=os.path.join(out_dir, "mask.png") if out_dir else None)
    print("目标数: %d（耗时 %.1f ms）" % (n, (time.time() - t0) * 1000))
    for r in res["regions"][:10]:
        print("  面积 %5d  质心 (%.1f, %.1f)" % (r["area"], r["cx"], r["cy"]))
    if len(res["regions"]) > 10:
        print("  ... 共 %d 个区域" % len(res["regions"]))
    return n


def main():
    ap = argparse.ArgumentParser(description="图像分割与计数 demo")
    ap.add_argument("--n", type=int, default=15, help="每个场景生成的图像数（默认 15）")
    ap.add_argument("--out", default="out", help="结果输出目录")
    ap.add_argument("--sweeps", action="store_true", help="额外跑鲁棒性扫描")
    ap.add_argument("--image", default=None, help="对单张图像计数（跳过仿真数据集）")
    args = ap.parse_args()

    np.random.seed(2026)
    if args.image:
        run_image(args.image, args.out)
    else:
        run_dataset(args.n, args.out, args.sweeps)


if __name__ == "__main__":
    main()
