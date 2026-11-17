"""图像分割与计数：仿真场景生成 + 分水岭分割计数 + 精度评估。"""

from .scene import SCENES, DEFAULT, build_dataset, build_scene, build_hard
from .pipeline import (preprocess, binarize, distance_and_seeds,
                       watershed_instances, baseline_cc, count_image)
from .metrics import evaluate_dataset, evaluate_one, summarize, sweep_overlap, sweep_noise

__all__ = [
    "SCENES", "DEFAULT", "build_dataset", "build_scene", "build_hard",
    "preprocess", "binarize", "distance_and_seeds", "watershed_instances",
    "baseline_cc", "count_image", "evaluate_dataset", "evaluate_one",
    "summarize", "sweep_overlap", "sweep_noise",
]
