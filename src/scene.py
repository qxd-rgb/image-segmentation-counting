"""仿真场景生成（M1/M2 模块）

从报告 notebook（第6组.ipynb）中抽出的同名实现，函数体与报告完全一致。
"""
import numpy as np
import cv2



SCENES = {
    "sparse": dict(base=65, vignette=22, n=7,  glo=165, ghi=210, gap=1.15, noise=2.5, sp=0.000, rlo=18, rhi=26),
    "dense":  dict(base=65, vignette=22, n=13, glo=165, ghi=210, gap=0.62, noise=2.5, sp=0.000, rlo=18, rhi=26),
    "dark":   dict(base=52, vignette=45, n=9,  glo=112, ghi=132, gap=0.75, noise=6.0, sp=0.002, rlo=18, rhi=26),
}
DEFAULT = dict(base=65, vignette=22, n=8, glo=165, ghi=210, gap=1.05, noise=2.5, sp=0.0, rlo=18, rhi=26)


def make_background(h, w, rng, base=95.0, texture=7.0, vignette=25.0):
    bg = np.full((h, w), float(base), np.float32)
    low = rng.normal(0, 1, (h // 16 + 2, w // 16 + 2)).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC) * texture
    bg += low
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
    bg += ((xx / w - 0.5) * gx + (yy / h - 0.5) * gy) * 45.0
    r = np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    bg -= vignette * (r ** 2)
    return np.clip(bg, 0, 255).astype(np.float32)


def draw_coin(img, cx, cy, rad, gray, rng):
    h, w = img.shape
    x0, x1 = max(0, int(cx - rad - 3)), min(w, int(cx + rad + 3))
    y0, y1 = max(0, int(cy - rad - 3)), min(h, int(cy + rad + 3))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip(rad + 0.5 - d, 0.0, 1.0)                 # 抗锯齿边缘
    if alpha.max() <= 0:
        return
    n = ((xx - cx) * -0.55 + (yy - cy) * -0.55) / (rad + 1e-6)
    shade = 1.0 + 0.10 * n                                    # 左上受光
    rim = np.clip((d - 0.86 * rad) / (0.14 * rad + 1e-6), 0.0, 1.0)
    shade += 0.20 * rim * (1 - rim)                           # 币缘亮环
    inner = np.exp(-((d - 0.68 * rad) ** 2) / (2 * (0.035 * rad) ** 2 + 1e-6))
    shade += 0.06 * inner                                     # 内圈压印纹
    spec = np.exp(-(((xx - cx + 0.35 * rad) ** 2 + (yy - cy + 0.35 * rad) ** 2) / (0.05 * rad ** 2 + 1e-6)))
    tex = rng.normal(0, 2.2, (y1 - y0, x1 - x0)).astype(np.float32)
    coin = gray * shade + 14.0 * spec + tex
    patch = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = patch * (1 - alpha) + np.clip(coin, 0, 255) * alpha


def build_scene(cfg, seed, H=300, W=400):
    '''生成一张仿真图像：返回 (BGR 图像, 真值掩膜, 目标列表[(cx,cy,r)], 配置)'''
    c = dict(DEFAULT); c.update(cfg)
    rng = np.random.default_rng(seed)
    g = make_background(H, W, rng, base=c["base"], texture=7, vignette=c["vignette"])

    coins, tries = [], 0
    while len(coins) < c["n"] and tries < 8000:
        tries += 1
        r = float(rng.uniform(c["rlo"], c["rhi"]))
        cx = float(rng.uniform(r + 5, W - r - 5))
        cy = float(rng.uniform(r + 5, H - r - 5))
        if all(np.hypot(cx - q[0], cy - q[1]) >= c["gap"] * (r + q[2]) for q in coins):
            coins.append((cx, cy, r))

    shadow = np.zeros((H, W), np.uint8)
    for (cx, cy, r) in coins:
        cv2.circle(shadow, (int(cx + 3), int(cy + 4)), int(r * 1.04), 255, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 6) / 255.0
    g = g * (1 - 0.30 * shadow)                               # 阴影压暗
    for (cx, cy, r) in coins:
        draw_coin(g, cx, cy, r, float(rng.uniform(c["glo"], c["ghi"])), rng)

    g = np.clip(g, 0, 255)
    if c["sp"] > 0:                                           # 椒盐噪声
        g = np.clip(g + rng.normal(0, c["noise"], g.shape), 0, 255)
        g[rng.random(g.shape) < c["sp"]] = 255
        g[rng.random(g.shape) < c["sp"]] = 0
    else:
        g = g + rng.normal(0, c["noise"], g.shape)

    img = cv2.GaussianBlur(np.clip(g, 0, 255).astype(np.uint8), (3, 3), 0)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gt = np.zeros((H, W), np.uint8)
    for (cx, cy, r) in coins:
        cv2.circle(gt, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    return bgr, gt, coins, c


def build_dataset(scene, n_img=15, seed0=1000):
    return [build_scene(SCENES[scene], seed0 + i) for i in range(n_img)]





def build_hard(kind, seed=5000):
    cfg = dict(SCENES["dense"])
    H, W = 300, 400
    if kind == "heavy":          # 严重重叠
        cfg.update(gap=0.42, n=12)
        return build_scene(cfg, seed, H, W)
    if kind == "tiny":           # 极小目标
        cfg.update(rlo=8, rhi=11, n=22, gap=1.2)
        return build_scene(cfg, seed, H, W)
    if kind == "glare":          # 强高光：币面镜面反射 + 背景中的灯光反射斑
        cfg.update(n=9, gap=0.9, glo=210, ghi=250)
        bgr, gt, coins, c = build_scene(cfg, seed, H, W)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        spot = 120 * np.exp(-(((xx - 90) ** 2 + (yy - 70) ** 2) / (2 * 55.0 ** 2)))
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) + spot
        # 三处"灯光在台面上的反射斑"：亮度接近饱和的小圆斑，尺寸刚好超过面积阈值
        for (sx, sy, sr) in [(300, 60, 11), (340, 210, 10), (120, 245, 12)]:
            blob = np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * sr ** 2))) * 190
            g = g + blob
        return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), gt, coins, c
    if kind == "border":         # 贴边目标 + 强阴影
        cfg.update(gap=1.0, n=8, base=110, vignette=10)
        bgr, gt, coins, c = build_scene(cfg, seed, H, W)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        rng2 = np.random.default_rng(seed + 7)
        extra = [(6.0, 150.0, 24.0), (float(W - 5), 250.0, 22.0)]   # 圆心落在画面外沿
        for (cx, cy, r) in extra:
            draw_coin(g, cx, cy, r, 190.0, rng2)
            cv2.circle(gt, (int(cx), int(cy)), int(r), 255, -1)
        coins = list(coins) + extra
        sh = np.zeros((H, W), np.uint8)
        for (cx, cy, r) in coins:
            cv2.circle(sh, (int(cx + 14), int(cy + 16)), int(r * 1.15), 255, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), 9) / 255.0
        g2 = g * (1 - 0.55 * sh)
        return cv2.cvtColor(np.clip(g2, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), gt, coins, c
    raise ValueError(kind)


