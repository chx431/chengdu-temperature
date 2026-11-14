# -*- coding: utf-8 -*-
"""
气温小实验：只用前几天的气温，能不能猜出明天的温度？

最常见的用法（什么都不用改）：
    双击 运行.bat        # 会问你三个问题，直接回车就是默认

命令行也行：
    python run.py                     # 成都、2016 年起、λ=1
    python run.py --city 2            # 北京
    python run.py --lam 0.1           # 换个约束强度
    python run.py --start 2010-01-01  # 从 2010 年开始
    python run.py --target tmin       # 改预测最低温
    python run.py --no-fetch          # 不联网，用已有数据

结果：outputs/metrics.json + outputs/chart.png（跑完会自动打开图）
"""
import argparse
import csv
import json
import math
import os
import random
import urllib.request
from datetime import date, timedelta

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = "outputs"
FONT = "C:/Windows/Fonts/msyh.ttc"
NAVY, TEAL, GRAY, AMBER = (31, 78, 121), (31, 158, 168), (150, 158, 168), (214, 150, 50)
SEASON = {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
          6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"}

# 城市编号 → (中文名, 纬度, 经度, 数据文件名)
CITIES = {
    "1": ("成都", 30.5728, 104.0668, "chengdu.csv"),
    "2": ("北京", 39.9042, 116.4074, "beijing.csv"),
    "3": ("上海", 31.2304, 121.4737, "shanghai.csv"),
    "4": ("广州", 23.1291, 113.2644, "guangzhou.csv"),
    "5": ("武汉", 30.5928, 114.3055, "wuhan.csv"),
    "6": ("西安", 34.3416, 108.9398, "xian.csv"),
}

URL = ("https://archive-api.open-meteo.com/v1/archive"
       "?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
       "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Asia%2FShanghai")


# ---------------- 一、拿数据 ----------------
def fetch(lat, lon, start="2016-01-01", end=None, timeout=15):
    end = end or date.today().isoformat()
    req = urllib.request.Request(URL.format(lat=lat, lon=lon, start=start, end=end),
                                 headers={"User-Agent": "cuit-course-project/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))["daily"]
    return list(zip(d["time"], d["temperature_2m_max"], d["temperature_2m_min"], d["precipitation_sum"]))


def synthetic(start="2016-01-01", days=3600, seed=7):
    """备用数据：只用来验证流程能跑，不能当成真实结果。"""
    random.seed(seed)
    d0, rows, noise = date.fromisoformat(start), [], 0.0
    for i in range(days):
        day = d0 + timedelta(days=i)
        season = math.sin(2 * math.pi * (day.timetuple().tm_yday - 110) / 365.25)
        noise = 0.0 if i == 0 else 0.72 * noise + random.gauss(0, 2.2)
        tmax = 22.0 + 9.5 * season + noise + random.gauss(0, 0.8)
        rows.append((day.isoformat(), round(tmax, 1), round(tmax - 6.5, 1),
                     round(max(0.0, random.gauss(1.2, 4.0)) if random.random() < 0.35 else 0.0, 1)))
    return rows


def save(rows, path):
    os.makedirs("data", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "tmax", "tmin", "prcp"])
        for r in rows:
            if r[1] is not None:
                w.writerow(r)


def ensure_data(path, lat, lon, start, no_fetch=False, force_synthetic=False):
    if force_synthetic:
        save(synthetic(start), path)
        print("使用合成数据（只能验证流程，结果不能引用）")
        return
    if no_fetch and os.path.exists(path):
        print(f"使用已有数据 {path}")
        return
    try:
        save(fetch(lat, lon, start), path)
        print("数据来源：Open-Meteo archive-api（真实逐日观测）")
    except Exception as e:
        print("抓取失败，退回合成数据：", repr(e))
        save(synthetic(start), path)


def load(path):
    dates, tmax, tmin = [], [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            d, a, b, _ = line.strip().split(",")
            dates.append(date.fromisoformat(d))
            tmax.append(float(a))
            tmin.append(float(b))
    return dates, np.array(tmax), np.array(tmin)


# ---------------- 二、特征和模型 ----------------
def doy(d):
    return d.timetuple().tm_yday


def build_features(dates, tmax, tmin, target="tmax"):
    rows, ys, idx = [], [], []
    for i in range(7, len(tmax)):
        nr = 2 * np.pi * doy(dates[i]) / 365.25
        rows.append([tmax[i - 1], tmax[i - 2], tmax[i - 3], tmax[i - 7], tmin[i - 1],
                     np.sin(nr), np.cos(nr)])
        ys.append(tmax[i] if target == "tmax" else tmin[i])
        idx.append(i)
    return np.array(rows, float), np.array(ys), np.array(idx)


def persistence(series, idx):
    return series[idx - 1]


def climatology(dates, series, idx, train_idx):
    td, tv = [dates[j] for j in train_idx], [series[j] for j in train_idx]
    out = []
    for i in idx:
        t = doy(dates[i])
        near = [v for d, v in zip(td, tv) if min(abs(doy(d) - t), 365 - abs(doy(d) - t)) <= 3]
        out.append(float(np.mean(near)) if near else float(np.mean(tv)))
    return np.array(out)


def ridge_fit(X, y, lam=1.0):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    A = X1.T @ X1
    reg = lam * np.eye(A.shape[0])
    reg[0, 0] = 0.0
    return np.linalg.solve(A + reg, X1.T @ y)


def ridge_predict(w, X):
    return np.hstack([np.ones((len(X), 1)), X]) @ w


# ---------------- 三、评估和画图 ----------------
def metrics(pred, true):
    return {"MAE": round(float(np.abs(pred - true).mean()), 2),
            "RMSE": round(float(np.sqrt(((pred - true) ** 2).mean())), 2)}


def by_season(dates, idx, pred, true):
    out = {}
    for s in ["春", "夏", "秋", "冬"]:
        m = np.array([SEASON[dates[i].month] == s for i in idx])
        if m.sum() > 5:
            out[s] = metrics(pred[m], true[m])
    return out


def chart(dates, idx, true, preds, season_mae, path, title, unit):
    W, H = 1700, 1150
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f = lambda s: ImageFont.truetype(FONT, s, index=0)
    t = lambda x, y, s, size=22, color=(40, 40, 40): d.text((x, y), s, font=f(size), fill=color)

    n, pad_l, pad_r, top, bot = 120, 100, 60, 90, 560
    y_true = true[-n:]
    lo = min(y_true.min(), *[p[-n:].min() for p in preds.values()]) - 1
    hi = max(y_true.max(), *[p[-n:].max() for p in preds.values()]) + 1
    px = lambda i: pad_l + (W - pad_r - pad_l) * i / (n - 1)
    py = lambda v: bot - (bot - top) * (v - lo) / (hi - lo)

    t(pad_l, 30, f"{title}：实况和三个预测（最近 120 天）", 30, NAVY)
    for gy in range(5):
        v = lo + (hi - lo) * gy / 4
        d.line([(pad_l, py(v)), (W - pad_r, py(v))], fill=(235, 238, 242), width=2)
        t(20, py(v) - 14, f"{v:.0f}°", 18, GRAY)

    last = None
    for i, j in enumerate(idx[-n:]):
        m = dates[j].month
        if m != last:
            d.line([(px(i), bot), (px(i), bot + 8)], fill=GRAY, width=2)
            t(px(i) - 14, bot + 14, f"{m}月", 18, GRAY)
            last = m

    for key, color in [("clim", AMBER), ("ridge", TEAL), ("persist", GRAY)]:
        d.line([(px(i), py(v)) for i, v in enumerate(preds[key][-n:])], fill=color, width=3)
    d.line([(px(i), py(v)) for i, v in enumerate(y_true)], fill=NAVY, width=4)

    lx = pad_l
    for label, color in [("实况", NAVY), ("我写的模型", TEAL), ("气候平均", AMBER), ("持久性", GRAY)]:
        d.line([(lx, 640), (lx + 34, 640)], fill=color, width=5)
        t(lx + 42, 624, label, 20)
        lx += 200

    t(pad_l, 680, f"分季节平均误差（{unit}，越低越好）", 26, NAVY)
    seasons = ["春", "夏", "秋", "冬"]
    vmax = max(season_mae[m][s]["MAE"] for m in ["persist", "clim", "ridge"] for s in seasons) * 1.15
    gw = (W - pad_r - pad_l) / len(seasons)
    for si, s in enumerate(seasons):
        gx = pad_l + gw * si + 40
        for mi, (key, color) in enumerate([("persist", GRAY), ("clim", AMBER), ("ridge", TEAL)]):
            v = season_mae[key][s]["MAE"]
            h = (1040 - 720) * v / vmax
            d.rectangle([gx + mi * 58, 1040 - h, gx + mi * 58 + 44, 1040], fill=color)
            t(gx + mi * 58 + 2, 1040 - h - 26, f"{v:.1f}", 17, color)
        t(gx + 50, 1050, s, 20)
    d.line([(pad_l, 1040), (W - pad_r, 1040)], fill=(210, 216, 224), width=2)
    for v in [0, vmax / 2, vmax]:
        y = 1040 - (1040 - 720) * v / vmax
        d.line([(pad_l, y), (W - pad_r, y)], fill=(235, 238, 242), width=2)

    img.save(path)


# ---------------- 四、主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="1", help="城市编号：1成都 2北京 3上海 4广州 5武汉 6西安")
    ap.add_argument("--start", default="2016-01-01", help="起始日期 YYYY-MM-DD")
    ap.add_argument("--lam", type=float, default=1.0, help="约束强度 λ，默认 1")
    ap.add_argument("--split", type=float, default=0.8, help="训练集比例，默认 0.8")
    ap.add_argument("--target", default="tmax", choices=["tmax", "tmin"], help="预测最高温还是最低温")
    ap.add_argument("--no-fetch", action="store_true", help="不联网，用已有 CSV")
    ap.add_argument("--synthetic", action="store_true", help="强制合成数据")
    ap.add_argument("--no-open", action="store_true", help="跑完不自动打开图")
    a = ap.parse_args()

    city, lat, lon, fname = CITIES.get(str(a.city), CITIES["1"])
    path = os.path.join("data", fname)
    unit = "℃"
    label = "最高" if a.target == "tmax" else "最低"

    os.makedirs(OUT, exist_ok=True)
    ensure_data(path, lat, lon, a.start, no_fetch=a.no_fetch, force_synthetic=a.synthetic)

    dates, tmax, tmin = load(path)
    X, y, idx = build_features(dates, tmax, tmin, target=a.target)
    cut = int(len(y) * a.split)
    mu, sd = X[:cut].mean(0), X[:cut].std(0) + 1e-9
    Xs = (X - mu) / sd
    w = ridge_fit(Xs[:cut], y[:cut], lam=a.lam)

    base = tmax if a.target == "tmax" else tmin
    preds = {"persist": persistence(base, idx),
             "clim": climatology(dates, base, idx, idx[:cut]),
             "ridge": ridge_predict(w, Xs)}
    res = {k: metrics(p[cut:], y[cut:]) for k, p in preds.items()}
    season = {k: by_season(dates, idx[cut:], p[cut:], y[cut:]) for k, p in preds.items()}
    skill = {k: round(1 - res[k]["MAE"] / res["persist"]["MAE"], 3) for k in res}

    out = {"城市": city, "目标": label + "温", "参数": {"λ": a.lam, "训练比例": a.split},
           "数据": {"起始": dates[0].isoformat(), "结束": dates[-1].isoformat(),
                    "天数": len(dates), "训练天数": int(cut), "测试天数": int(len(y) - cut)},
           "整体": res, "分季节": season, "相对持久性基线的技能分": skill}
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    chart_path = os.path.join(OUT, "chart.png")
    chart(dates, idx[cut:], y[cut:], {k: p[cut:] for k, p in preds.items()}, season, chart_path,
          f"{city}日{label}气温", unit)

    print(f"\n{city} · 预测{label}温 · λ={a.lam} · 训练 {cut} 天 / 测试 {len(y) - cut} 天")
    print(f"{'模型':<8}{'MAE':>8}{'RMSE':>8}{'技能分':>9}")
    for k, name in [("persist", "持久性"), ("clim", "气候平均"), ("ridge", "岭回归")]:
        print(f"{name:<8}{res[k]['MAE']:>8.2f}{res[k]['RMSE']:>8.2f}{skill[k]:>9.3f}")
    print(f"已输出 {OUT}/metrics.json、{OUT}/chart.png")

    if not a.no_open:
        try:
            os.startfile(chart_path)  # 自动打开图
        except Exception:
            pass


if __name__ == "__main__":
    main()
