# -*- coding: utf-8 -*-
"""
网页版（可选）：streamlit run app.py
"""
import json
import os

import streamlit as st

import run as R

st.set_page_config(page_title="气温小实验", page_icon="🌦️", layout="wide")
st.title("成都日最高气温：三个模型谁更准")
st.caption("数据 Open-Meteo｜任务：用前几天的气温预测当天最高温")

PATH = os.path.join("data", "chengdu.csv")
if not os.path.exists(PATH):
    st.error("还没有数据。先运行：python run.py")
    st.stop()


@st.cache_data(show_spinner="正在训练…")
def train():
    dates, tmax, tmin = R.load(PATH)
    X, y, idx = R.build_features(dates, tmax, tmin, "tmax")
    cut = int(len(y) * 0.8)
    mu, sd = X[:cut].mean(0), X[:cut].std(0) + 1e-9
    Xs = (X - mu) / sd
    w = R.ridge_fit(Xs[:cut], y[:cut], lam=1.0)
    preds = {"persist": R.persistence(tmax, idx),
             "clim": R.climatology(dates, tmax, idx, idx[:cut]),
             "ridge": R.ridge_predict(w, Xs)}
    return dates, idx, y, preds, cut


dates, idx, y, preds, cut = train()

m = json.load(open(os.path.join(R.OUT, "metrics.json"), encoding="utf-8"))["整体"]
c1, c2, c3 = st.columns(3)
c1.metric("我写的模型", f"{m['ridge']['MAE']} ℃")
c2.metric("明天=今天", f"{m['persist']['MAE']} ℃")
c3.metric("历年同期平均", f"{m['clim']['MAE']} ℃")

st.image(os.path.join(R.OUT, "chart.png"))

st.subheader("按日期查一条")
i = st.slider("测试集里的第几天", 0, len(y) - cut - 1, len(y) - cut - 10)
j = cut + i
st.write(f"**{dates[idx[j]]}**　实况 {y[j]:.1f} ℃")
st.table([{"模型": name, "预测": round(float(preds[k][j]), 1),
           "误差": round(abs(float(preds[k][j]) - y[j]), 1)}
          for k, name in [("persist", "明天=今天"), ("clim", "历年同期平均"), ("ridge", "我写的模型")]])

st.caption("最小原型，不是业务预报：只靠前几天温度外推，误差就在 2℃ 左右。")
