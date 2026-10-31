import json, joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st
from pathlib import Path
from eq_features import add_t_days, live_features
from eq_live import fetch_usgs

MD, WINDOWS, THRS = Path("dashboard_models"), [7, 10, 15, 30], [4.0, 4.5, 5.0, 5.5]
st.set_page_config(page_title="Himalayan Earthquake Risk", layout="wide")
cfg = json.load(open(MD / "config.json")); F = cfg["features"]
lo, hi, med = (pd.Series(cfg[k])[F] for k in ("lo", "hi", "med"))

@st.cache_resource
def bundles(): return {(d, t): joblib.load(MD / f"{d}_{t}.joblib") for d in WINDOWS for t in THRS}

@st.cache_data(ttl=900, show_spinner="Fetching live catalogue…")      # auto-refreshes every 15 min
def usgs(bbox, minmag, days):
    return fetch_usgs(bbox, minmag, (pd.Timestamp.now("UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%d"))

def prep(X):
    X = X[F].replace([np.inf, -np.inf], np.nan)
    return X.clip(lo, hi, axis=1).fillna(med)

with st.sidebar:
    st.header("Catalogue")
    src = st.radio("Source", ["USGS (live)", "Upload NCS / own file"])
    minmag = st.number_input("Min magnitude", 1.0, 6.0, 4.0, 0.5)
    days = st.slider("History (days)", 90, 1500, 540)
    up = st.file_uploader("CSV/XLSX in the training-file format", ["csv", "xlsx"]) if src.startswith("Upload") else None
    if st.button("Refresh now"): st.cache_data.clear()
    win = st.selectbox("Window (days)", WINDOWS); thr = st.selectbox("Threshold magnitude ≥", THRS)

if up is not None:
    raw = pd.read_csv(up) if up.name.endswith("csv") else pd.read_excel(up)
    raw["time"] = pd.to_datetime(raw[["year", "month", "day", "hour", "min"]].rename(columns={"min": "minute"}))
elif src.startswith("USGS"):
    raw = usgs(tuple(cfg["bbox"]), minmag, days)
else: st.info("Upload a file to continue."); st.stop()

cat = add_t_days(raw)
if len(cat) < 60: st.error(f"Only {len(cat)} events — lower the min magnitude or extend the history."); st.stop()
now = live_features(cat, 1)[F]
drift = float(((now.replace([np.inf, -np.inf], np.nan).iloc[0] < lo) | (now.replace([np.inf, -np.inf], np.nan).iloc[0] > hi)).mean())
Xn, B = prep(now).values, bundles()
P = pd.DataFrame({d: [B[(d, t)]["model"].predict_proba(Xn)[0, 1] for t in THRS] for d in WINDOWS}, index=THRS)
R = pd.DataFrame({d: [P.loc[t, d] / B[(d, t)]["base_rate"] for t in THRS] for d in WINDOWS}, index=THRS)

st.title("Himalayan belt — earthquake risk, next 7–30 days")
st.caption(f"{len(cat)} events · latest {cat['time'].iloc[-1]:%Y-%m-%d %H:%M} UTC · model: {cfg['model']} · window = last 50 events spanning {now['T_Days'].iloc[0]:.1f} days")
if drift > 0.3: st.warning(f"{drift:.0%} of the 61 features fall outside the training range — this catalogue differs from the training one "
                           "(source / completeness). Treat scores as indicative only, or retrain on the same catalogue.")
b = B[(win, thr)]
c = st.columns(4)
c[0].metric(f"P(M ≥ {thr} in {win} d)", f"{P.loc[thr, win]:.1%}"); c[1].metric("vs. training base rate", f"×{R.loc[thr, win]:.2f}", f"base {b['base_rate']:.1%}", delta_color="off")
c[2].metric("Chronological-split AUC", f"{b['auc_chrono']:.2f}"); c[3].metric("Acc: random / chrono", f"{b['acc_rand']:.0%} / {b['acc_chrono']:.0%}")

l, r = st.columns(2)
fig = px.imshow(R, text_auto=".2f", color_continuous_scale="YlOrRd", labels=dict(x="Window (days)", y="Threshold M ≥", color="× base rate"), aspect="auto")
fig.update_layout(title="Risk relative to base rate (1.0 = typical)", yaxis=dict(type="category"), xaxis=dict(type="category"))
l.plotly_chart(fig, use_container_width=True)
rec = cat.tail(150 if len(cat) > 200 else len(cat) - 49)
rec = rec.assign(p=B[(win, thr)]["model"].predict_proba(prep(live_features(cat, len(rec))).values)[:, 1])
r.plotly_chart(px.line(rec, x="time", y="p", title=f"Model score over recent events (≥{thr}, {win} d)"), use_container_width=True)
st.plotly_chart(px.scatter(cat.tail(500), x="longitude", y="lattitude", color="magnitude", size=cat.tail(500)["magnitude"] ** 2,
                           color_continuous_scale="Turbo", hover_data=["time", "depth"], title="Recent events"), use_container_width=True)
with st.expander("Probability table & reliability"):
    st.dataframe(P.style.format("{:.1%}"))
    st.caption("Scores come from class-weighted models and are not calibrated probabilities; use the ×base-rate view. "
               "Random-split accuracy is inflated by overlapping windows — trust the chronological-split numbers. Research tool, not an official forecast.")
