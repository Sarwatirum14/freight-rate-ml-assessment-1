"""Builds outputs/report.docx from pipeline outputs. Run after src/pipeline.py and score.py."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

import pandas as pd
from docx import Document
from docx.shared import Inches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
tr = pd.read_csv(ROOT / "data" / "train_test.csv")
tr["rpm"] = tr.posted_rate / tr.distance
tr["month"] = tr.date.str[:7]
ok = tr[(tr.rpm > 1.2) & (tr.rpm < 4)]
corr = pd.Series({m: g.rpm.corr(g.quote_signal) for m, g in ok.groupby("month")})

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar([m[5:] for m in corr.index], corr.values, color=["#064A56" if v > 0 else "#B5483A" for v in corr.values])
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("corr(rate/mile, quote_signal)")
ax.set_xlabel("2025 month")
ax.set_title("quote_signal relationship flips sign by month", loc="left", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "quote_signal_by_month.png", dpi=160)
plt.close(fig)

cv = pd.read_csv(OUT / "cv_summary.csv")
n_out = json.load(open(OUT / "run_info.json"))["outliers_flagged"]
regime = pd.read_csv(OUT / "quote_signal_regime.csv", index_col=0).iloc[:, 0]
imp = pd.read_csv(OUT / "feature_importance.csv", index_col=0).iloc[:, 0]

d = Document()
d.add_heading("Freight Rate Prediction: Validation and Split Approach", 0)

d.add_heading("1. Data and key findings", 1)
for t in [
    "48,000 labeled loads (2025-01-01 to 2025-10-31, 64 cities, 3 equipment types); 12,000 validation loads dated 2025-11-01 to 2025-12-31.",
    "The validation set lies entirely in the future of the development data, so the real task is forecasting forward in time.",
    "Rate scales almost linearly with distance (corr 0.91), so the target modelled is log(rate per mile), median about $2.15/mile. Reefer and Flatbed price above Dry Van.",
    "market_index is a daily level (about 0.68 to 1.47) that tracks the daily rate per mile (daily corr about 0.64).",
]:
    d.add_paragraph(t, style="List Bullet")

d.add_heading("2. Data-quality issues and handling", 1)
for t in [
    "weight: 292 negative values (sign flipped, magnitudes 5,000-47,500) were converted with abs(); 300 missing values were filled with the equipment median. Validation has 165 missing weights, handled the same way.",
    "market_index: 374 missing values in train (249 in validation). It is a date-level series, so gaps are filled with the median of that date's other loads.",
    f"Corrupted labels: about {n_out / 480:.1f}% of rows ({n_out}) have implausible rate per mile (0.33 to 14.1 vs a typical 1.7 to 3.2). They were flagged with an out-of-fold residual screen (5-fold model, residual beyond 5 robust sigma) and dropped from training only.",
    "quote_signal is a leaky proxy and was NOT used. In Jan-Mar, Jun and Sep it equals the rate per mile almost exactly (corr about +0.99); in Apr, May, Jul and Oct it is the mirror image (rate per mile + quote_signal is roughly constant, corr about -0.99); Aug is uncorrelated. As a check on the validation period, I correlated quote_signal with the quote-free predicted rate per mile: Nov = %.3f and Dec = %.3f, i.e. no relationship, the same as Aug. So in Nov-Dec the signal carries no usable information, and a model trained on it would apply a stale sign. It is also absent from the December chart inputs." % (regime["2025-11"], regime["2025-12"]) + "",
    "No duplicate load_ids or duplicate rows; no pickup == delivery rows; categories identical across files.",
    "December chart inputs contain no market_index (or coordinates, which the final model does not use). market_index uses each date's value from validation.csv.",
]:
    d.add_paragraph(t, style="List Bullet")
d.add_picture(str(OUT / "quote_signal_by_month.png"), width=Inches(5.8))

d.add_heading("3. Train / validation split", 1)
d.add_paragraph(
    "A random K-fold split would be misleading: the test set is two future months, and rates and market_index drift over time. "
    "I used forward-chaining time splits, always training strictly on earlier months and predicting the next 2-month block, "
    "mirroring the real Jan-Oct to Nov-Dec setup:")
for t in ["Train Jan-Apr, validate May-Jun", "Train Jan-Jun, validate Jul-Aug", "Train Jan-Aug, validate Sep-Oct"]:
    d.add_paragraph(t, style="List Bullet")
d.add_paragraph(
    "Outlier cleaning is applied to training rows only; each fold is reported on raw labels (honest) and on labels with the "
    "corrupted rows removed (signal quality). Month and day-of-year are deliberately not features, since they cannot "
    "extrapolate to Nov-Dec; market_index carries the time-varying level instead. The final model is refit on all of Jan-Oct.")

d.add_heading("4. Model and results", 1)
d.add_paragraph(
    "Model: gradient-boosted trees (scikit-learn HistGradientBoostingRegressor, squared-error loss on the cleaned labels) predicting log(rate per mile), "
    "multiplied back by distance. Features: distance, weight, market_index, equipment, pickup city, delivery city. Day-of-week, raw coordinates, day-of-year and rolling market_index were each tested and removed because they did not improve (day-of-week and coordinates worsened) forward-chaining error. Squared-error loss on the log target beat absolute-error loss once the corrupted labels were removed. "
    "Boosted trees capture lane x equipment x distance interactions without manual encoding, and corrupted labels are removed before fitting. Mean over the three folds:")
tbl = d.add_table(rows=1, cols=6)
tbl.style = "Light Grid Accent 1"
for i, h in enumerate(["Model", "Eval", "MAE $", "RMSE $", "MAPE %", "WAPE %"]):
    tbl.rows[0].cells[i].text = h
for _, r in cv.iterrows():
    c = tbl.add_row().cells
    c[0].text, c[1].text = r["model"], r["eval"]
    for i, k in enumerate(["MAE", "RMSE", "MAPE%", "WAPE%"]):
        c[2 + i].text = f"{r[k]:.2f}"
d.add_paragraph()
d.add_paragraph(
    "The final model more than halves the error of the equipment/distance baseline, and market_index gives a further gain. "
    "The quote_signal variant scores slightly better in CV but is rejected for the reason in section 2. Raw RMSE is dominated by "
    "the corrupted labels, which no model can predict. Permutation importance (increase in log-MAE): "
    + ", ".join(f"{k} {v:.4f}" for k, v in imp.head(5).items()) + ".")

d.add_heading("5. Fixed December prediction chart (produced by score.py)", 1)
d.add_paragraph("Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb; only the date varies. Predictions are about $801-$823, "
                "driven by the market_index values for each December date (no day-of-week effect, since it was not predictive in CV).")
d.add_picture(str(ROOT / "scorer_results" / "candidate_december.png"), width=Inches(6.4))

d.add_heading("6. Reproducing", 1)
d.add_paragraph("python -m pip install -r requirements.txt; python src/pipeline.py; python score.py --predictions validation_predictions.csv "
                "--december-predictions data/december_chart_inputs.csv; python src/make_report.py")
d.save(OUT / "report.docx")
print("saved", OUT / "report.docx")
