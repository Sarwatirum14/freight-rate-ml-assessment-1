"""Freight rate pipeline: cleaning, features, time-based CV, final fit, predictions.

Run from repo root:  python src/pipeline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
SEED = 42

# quote_signal is deliberately NOT used (see report: it is a target proxy whose sign flips by month).
# Day-of-week and raw coordinates were tested and removed: both worsened forward-chaining CV error.
NUM = ["distance", "weight", "market_index"]
CAT = ["equipment", "pickup", "delivery"]
FEATURES = NUM + CAT


def clean(df: pd.DataFrame, mi_by_date: pd.Series | None = None) -> pd.DataFrame:
    """Feature-side cleaning; applied identically to train / validation / December inputs."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["weight"] = d["weight"].abs()  # 292 train rows have sign-flipped weight
    d["weight"] = d["weight"].fillna(d.groupby("equipment")["weight"].transform("median"))
    d["weight"] = d["weight"].fillna(31_000.0)
    if "market_index" not in d:
        d["market_index"] = np.nan
    if mi_by_date is not None:  # market_index is effectively a date-level series
        d["market_index"] = d["market_index"].fillna(d["date"].dt.strftime("%Y-%m-%d").map(mi_by_date))
    d["market_index"] = d["market_index"].fillna(d.groupby("date")["market_index"].transform("median"))
    return d


def featurize(d: pd.DataFrame, cats: dict[str, list[str]]) -> pd.DataFrame:
    x = d.copy()
    for c in CAT:
        x[c] = pd.Categorical(x[c], categories=cats[c])
    return x[FEATURES]


def make_cats(*frames: pd.DataFrame) -> dict[str, list[str]]:
    all_ = pd.concat(frames)
    return {"equipment": sorted(all_.equipment.unique()), "pickup": sorted(all_.pickup.unique()),
            "delivery": sorted(all_.delivery.unique())}


PARAMS = dict(loss="squared_error", learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
              min_samples_leaf=20, l2_regularization=10.0, categorical_features="from_dtype",
              early_stopping=False, random_state=SEED)


def fit_model(X: pd.DataFrame, y_log: np.ndarray, params=PARAMS) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**params).fit(X, y_log)


def predict_rate(model, X: pd.DataFrame, distance: pd.Series) -> np.ndarray:
    """Model predicts log(rate per mile); convert back to dollars."""
    return np.exp(model.predict(X)) * distance.to_numpy()


def flag_label_outliers(d: pd.DataFrame, cats, k: float = 5.0) -> pd.Series:
    """Out-of-fold residual screen for corrupted posted_rate labels (random 5-fold, log rate/mile)."""
    X = featurize(d, cats)
    y = np.log(d["posted_rate"] / d["distance"]).to_numpy()
    fold = np.random.RandomState(SEED).randint(0, 5, len(d))
    oof = np.zeros(len(d))
    p = {**PARAMS, "max_iter": 200, "learning_rate": 0.1}
    for f in range(5):
        m = fit_model(X[fold != f], y[fold != f], p)
        oof[fold == f] = m.predict(X[fold == f])
    res = y - oof
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    return pd.Series(np.abs(res - np.median(res)) > k * mad, index=d.index)


def metrics(y, p) -> dict:
    e = p - y
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAPE%": float(np.mean(np.abs(e) / y) * 100), "WAPE%": float(np.abs(e).sum() / y.sum() * 100)}


def cross_validate(train: pd.DataFrame, cats) -> pd.DataFrame:
    """Forward-chaining time split: train on everything before the block, predict the next 2 months
    (mirrors the real task: train Jan-Oct, predict Nov-Dec)."""
    m = train["date"].dt.month
    folds = [(4, [5, 6]), (6, [7, 8]), (8, [9, 10])]
    variants = {
        "baseline: median $/mile by equipment x distance": None,
        "boosted trees without market_index": [c for c in FEATURES if c != "market_index"],
        "boosted trees (FINAL)": FEATURES,
        "boosted trees + quote_signal (rejected)": "qs",
    }
    rows = []
    for last_train, val_months in folds:
        tr, va = train[m <= last_train], train[m.isin(val_months)]
        tr_c = tr[~tr["outlier"]]
        for name, cols in variants.items():
            if cols is None:
                med = (tr_c.posted_rate / tr_c.distance).groupby(tr_c["equipment"]).median()
                pred = va["equipment"].map(med).to_numpy() * va["distance"].to_numpy()
            else:
                Xtr, Xva = featurize(tr_c, cats), featurize(va, cats)
                if cols == "qs":
                    Xtr["quote_signal"] = tr_c["quote_signal"].to_numpy()
                    Xva["quote_signal"] = va["quote_signal"].to_numpy()
                else:
                    Xtr, Xva = Xtr[cols], Xva[cols]
                mdl = fit_model(Xtr, np.log(tr_c.posted_rate / tr_c.distance))
                pred = predict_rate(mdl, Xva, va["distance"])
            for subset, mask in (("raw", np.ones(len(va), bool)), ("clean", ~va["outlier"].to_numpy())):
                rows.append({"fold": f"train<=m{last_train} -> val m{val_months}", "model": name, "eval": subset,
                             **metrics(va["posted_rate"].to_numpy()[mask], pred[mask])})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tr = pd.read_csv(DATA / "train_test.csv")
    va = pd.read_csv(DATA / "validation.csv")
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")
    tmpl = pd.read_csv(DATA / "validation_predictions_template.csv")

    mi_by_date = pd.concat([tr, va]).groupby("date")["market_index"].median()
    cats = make_cats(tr, va, dec)
    tr = clean(tr, mi_by_date)
    va_c = clean(va, mi_by_date)

    tr["outlier"] = flag_label_outliers(tr, cats)
    print(f"label outliers flagged: {tr['outlier'].sum()} ({tr['outlier'].mean():.2%})")

    cv = cross_validate(tr, cats)
    cv.to_csv(OUT / "cv_results.csv", index=False)
    summary = cv.groupby(["model", "eval"])[["MAE", "RMSE", "MAPE%", "WAPE%"]].mean().round(3)
    summary.to_csv(OUT / "cv_summary.csv")
    print(summary)

    # Final fit on all cleaned development data (Jan-Oct)
    tc = tr[~tr["outlier"]]
    model = fit_model(featurize(tc, cats), np.log(tc.posted_rate / tc.distance))
    from sklearn.inspection import permutation_importance
    Xs = featurize(tc, cats).sample(5000, random_state=SEED)
    ys = np.log(tc.posted_rate / tc.distance).loc[Xs.index]
    pi = permutation_importance(model, Xs, ys, n_repeats=3, random_state=SEED, scoring="neg_mean_absolute_error")
    pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False).round(4).to_csv(OUT / "feature_importance.csv")

    pred = predict_rate(model, featurize(va_c, cats), va_c["distance"])

    # Diagnostic: does quote_signal still track the rate in Nov-Dec? Compare it with our (quote-free)
    # predicted rate per mile, month by month. ~0 means the signal is uninformative in the validation period.
    qs = pd.concat([
        tr.assign(prpm=np.exp(model.predict(featurize(tr, cats))))[["date", "quote_signal", "prpm"]],
        va_c.assign(prpm=pred / va_c["distance"])[["date", "quote_signal", "prpm"]]])
    qs["month"] = qs["date"].dt.strftime("%Y-%m")
    qs.groupby("month").apply(lambda g: g["quote_signal"].corr(g["prpm"]), include_groups=False)         .rename("corr_quote_signal_vs_predicted_rate_per_mile").round(3).to_csv(OUT / "quote_signal_regime.csv")
    out = tmpl[["load_id"]].merge(
        pd.DataFrame({"load_id": va["load_id"], "predicted_rate": pred.round(2)}), on="load_id", how="left")
    assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all() and len(out) == 12000
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    # December chart: fixed lane inputs; only the date varies. market_index comes from validation.csv's
    # per-date values (the only date-varying driver available); quote_signal is not used.
    dec_c = clean(dec, mi_by_date)
    dec["predicted_rate"] = predict_rate(model, featurize(dec_c, cats), dec_c["distance"]).round(2)
    dec.to_csv(DATA / "december_chart_inputs.csv", index=False)
    json.dump({"outliers_flagged": int(tr["outlier"].sum())}, open(OUT / "run_info.json", "w"))
    print(dec[["date", "predicted_rate"]].describe())


if __name__ == "__main__":
    main()
