# Freight Rate Prediction

Predicts `posted_rate` for 12,000 validation loads (Nov-Dec 2025) plus a fixed December lane chart.

## Run
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/Mac: source .venv/bin/activate)
python -m pip install -r requirements.txt
python src/pipeline.py          # cleaning, time-based CV, final fit -> validation_predictions.csv + data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python src/make_report.py       # -> outputs/report.docx
```
Inputs live in `data/` (`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`).

## Approach (details in `outputs/report.docx`)
- Forward-chaining time CV, because the validation set is a future period.
- Cleaning: sign-flipped/missing weight, missing market_index, ~1.7% corrupted labels (dropped from training only).
- `quote_signal` excluded: it mirrors the target with a sign that flips by month (leaky proxy).
- HistGradientBoosting on log(rate per mile), multiplied by distance.
