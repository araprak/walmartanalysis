# Walmart Weekly Sales Analysis

An exploratory Python analysis of weekly store sales. The project compares store-level sales distributions, examines holiday and regular weeks, and checks associations with temperature, fuel price, and CPI.

[Open the analysis notebook](walmart-weekly-sales.ipynb) to see editable charts and the data checks. The saved charts use a small, fictional dataset to illustrate the workflow. Supply your own compatible CSV to run the notebook with observed sales.

## Run the notebook

Install Python 3 and the notebook dependencies.

```bash
python -m pip install jupyterlab pandas numpy matplotlib
jupyter lab walmart-weekly-sales.ipynb
```

Set `INPUT_CSV` to a local file with one row per store and week. Required columns are `Store`, `Date`, `Weekly_Sales`, and `Holiday_Flag` (0 or 1). The notebook also uses `Temperature`, `Fuel_Price`, and `CPI` when available. The original CSV referenced by the legacy script is unavailable in this repository.

The notebook compares mean weekly sales within each store and reports the number of observations behind each comparison. Holiday patterns and correlations are descriptive. Store and time effects can influence these comparisons.

The [original script](walmartanalysis.py) preserves the earlier pandas, Seaborn, Matplotlib, and SciPy analysis.
