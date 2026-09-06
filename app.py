import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

warnings.filterwarnings("ignore")

OUTPUTS = {
    "GDP_growth": "GDP_growth",
    "Inflation_rate": "Inflation_rate",
    "Real_wage": "Real_wage",
    "Labor_productivity": "Labor_productivity",
    "Exchange_rate_PHP_to_USD": "Exchange_rate_PHP_to_USD",
}
INDEPENDENTS = [
    "Private_investment", "Public_investment", "Employment_rate", "HCI",
    "Capital_formation", "Real_wage", "GNI", "Labor_productivity",
    "GDP_growth", "Gov_exp", "Exchange_rate_PHP_to_USD", "Interest_rate",
    "Inflation_rate", "CSPI", "Business_confidence",
]
FORECAST_VARIABLES = [
    "GDP_growth", "Inflation_rate", "Real_wage", "Labor_productivity",
    "Exchange_rate_PHP_to_USD", "Capital_formation", "Business_confidence",
    "Private_investment", "Public_investment", "Employment_rate", "HCI",
    "GNI", "Gov_exp", "Interest_rate", "CSPI",
]


def normalise(value):
    return "".join(c.lower() for c in str(value) if c.isalnum())


def resolve_columns(frame):
    lookup = {normalise(c): c for c in frame.columns}
    aliases = {
        "Exchange_rate_PHP_to_USD": ["exchangerate", "exchangeratephptousd"],
        "Public_investment": ["publicinvestmentt"],
    }
    resolved = {}
    for name in set(INDEPENDENTS) | set(OUTPUTS.values()):
        candidates = [name] + aliases.get(name, [])
        resolved[name] = next((lookup[normalise(c)] for c in candidates if normalise(c) in lookup), None)
    return resolved


def parse_years(frame, column):
    values = frame[column]
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values.astype("Int64").astype(str), format="%Y", errors="coerce")
    return pd.to_datetime(values, errors="coerce")


def clean_data(frame, date_column, fill_missing=True):
    result = frame.copy()
    result[date_column] = parse_years(result, date_column)
    result = result.dropna(subset=[date_column]).sort_values(date_column).set_index(date_column)
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.select_dtypes(include=[np.number])
    if fill_missing:
        result = result.interpolate(limit_direction="both").ffill().bfill()
    return result


def safe_log(series):
    return np.log(series.where(series > 0))


def equation_features(frame, columns):
    def column(name):
        source = columns.get(name)
        return frame[source] if source else pd.Series(np.nan, index=frame.index)

    result = pd.DataFrame(index=frame.index)
    result["eq_capital_formation"] = safe_log(column("Private_investment")) + safe_log(column("Public_investment"))
    result["eq_labor_productivity"] = column("Employment_rate") + column("HCI") + column("Capital_formation") + column("Real_wage") + column("Inflation_rate") + safe_log(column("GNI"))
    result["eq_gdp_growth"] = column("Labor_productivity").diff()
    result["eq_business_confidence"] = column("GDP_growth") + safe_log(column("Gov_exp")) + column("Interest_rate") + column("Inflation_rate") + safe_log(column("Exchange_rate_PHP_to_USD")) + safe_log(column("CSPI"))
    result["eq_private_investment"] = safe_log(column("Private_investment")).diff() + column("GDP_growth") + column("Business_confidence")
    return result.replace([np.inf, -np.inf], np.nan)


def metric_values(actual, predicted, training):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    scale = np.mean(np.abs(np.diff(np.asarray(training, dtype=float))))
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    denominator = np.where(np.abs(actual) < 1e-8, 1e-8, np.abs(actual))
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MAPE": float(np.mean(np.abs((actual - predicted) / denominator)) * 100),
        "MASE": float(np.mean(np.abs(actual - predicted)) / scale),
    }


def arima_forecast(train, steps):
    return np.column_stack([ARIMA(train[column], order=(1, 1, 1), trend="t").fit().forecast(steps) for column in train.columns])


def var_forecast(train, steps):
    fitted = VAR(train).fit(maxlags=min(4, max(1, len(train) // 10)), ic="aic", trend="c")
    return fitted.forecast(train.values[-fitted.k_ar:], steps)


def vecm_forecast(train, steps):
    fitted = VECM(train, k_ar_diff=1, coint_rank=min(1, len(train.columns) - 1), deterministic="co").fit()
    return fitted.predict(steps=steps)


def garch_forecast(train, steps):
    from arch import arch_model
    forecasts = []
    for column in train.columns:
        changes = train[column].diff().dropna()
        fitted = arch_model(changes, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
        mean_changes = fitted.forecast(horizon=steps).mean.iloc[-1].to_numpy()
        forecasts.append(train[column].iloc[-1] + np.cumsum(mean_changes))
    return np.column_stack(forecasts)


def xgb_features(frame, targets, exogenous, equations, index, history):
    values = {}
    for lag in (1, 2, 3):
        for target in targets:
            values[f"{target}_lag{lag}"] = history[target].iloc[-lag]
    source = frame.loc[index] if index in frame.index else frame.iloc[-1]
    for column in exogenous + equations:
        values[column] = source.get(column, frame[column].iloc[-1] if column in frame else 0)
    return values


def xgboost_forecast(frame, train, targets, exogenous, equations, steps, future=False):
    from xgboost import XGBRegressor
    rows, labels = [], {target: [] for target in targets}
    for position in range(3, len(train)):
        rows.append(xgb_features(frame, targets, exogenous, equations, train.index[position], train.iloc[:position]))
        for target in targets:
            labels[target].append(train.iloc[position][target])
    design = pd.DataFrame(rows).fillna(0)
    models = {
        target: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=2, verbosity=0).fit(design, labels[target])
        for target in targets
    }
    history = train.copy()
    indexes = frame.index[-steps:] if future else frame.index[len(train):len(train) + steps]
    predictions = []
    for index in indexes:
        row = pd.DataFrame([xgb_features(frame, targets, exogenous, equations, index, history)]).fillna(0)
        prediction = {target: float(models[target].predict(row)[0]) for target in targets}
        predictions.append([prediction[target] for target in targets])
        history = pd.concat([history, pd.DataFrame([prediction], index=[index])])
    return np.asarray(predictions)


def run_method(method, frame, train, targets, exogenous, equations, steps):
    if method == "ARIMA":
        return arima_forecast(train[targets], steps)
    if method == "VAR":
        return var_forecast(train[targets], steps)
    if method == "VECM":
        return vecm_forecast(train[targets], steps)
    if method == "GARCH":
        return garch_forecast(train[targets], steps)
    return xgboost_forecast(frame, train[targets], targets, exogenous, equations, steps)


st.set_page_config(page_title="Macroeconomic Forecasting Lab", layout="wide")
st.title("Macroeconomic Forecasting Lab")
st.caption("Equation-informed comparison of ARIMA, VECM, VAR, GARCH and XGBoost")

csv_path = Path(__file__).with_name("R1_model.csv")
if not csv_path.exists():
    st.error(f"Required input file not found: {csv_path.name}")
    st.stop()

raw = pd.read_csv(csv_path)
with st.sidebar:
    st.header("Forecast settings")
    with st.form("forecast_settings"):
        st.caption("Choose variables, methods, and horizon, then submit the settings.")
        forecast_outputs = st.multiselect(
            "Forecast outputs",
            FORECAST_VARIABLES,
            default=list(OUTPUTS.keys()),
        )
        forecast_inputs = st.multiselect(
            "Forecast inputs",
            FORECAST_VARIABLES,
            default=FORECAST_VARIABLES,
        )
        selected_methods = st.multiselect(
            "Methods to compare",
            ["ARIMA", "VAR", "VECM", "GARCH", "XGBoost"],
            default=["ARIMA", "VAR", "VECM"],
        )
        forecast_horizon = st.number_input("Forecast horizon (years)", min_value=1, max_value=5, value=5, step=1)
        test_fraction = st.slider("Backtest share", 0.1, 0.4, 0.2, 0.05)
        run_comparison = st.form_submit_button("Run comparison", type="primary", use_container_width=True)

resolved = resolve_columns(raw)
missing = [name for name, source in resolved.items() if source is None]
if missing:
    st.error("Missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

date_column = "Year" if "Year" in raw.columns else raw.columns[0]
data = clean_data(raw, date_column)
preview_data = clean_data(raw, date_column, fill_missing=False)
input_columns = [column for column in preview_data.columns]
preview = preview_data.loc[preview_data.index.year.isin([2025, 2026, 2027]), input_columns].copy()
preview.index = preview.index.year
preview.index.name = "Year"
if preview.empty:
    st.error("The CSV must contain rows for 2025, 2026, and 2027.")
    st.stop()

st.header("Editable input and output data preview")
st.caption("Edit any of the 16 non-date variables for 2025-2027. Year is shown only as the row index.")
edited_preview = st.data_editor(preview, num_rows="fixed", use_container_width=True, key="input_preview")
for year in edited_preview.index:
    row_mask = data.index.year == int(year)
    for column in input_columns:
        value = pd.to_numeric(edited_preview.loc[year, column], errors="coerce")
        if pd.notna(value):
            data.loc[row_mask, column] = value

if not run_comparison:
    st.info("Choose your forecast settings, then click Run comparison in the sidebar.")
    st.stop()

if not forecast_outputs or not forecast_inputs or not selected_methods:
    st.warning("Select at least one output, input, and method in Forecast settings.")
    st.stop()

equations = equation_features(data, resolved)
model_frame = pd.concat([data, equations], axis=1)
targets = [resolved[name] for name in forecast_outputs]
target_labels = forecast_outputs
equation_columns = list(equations.columns)
exogenous = [resolved[name] for name in forecast_inputs]
exogenous = [column for column in exogenous if column in model_frame.columns and column not in targets]
feature_columns = exogenous + equation_columns

if len(model_frame) < 30:
    st.error("At least 30 observations are recommended.")
    st.stop()

split = max(15, int(len(model_frame) * (1 - test_fraction)))
train, test = model_frame.iloc[:split], model_frame.iloc[split:]
predictions, errors = {}, {}
with st.spinner("Fitting models and running the backtest..."):
    for method in selected_methods:
        try:
            predictions[method] = run_method(method, model_frame, train, targets, exogenous, equation_columns, len(test))
            errors[method] = pd.DataFrame([metric_values(test[target], predictions[method][:, position], train[target]) for position, target in enumerate(targets)], index=target_labels)
        except Exception as error:
            st.warning(f"{method} could not be fitted: {error}")

if not errors:
    st.stop()

st.header("Backtest results")
metric_name = st.selectbox("Metric to rank", ["RMSE", "MAE", "MAPE", "MASE"])
ranking = pd.DataFrame({method: result[metric_name] for method, result in errors.items()}, index=target_labels)
st.dataframe(ranking.style.format("{:.4f}"), use_container_width=True)
best = pd.DataFrame(index=target_labels)
for metric in ["RMSE", "MAE", "MAPE", "MASE"]:
    scores = pd.DataFrame({method: result[metric] for method, result in errors.items()}, index=target_labels)
    best[f"Best model by {metric}"] = scores.idxmin(axis=1)
best.insert(0, "Target", best.index)
st.subheader("Best forecasting method")
st.dataframe(best.reset_index(drop=True), hide_index=True, use_container_width=True)

forecast_years = list(range(2027, 2027 + int(forecast_horizon)))
forecast_index = pd.to_datetime([f"{year}-12-31" for year in forecast_years])
future_base = pd.DataFrame([data.loc[data.index.year == 2027].iloc[-1].to_dict()] * len(forecast_years), index=forecast_index).reindex(columns=data.columns)
future_frame = pd.concat([future_base, equation_features(future_base, resolved)], axis=1)
forecast_frame = pd.concat([model_frame, future_frame])
future_predictions = {}
for method in predictions:
    try:
        future_predictions[method] = run_method(method, forecast_frame, model_frame, targets, exogenous, equation_columns, len(forecast_years))
    except Exception as error:
        st.warning(f"{method} future forecast failed: {error}")

if future_predictions:
    forecast_table = pd.DataFrame(index=forecast_years)
    forecast_table.index.name = "Year"
    for position, target in enumerate(target_labels):
        for method, values in future_predictions.items():
            forecast_table[(target, method)] = values[:, position]
    forecast_table.columns = pd.MultiIndex.from_tuples(forecast_table.columns)
    st.header(f"{forecast_years[0]}-{forecast_years[-1]} forecasts")
    st.dataframe(forecast_table.style.format("{:.4f}"), use_container_width=True)
    download = forecast_table.copy()
    download.columns = [f"{target}_{method}" for target, method in download.columns]
    st.download_button("Download forecasts as CSV", download.reset_index().to_csv(index=False), f"macroeconomic_forecasts_{forecast_years[0]}_{forecast_years[-1]}.csv", "text/csv")
