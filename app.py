import ast
import warnings
import hmac
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


def filter_multicollinear_variables(frame, variables, threshold=0.95):
    ordered = []
    for variable in variables:
        if variable not in frame.columns:
            continue
        if not ordered:
            ordered.append(variable)
            continue
        correlations = []
        for existing in ordered:
            corr = frame[[variable, existing]].corr().iloc[0, 1]
            correlations.append(abs(float(corr)) if pd.notna(corr) else 0.0)
        if max(correlations, default=0.0) < threshold:
            ordered.append(variable)
    return ordered


def equation_features(frame, columns, keep_variables=None):
    keep_set = set(keep_variables) if keep_variables is not None else None

    def column(name):
        source = columns.get(name)
        if source is None:
            return pd.Series(np.nan, index=frame.index)
        if keep_set is not None and source not in keep_set:
            return pd.Series(np.nan, index=frame.index)
        return frame[source]

    result = pd.DataFrame(index=frame.index)
    result["eq_capital_formation"] = safe_log(column("Private_investment")) + safe_log(column("Public_investment"))
    result["eq_labor_productivity"] = column("Employment_rate") + column("HCI") + column("Capital_formation") + column("Real_wage") + column("Inflation_rate") + safe_log(column("GNI"))
    result["eq_gdp_growth"] = column("Labor_productivity").diff()
    result["eq_business_confidence"] = column("GDP_growth") + safe_log(column("Gov_exp")) + column("Interest_rate") + column("Inflation_rate") + safe_log(column("Exchange_rate_PHP_to_USD")) + safe_log(column("CSPI"))
    result["eq_private_investment"] = safe_log(column("Private_investment")).diff() + column("GDP_growth") + column("Business_confidence")
    return result.replace([np.inf, -np.inf], np.nan)


def evaluate_user_equation(expression, frame):
    expr = (expression or "").strip()
    if not expr:
        return None

    if expr.count("=") != 1:
        raise ValueError("Equation must contain exactly one '=' with the dependent variable on the left and expression on the right.")

    lhs, rhs = [part.strip() for part in expr.split("=", 1)]
    if not lhs or not rhs:
        raise ValueError("Equation must have both a dependent variable and a right-hand expression.")

    if not lhs.replace("_", "").isalnum():
        raise ValueError("Dependent variable name can contain letters, numbers, and underscores only.")

    try:
        parsed = ast.parse(rhs, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid syntax in equation: {exc.msg}") from exc

    allowed_funcs = {"log": np.log, "sqrt": np.sqrt, "abs": np.abs}
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )

    def validate(node):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")
        if isinstance(node, ast.BinOp):
            validate(node.left)
            validate(node.right)
        elif isinstance(node, ast.UnaryOp):
            validate(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs:
                raise ValueError("Only log(), sqrt(), and abs() are allowed in custom equations.")
            for arg in node.args:
                validate(arg)
            for keyword in node.keywords:
                validate(keyword.value)
        elif isinstance(node, ast.Name):
            if node.id not in set(frame.columns) | set(allowed_funcs):
                raise ValueError(f"Unknown variable or function: {node.id}")

    validate(parsed)

    local_context = {name: frame[name] for name in frame.columns}
    local_context.update(allowed_funcs)
    result = eval(compile(parsed, "<custom_equation>", "eval"), {"__builtins__": {}}, local_context)
    output = result if isinstance(result, pd.Series) else pd.Series(result, index=frame.index)
    output = output.replace([np.inf, -np.inf], np.nan)
    return lhs, output


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


def select_variance_stable_vector_data(frame, targets, equations, columns, threshold=0.95, max_equations=2):
    selected = list(targets)
    if not equations:
        return frame[selected]

    equation_columns = [column for column in equations if column in frame.columns and column not in targets]
    if not equation_columns:
        return frame[selected]

    candidate_sources = [source for source in columns.values() if source is not None and source not in targets]
    filtered_sources = filter_multicollinear_variables(frame, candidate_sources, threshold=threshold)
    filtered_equations = equation_features(frame, columns, keep_variables=set(filtered_sources))
    equation_candidates = [column for column in equation_columns if column in filtered_equations.columns and filtered_equations[column].notna().any()]

    if not equation_candidates:
        return pd.concat([frame[selected], filtered_equations.iloc[:, :1]], axis=1)

    kept = equation_candidates[:max_equations]
    if not kept:
        kept = [filtered_equations.columns[0]]
    return pd.concat([frame[selected], filtered_equations[kept]], axis=1)


def run_method(method, frame, train, targets, exogenous, equations, steps, use_equations_in_var_vecm=False, resolved_columns=None):
    if method in ("VAR", "VECM"):
        vector_data = train[targets]
        if use_equations_in_var_vecm and equations:
            resolved_columns = resolved_columns or {target: target for target in targets}
            vector_data = select_variance_stable_vector_data(train, targets, equations, resolved_columns)
        try:
            if method == "VAR":
                return var_forecast(vector_data, steps)[:, :len(targets)]
            return vecm_forecast(vector_data, steps)[:, :len(targets)]
        except Exception:
            if use_equations_in_var_vecm and equations:
                fallback = train[targets]
                if method == "VAR":
                    return var_forecast(fallback, steps)[:, :len(targets)]
                return vecm_forecast(fallback, steps)[:, :len(targets)]
            raise

    signal_columns = list(targets) + [column for column in equations if column not in targets]
    feature_set = train[signal_columns] if signal_columns else train[targets]

    if method == "ARIMA":
        return arima_forecast(feature_set, steps)[:, :len(targets)]
    if method == "GARCH":
        return garch_forecast(feature_set, steps)[:, :len(targets)]
    return xgboost_forecast(frame, train[targets], targets, exogenous, equations, steps)


st.set_page_config(page_title="Macroeconomic Forecasting Lab", layout="wide")

st.markdown(
    """
    <style>
        div[data-baseweb="tag"],
        [role="option"][aria-selected="true"],
        [role="listbox"] [aria-selected="true"],
        [aria-selected="true"] {
            background-color: #1ed760 !important;
            color: #0b0f0d !important;
            border: 1px solid rgba(30, 215, 96, 0.7) !important;
        }
        div[data-baseweb="tag"] span,
        [role="option"][aria-selected="true"] span,
        [role="listbox"] [aria-selected="true"] span,
        [aria-selected="true"] span {
            color: #0b0f0d !important;
        }
        div[data-baseweb="select"] [role="combobox"],
        div[data-baseweb="select"] {
            border-color: rgba(30, 215, 96, 0.7) !important;
            box-shadow: 0 0 0 1px rgba(30, 215, 96, 0.5) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/1/1c/Philippine_Institute_for_Development_Studies_%28PIDS%29.svg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original"


def render_app_title():
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 1.5rem; margin: 0.5rem 0 1rem; width: 100%;">
            <div style="display: flex; flex-direction: column; justify-content: center; min-width: 0; flex: 3; padding-right: 0.5 rem;">
                <div style="font-size: 4.2rem; line-height: 0.9; font-weight: 800; margin: 0; letter-spacing: -0.06em;">Macroeconomic</div>
                <div style="font-size: 4.2rem; line-height: 0.9; font-weight: 800; margin: 0; letter-spacing: -0.06em;">Forecasting Lab</div>
            </div>
            <div style="flex: 1; display: flex; justify-content: flex-end; align-items: center;">
                <img src="{LOGO_URL}" alt="App logo" style="width: 170px; height: 170px; object-fit: contain;">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_password():
    configured_password = st.secrets.get("APP_PASSWORD")
    if not configured_password:
        st.error("APP_PASSWORD is not configured in Streamlit Secrets.")
        st.stop()
    if st.session_state.get("authenticated"):
        return
    render_app_title()
    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if hmac.compare_digest(password, str(configured_password)):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


require_password()
render_app_title()
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
            default=[],
        )
        forecast_inputs = st.multiselect(
            "Forecast inputs",
            FORECAST_VARIABLES,
            default=[],
        )
        st.caption("Example: GDP_growth = Inflation_rate + Employment_rate. Use one dependent variable on the left and allowed math on the right.")
        custom_equations = [
            st.text_input(f"Custom equation {index}", value="", placeholder="e.g. GDP_growth = Inflation_rate + Employment_rate")
            for index in range(1, 4)
        ]
        selected_methods = st.multiselect(
            "Methods to compare",
            ["ARIMA", "VAR", "VECM", "GARCH", "XGBoost"],
            default=[],
        )
        use_equations_in_var_vecm = True
        forecast_horizon = st.number_input("Forecast horizon (years)", min_value=5, max_value=5, value=5, step=1)
        st.markdown(
            "<div style='display:flex; align-items:center; gap:0.5rem; margin-top:0.25rem;'>"
            "<span style='font-size:1.1rem; font-weight:700;'>Backtest share</span>"
            "<span title='Share of the dataset used for backtesting; the rest is used for training. Example: 0.20 means 20% held out for validation.' style='display:inline-flex; align-items:center; justify-content:center; width:1.2rem; height:1.2rem; border-radius:50%; background:#2d7df6; color:white; font-size:0.8rem; font-weight:700; cursor:help;'>?</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        test_fraction = st.slider("", 0.1, 0.4, 0.2, 0.05, label_visibility="collapsed")
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
preview = preview_data.loc[preview_data.index.year.isin([2024, 2025, 2026, 2027]), input_columns].copy()
preview.index = preview.index.year
preview.index.name = "Year"
if preview.empty:
    st.error("The CSV must contain rows for 2024, 2025, 2026, and 2027.")
    st.stop()

st.header("Editable input and output data preview")
st.caption("Edit any of the 16 non-date variables for 2024-2027. Year is shown only as the row index.")
edited_preview = st.data_editor(preview, num_rows="fixed", use_container_width=True, key="input_preview")
for year in edited_preview.index:
    row_mask = data.index.year == int(year)
    for column in input_columns:
        value = pd.to_numeric(edited_preview.loc[year, column], errors="coerce")
        if pd.notna(value):
            data.loc[row_mask, column] = value

if not run_comparison and "backtest_results" not in st.session_state:
    st.info("Choose your forecast settings, then click Run comparison in the sidebar.")
    st.stop()

if not forecast_outputs or not forecast_inputs or not selected_methods:
    st.warning("Select at least one output, input, and method in Forecast settings.")
    st.stop()

equations = equation_features(data, resolved)
for index, expression in enumerate(custom_equations, start=1):
    if not expression or not expression.strip():
        continue
    try:
        lhs, rhs_result = evaluate_user_equation(expression, data)
        equation_name = f"eq_custom_{index}_{lhs}"
        equations[equation_name] = rhs_result
    except ValueError as exc:
        st.warning(f"Custom equation {index} is invalid: {exc}")

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

if run_comparison or "backtest_results" not in st.session_state:
    predictions, errors = {}, {}
    with st.spinner("Fitting models and running the backtest..."):
        for method in selected_methods:
            try:
                predictions[method] = run_method(method, model_frame, train, targets, exogenous, equation_columns, len(test), use_equations_in_var_vecm=use_equations_in_var_vecm, resolved_columns=resolved)
                errors[method] = pd.DataFrame([metric_values(test[target], predictions[method][:, position], train[target]) for position, target in enumerate(targets)], index=target_labels)
            except Exception as error:
                st.warning(f"{method} could not be fitted: {error}")
    if not errors:
        st.stop()
    st.session_state["backtest_results"] = {
        "predictions": predictions,
        "errors": errors,
        "target_labels": target_labels,
    }
else:
    predictions = st.session_state["backtest_results"]["predictions"]
    errors = st.session_state["backtest_results"]["errors"]
    target_labels = st.session_state["backtest_results"]["target_labels"]

if not errors:
    st.stop()

st.header("Backtest results")
metric_name = st.selectbox("Metric to rank", ["RMSE", "MAE", "MAPE", "MASE"], index=0)
ranking = pd.DataFrame({method: result[metric_name] for method, result in errors.items()}, index=target_labels)
st.dataframe(ranking.style.format("{:.4f}"), use_container_width=True)
best = pd.DataFrame(index=target_labels)
for metric in ["RMSE", "MAE", "MAPE", "MASE"]:
    scores = pd.DataFrame({method: result[metric] for method, result in errors.items()}, index=target_labels)
    best[f"Best model by {metric}"] = scores.idxmin(axis=1)
best.insert(0, "Target", best.index)
st.subheader("Best forecasting method")
st.dataframe(best.reset_index(drop=True), hide_index=True, use_container_width=True)

forecast_years = list(range(2026, 2026 + int(forecast_horizon)))
forecast_index = pd.to_datetime([f"{year}-12-31" for year in forecast_years])
future_base = pd.DataFrame([data.loc[data.index.year == 2027].iloc[-1].to_dict()] * len(forecast_years), index=forecast_index).reindex(columns=data.columns)
future_frame = pd.concat([future_base, equation_features(future_base, resolved)], axis=1)
forecast_frame = pd.concat([model_frame, future_frame])
future_predictions = {}
for method in predictions:
    try:
        future_predictions[method] = run_method(method, forecast_frame, model_frame, targets, exogenous, equation_columns, len(forecast_years), use_equations_in_var_vecm=use_equations_in_var_vecm, resolved_columns=resolved)
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
