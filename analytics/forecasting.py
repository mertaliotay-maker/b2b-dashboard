"""Leakage-free weekly model comparison, shared by Jupyter and Streamlit.

Only training observations enter lag features, fitting and model selection.
The last eight weeks are an untouched audit set, never a selection signal.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

MODELS = ("Naive", "Moving average (4)", "Holt damped", "ARIMA(1,1,1)", "Random Forest")
LAGS = (1, 2, 3, 4, 8, 13)
MIN_WEEKS = 40


def weekly_series(df: pd.DataFrame, target: str = "Quantity") -> tuple[pd.Series, dict]:
    """Monday-Sunday totals. Trim both boundary weeks unless fully covered.

    Missing interior weeks are zero observed sales, assuming continuous data
    coverage. We cannot distinguish stockouts from zero underlying demand.
    """
    if df.empty:
        raise ValueError("No transactions in the selected filters.")
    if target not in {"Quantity", "Revenue"}:
        raise ValueError("Target must be Quantity or Revenue.")
    start = pd.Timestamp(df.InvoiceDate.min()).normalize()
    end = pd.Timestamp(df.InvoiceDate.max()).normalize()
    first = start + pd.Timedelta(days=(7 - start.weekday()) % 7)
    last = end - pd.Timedelta(days=(end.weekday() + 1) % 7)
    if first > last:
        raise ValueError("No complete Monday-Sunday week in the selected period.")
    use = df.loc[df.InvoiceDate.between(first, last), ["InvoiceDate", target]]
    y = use.set_index("InvoiceDate")[target].resample("W-SUN").sum()
    y = y.reindex(pd.date_range(first + pd.Timedelta(days=6), last, freq="W-SUN"), fill_value=0).astype(float)
    y.index.name, y.name = "WeekEnd", target
    return y, {"transaction_start": start, "transaction_end": end,
               "complete_start": first, "complete_end": last,
               "excluded_lines": int(len(df) - len(use)), "n_weeks": len(y),
               "included_total": float(y.sum()), "excluded_total": float(df[target].sum() - y.sum())}


def metrics(actual, predicted) -> dict:
    """MAPE excludes zero actuals, reports denominator; WAPE uses all periods."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != p.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require matching non-empty finite arrays.")
    err = a - p
    nonzero = a != 0
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.abs(err).mean()),
            "MAPE_pct": float(np.mean(np.abs(err[nonzero] / a[nonzero])) * 100) if nonzero.any() else np.nan,
            "WAPE_pct": float(np.abs(err).sum() / np.abs(a).sum() * 100) if np.abs(a).sum() else np.nan,
            "MAPE_n": int(nonzero.sum()), "n": len(a)}


def _features(history, date, position):
    """Every statistic ends at t-1. No contemporaneous target or future actuals."""
    v = list(history)
    # Calendar features are intentionally omitted: <2 annual cycles cannot
    # establish a repeatable yearly pattern. Position supplies a trend feature.
    return [v[-lag] for lag in LAGS] + [np.mean(v[-4:]), np.std(v[-4:], ddof=1),
                                       np.mean(v[-13:]), float(position)]


def forecast_model(name: str, train: pd.Series, steps: int) -> np.ndarray:
    """Fixed model settings; recursive multi-step RF never consumes test values."""
    if steps < 1 or len(train) < 20 or not np.isfinite(train).all() or (train < 0).any():
        raise ValueError("Need >=20 finite, nonnegative training weeks and positive steps.")
    values = train.to_numpy(dtype=float)
    if name == "Naive":
        pred = np.repeat(values[-1], steps)
    elif name == "Moving average (4)":
        pred = np.repeat(values[-4:].mean(), steps)
    elif name == "Random Forest":
        X = [_features(values[:i], train.index[i], i) for i in range(max(LAGS), len(values))]
        model = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=2,
                                      max_features=1.0, random_state=42, n_jobs=1)
        model.fit(np.asarray(X), values[max(LAGS):])
        history, pred = list(values), []
        for h in range(steps):
            date = train.index[-1] + pd.Timedelta(weeks=h + 1)
            p = float(model.predict([_features(history, date, len(history))])[0])
            history.append(max(0., p))
            pred.append(max(0., p))
    elif name in {"Holt damped", "ARIMA(1,1,1)"}:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            scale = max(float(train.mean()), 1.)
            if name == "Holt damped":
                # Holt-Winters family, NO seasonal component: only one year of
                # data exists. Two complete yearly cycles would be required.
                model = ExponentialSmoothing(train / scale, trend="add", damped_trend=True,
                                             seasonal=None, initialization_method="estimated").fit()
                converged = bool(model.mle_retvals.get("success", True))
            else:
                # Normalize using TRAIN mean only for numerical conditioning.
                model = ARIMA(train / scale, order=(1, 1, 1), trend="t").fit()
                converged = bool(model.mle_retvals.get("converged", True))
            if not converged or any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError(f"{name}: optimizer did not converge; excluded, not silently replaced.")
            pred = np.asarray(model.forecast(steps), dtype=float)
            pred = pred * scale
    else:
        raise ValueError(f"Unknown model: {name}")
    pred = np.maximum(np.asarray(pred, dtype=float), 0.)
    if len(pred) != steps or not np.isfinite(pred).all():
        raise RuntimeError(f"{name} produced invalid forecasts.")
    return pred


def evaluate_models(y: pd.Series) -> dict:
    """Pretest expanding-window CV selects model; final 8 weeks audit it.

    CV uses up to three non-overlapping 8-week horizons, at least 20 training
    weeks per origin. For the 51-week dataset there are two CV origins.
    RMSE is the sole predeclared selection criterion; MAPE is supplementary.
    """
    if len(y) < MIN_WEEKS:
        raise ValueError(f"At least {MIN_WEEKS} complete weeks are required; selected: {len(y)}.")
    test_size, cv_h = 8, 8
    development, test = y.iloc[:-test_size], y.iloc[-test_size:]
    origins = sorted([len(development) - cv_h * i for i in range(1, 4)
                      if len(development) - cv_h * i >= 20])
    cv_rows, prediction_rows, issues, ok = [], [], [], []
    for name in MODELS:
        local = []
        try:
            for origin in origins:
                train, valid = development.iloc[:origin], development.iloc[origin:origin + cv_h]
                pred = forecast_model(name, train, len(valid))
                local.extend({"Model": name, "Origin": train.index[-1], "WeekEnd": date,
                              "Horizon": i + 1, "Actual": float(a), "Prediction": float(p),
                              "Error": float(a - p)}
                             for i, (date, a, p) in enumerate(zip(valid.index, valid, pred)))
            frame = pd.DataFrame(local)
            cv_rows.append({"Model": name, **metrics(frame.Actual, frame.Prediction)})
            prediction_rows.extend(local)
            ok.append(name)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            issues.append({"Model": name, "Stage": "CV", "Reason": str(exc)})
    cv = pd.DataFrame(cv_rows).sort_values(["RMSE", "Model"]).reset_index(drop=True)
    # A test failure cannot trigger reselection. It is a visible audit failure.
    winner = str(cv.iloc[0].Model)
    test_rows, test_predictions = [], pd.DataFrame({"Actual": test})
    for name in ok:
        try:
            pred = forecast_model(name, development, len(test))
            test_predictions[name] = pred
            test_rows.append({"Model": name, **metrics(test, pred)})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            issues.append({"Model": name, "Stage": "Test", "Reason": str(exc)})
    cv_predictions = pd.DataFrame(prediction_rows)
    calibration = cv_predictions[cv_predictions.Model == winner]
    scaled_errors = np.abs(calibration.Error.to_numpy()) / np.sqrt(calibration.Horizon.to_numpy())
    # Approximate 95% prediction band. All calibration errors precede the test.
    # Sparse dependent residuals and sqrt(h) growth do NOT guarantee 95% coverage.
    radius = float(np.quantile(scaled_errors, .95, method="higher"))
    audit_band = prediction_band(test_predictions[winner].to_numpy(), radius) if winner in test_predictions else None
    coverage = float(((test.to_numpy() >= audit_band[:, 0]) & (test.to_numpy() <= audit_band[:, 1])).mean()) if audit_band is not None else np.nan
    return {"winner": winner, "cv": cv, "test_scores": pd.DataFrame(test_rows),
            "test_predictions": test_predictions, "cv_predictions": cv_predictions,
            "issues": pd.DataFrame(issues, columns=["Model", "Stage", "Reason"]),
            "radius": radius, "test_coverage": coverage,
            "origins": [development.index[i-1] for i in origins],
            "development_end": development.index[-1], "test_start": test.index[0],
            "test_end": test.index[-1], "calibration_n": len(calibration)}


def prediction_band(point, radius: float) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    width = radius * np.sqrt(np.arange(1, len(point) + 1))
    return np.column_stack([np.maximum(0., point - width), point + width])


def future_forecast(y: pd.Series, evaluation: dict, steps: int = 13) -> pd.DataFrame:
    """Refit the CV winner on all complete historical weeks for future use."""
    if steps not in {13, 26, 52}:
        raise ValueError("Forecast horizon must be 13, 26, or 52 weeks.")
    point = forecast_model(evaluation["winner"], y, steps)
    band = prediction_band(point, evaluation["radius"])
    return pd.DataFrame({"Forecast": point, "Lower95": band[:, 0], "Upper95": band[:, 1]},
                        index=pd.date_range(y.index[-1] + pd.Timedelta(weeks=1), periods=steps, freq="W-SUN", name="WeekEnd"))
