"""
backend/services/forecaster.py
------------------------------
Demand Forecasting & Evaluation Engine.

Provides:
- Chronological train/test splitting (zero leakage).
- Baseline Models:
    1. Naive Model (Last observed value carried forward)
    2. Moving Average Model (7-day rolling average)
- ML Model:
    3. Facebook Prophet (trend + weekly seasonality, 80% confidence interval)
- Evaluation Metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error, handles zeros safely)
    - WAPE (Weighted Absolute Percentage Error, robust to sparse demand)
- Multi-horizon forecasting (7, 14, 30 days) with confidence intervals.
- Persistence helpers for DB (ForecastRun, ForecastPoint, ForecastEvaluation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db.models import (
    DailyProductDemand,
    ForecastEvaluation,
    ForecastPoint,
    ForecastRun,
    UploadJob,
)

logger = logging.getLogger(__name__)

# Normal distribution critical value for 80% two-tailed CI (alpha = 0.20 => z_0.90 ~ 1.28155)
Z_80_PERCENT = 1.28155


@dataclass
class ForecastPointData:
    forecast_date: date
    yhat: float
    yhat_lower: float
    yhat_upper: float
    actual: Optional[float] = None
    is_future: bool = False


@dataclass
class EvaluationMetricsData:
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    wape: Optional[float] = None
    n_eval_points: int = 0


@dataclass
class ForecastResult:
    model_name: str
    product_id: str
    city_name: str
    horizon_days: int
    train_start: Optional[date]
    train_end: Optional[date]
    val_start: Optional[date]
    val_end: Optional[date]
    points: List[ForecastPointData]
    metrics: EvaluationMetricsData


# ---------------------------------------------------------------------------
# Metric Calculation Utilities
# ---------------------------------------------------------------------------

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationMetricsData:
    """
    Calculate MAE, RMSE, MAPE, and WAPE between actual and predicted arrays.
    Handles zeros safely: MAPE is None if any true value is 0.
    """
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return EvaluationMetricsData(n_eval_points=0)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE: Mean Absolute Percentage Error (undefined if any actual is 0)
    if np.any(y_true == 0):
        mape = None
    else:
        mape = float(np.mean(abs_errors / y_true) * 100.0)

    # WAPE: Weighted Absolute Percentage Error = sum(|y - yhat|) / sum(y) * 100
    sum_true = float(np.sum(y_true))
    if sum_true > 0:
        wape = float((np.sum(abs_errors) / sum_true) * 100.0)
    else:
        wape = None

    return EvaluationMetricsData(
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        mape=round(mape, 4) if mape is not None else None,
        wape=round(wape, 4) if wape is not None else None,
        n_eval_points=len(y_true),
    )


# ---------------------------------------------------------------------------
# Data Regularization & Splitting
# ---------------------------------------------------------------------------

def regularize_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the time series is daily continuous from min date to max date.
    Missing calendar days are filled with 0 quantity.
    """
    if df.empty:
        return df

    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.date
    df = df.groupby("sale_date", as_index=False)["total_quantity"].sum()
    df = df.sort_values("sale_date").reset_index(drop=True)

    min_date = df["sale_date"].min()
    max_date = df["sale_date"].max()

    all_dates = pd.date_range(min_date, max_date, freq="D").date
    full_df = pd.DataFrame({"sale_date": all_dates})
    merged = full_df.merge(df, on="sale_date", how="left")
    merged["total_quantity"] = merged["total_quantity"].fillna(0.0).astype(float)
    return merged


def split_train_test(
    df: pd.DataFrame, horizon_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split.
    Last horizon_days are held out for test evaluation.
    Earlier rows are used for training.
    """
    n = len(df)
    if n <= horizon_days:
        raise ValueError(
            f"Not enough historical data points ({n} days) for horizon of {horizon_days} days. "
            f"Need at least {horizon_days + 3} days."
        )

    split_idx = n - horizon_days
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Forecasting Models
# ---------------------------------------------------------------------------

def forecast_naive(
    train_df: pd.DataFrame, test_df: pd.DataFrame, horizon_days: int
) -> Tuple[List[ForecastPointData], EvaluationMetricsData]:
    """
    Naive baseline: Carry the last observed value forward.
    Uncertainty estimated from the standard deviation of historical 1-day differences.
    """
    train_y = train_df["total_quantity"].to_numpy(dtype=float)
    last_val = float(train_y[-1]) if len(train_y) > 0 else 0.0

    # Residual std for CI
    diffs = np.diff(train_y) if len(train_y) > 1 else np.array([0.0])
    sigma = float(np.std(diffs)) if len(diffs) > 0 and not np.isnan(np.std(diffs)) else 1.0
    half_ci = Z_80_PERCENT * sigma

    points: List[ForecastPointData] = []

    # 1. Historical evaluation points (test window)
    test_yhat = np.full(len(test_df), last_val)
    test_actuals = test_df["total_quantity"].to_numpy(dtype=float)

    for i, row in test_df.iterrows():
        yhat = max(0.0, last_val)
        lower = max(0.0, yhat - half_ci)
        upper = yhat + half_ci
        points.append(
            ForecastPointData(
                forecast_date=row["sale_date"],
                yhat=round(yhat, 4),
                yhat_lower=round(lower, 4),
                yhat_upper=round(upper, 4),
                actual=round(float(row["total_quantity"]), 4),
                is_future=False,
            )
        )

    metrics = calculate_metrics(test_actuals, test_yhat)

    # 2. Future forecast points beyond last test date
    # Re-anchor with the latest known observation (from test_df)
    future_anchor = float(test_actuals[-1]) if len(test_actuals) > 0 else last_val
    last_date = test_df["sale_date"].max()

    for d in range(1, horizon_days + 1):
        future_date = last_date + timedelta(days=d)
        yhat = max(0.0, future_anchor)
        lower = max(0.0, yhat - half_ci)
        upper = yhat + half_ci
        points.append(
            ForecastPointData(
                forecast_date=future_date,
                yhat=round(yhat, 4),
                yhat_lower=round(lower, 4),
                yhat_upper=round(upper, 4),
                actual=None,
                is_future=True,
            )
        )

    return points, metrics


def forecast_moving_average(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon_days: int,
    window: int = 7,
) -> Tuple[List[ForecastPointData], EvaluationMetricsData]:
    """
    Moving Average baseline: Rolling mean of recent observations (default 7 days).
    """
    train_y = train_df["total_quantity"].to_numpy(dtype=float)
    k = min(window, len(train_y))
    ma_val = float(np.mean(train_y[-k:])) if k > 0 else 0.0

    # Residual std based on rolling differences
    rolling_means = pd.Series(train_y).rolling(window=k, min_periods=1).mean().to_numpy()
    residuals = train_y - rolling_means
    sigma = float(np.std(residuals)) if len(residuals) > 0 and not np.isnan(np.std(residuals)) else 1.0
    half_ci = Z_80_PERCENT * sigma

    points: List[ForecastPointData] = []

    # 1. Historical evaluation points (test window)
    test_yhat = np.full(len(test_df), ma_val)
    test_actuals = test_df["total_quantity"].to_numpy(dtype=float)

    for i, row in test_df.iterrows():
        yhat = max(0.0, ma_val)
        lower = max(0.0, yhat - half_ci)
        upper = yhat + half_ci
        points.append(
            ForecastPointData(
                forecast_date=row["sale_date"],
                yhat=round(yhat, 4),
                yhat_lower=round(lower, 4),
                yhat_upper=round(upper, 4),
                actual=round(float(row["total_quantity"]), 4),
                is_future=False,
            )
        )

    metrics = calculate_metrics(test_actuals, test_yhat)

    # 2. Future forecast points
    # Recompute moving average including test actuals for future projection
    full_history = np.concatenate([train_y, test_actuals])
    future_ma = float(np.mean(full_history[-min(window, len(full_history)):]))
    last_date = test_df["sale_date"].max()

    for d in range(1, horizon_days + 1):
        future_date = last_date + timedelta(days=d)
        yhat = max(0.0, future_ma)
        lower = max(0.0, yhat - half_ci)
        upper = yhat + half_ci
        points.append(
            ForecastPointData(
                forecast_date=future_date,
                yhat=round(yhat, 4),
                yhat_lower=round(lower, 4),
                yhat_upper=round(upper, 4),
                actual=None,
                is_future=True,
            )
        )

    return points, metrics


def forecast_prophet(
    train_df: pd.DataFrame, test_df: pd.DataFrame, horizon_days: int
) -> Tuple[List[ForecastPointData], EvaluationMetricsData]:
    """
    Facebook Prophet model: trend + weekly seasonality with 80% confidence intervals.
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not installed. Falling back to moving average.")
        return forecast_moving_average(train_df, test_df, horizon_days)

    # Prepare training dataframe for Prophet: ds, y
    prophet_train = pd.DataFrame({
        "ds": pd.to_datetime(train_df["sale_date"]),
        "y": train_df["total_quantity"].astype(float),
    })

    # Fit model on training split
    m_eval = Prophet(
        interval_width=0.80,
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
    )
    m_eval.fit(prophet_train)

    # Predict over test window
    test_future_df = pd.DataFrame({"ds": pd.to_datetime(test_df["sale_date"])})
    pred_test = m_eval.predict(test_future_df)

    test_actuals = test_df["total_quantity"].to_numpy(dtype=float)
    test_yhat = np.clip(pred_test["yhat"].to_numpy(), a_min=0.0, a_max=None)
    metrics = calculate_metrics(test_actuals, test_yhat)

    points: List[ForecastPointData] = []
    for i, row in test_df.iterrows():
        pred_row = pred_test.iloc[i]
        yhat = max(0.0, float(pred_row["yhat"]))
        lower = max(0.0, float(pred_row["yhat_lower"]))
        upper = max(0.0, float(pred_row["yhat_upper"]))
        points.append(
            ForecastPointData(
                forecast_date=row["sale_date"],
                yhat=round(yhat, 4),
                yhat_lower=round(lower, 4),
                yhat_upper=round(upper, 4),
                actual=round(float(row["total_quantity"]), 4),
                is_future=False,
            )
        )

    # Now fit on full history (train + test) to produce the forward-looking future forecast
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    prophet_full = pd.DataFrame({
        "ds": pd.to_datetime(full_df["sale_date"]),
        "y": full_df["total_quantity"].astype(float),
    })

    m_full = Prophet(
        interval_width=0.80,
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
    )
    m_full.fit(prophet_full)

    future_dates_df = m_full.make_future_dataframe(periods=horizon_days, freq="D", include_history=False)
    pred_future = m_full.predict(future_dates_df)

    for _, row in pred_future.iterrows():
        f_date = row["ds"].date() if hasattr(row["ds"], "date") else row["ds"]
        yhat = max(0.0, float(row["yhat"]))
        lower = max(0.0, float(row["yhat_lower"]))
        upper = max(0.0, float(row["yhat_upper"]))
        points.append(
            ForecastPointData(
                forecast_date=f_date,
                yhat=round(yhat, 4),
                yhat_lower=round(lower, 4),
                yhat_upper=round(upper, 4),
                actual=None,
                is_future=True,
            )
        )

    return points, metrics


# ---------------------------------------------------------------------------
# Forecast Dispatcher
# ---------------------------------------------------------------------------

MODEL_DISPATCHER = {
    "naive": forecast_naive,
    "moving_avg": forecast_moving_average,
    "prophet": forecast_prophet,
}


def run_forecast(
    df: pd.DataFrame,
    product_id: str,
    city_name: str,
    horizon_days: int,
    model_name: str,
) -> ForecastResult:
    """
    Execute end-to-end forecasting and evaluation for one series and model.
    """
    if model_name not in MODEL_DISPATCHER:
        raise ValueError(f"Unsupported model '{model_name}'. Allowed: {list(MODEL_DISPATCHER.keys())}")

    reg_df = regularize_time_series(df)
    train_df, test_df = split_train_test(reg_df, horizon_days=horizon_days)

    train_start = train_df["sale_date"].min()
    train_end = train_df["sale_date"].max()
    val_start = test_df["sale_date"].min()
    val_end = test_df["sale_date"].max()

    model_fn = MODEL_DISPATCHER[model_name]
    points, metrics = model_fn(train_df, test_df, horizon_days)

    return ForecastResult(
        model_name=model_name,
        product_id=product_id,
        city_name=city_name,
        horizon_days=horizon_days,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        points=points,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------

def get_top_product_ids(
    db: Session, limit: int = 5, upload_job_id: Optional[int] = None
) -> List[str]:
    """Retrieve top N products by total aggregated quantity."""
    q = (
        db.query(
            DailyProductDemand.product_id,
            func.sum(DailyProductDemand.total_quantity).label("total_qty"),
        )
        .filter(DailyProductDemand.is_zero_filled == False)  # noqa: E712
    )
    if upload_job_id is not None:
        q = q.filter(DailyProductDemand.upload_job_id == upload_job_id)

    results = (
        q.group_by(DailyProductDemand.product_id)
        .order_by(func.sum(DailyProductDemand.total_quantity).desc())
        .limit(limit)
        .all()
    )
    return [str(r.product_id) for r in results]


def fetch_product_demand_df(
    db: Session,
    product_id: str,
    city_name: Optional[str] = None,
    upload_job_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch historical daily product demand from DB.
    If city_name is None, aggregates demand across all cities.
    """
    q = db.query(
        DailyProductDemand.sale_date,
        func.sum(DailyProductDemand.total_quantity).label("total_quantity"),
    ).filter(
        DailyProductDemand.product_id == product_id,
        DailyProductDemand.is_zero_filled == False,  # noqa: E712
    )

    if upload_job_id is not None:
        q = q.filter(DailyProductDemand.upload_job_id == upload_job_id)

    if city_name and city_name.upper() != "ALL":
        q = q.filter(DailyProductDemand.city_name == city_name)

    rows = (
        q.group_by(DailyProductDemand.sale_date)
        .order_by(DailyProductDemand.sale_date.asc())
        .all()
    )

    if not rows:
        return pd.DataFrame(columns=["sale_date", "total_quantity"])

    return pd.DataFrame([{"sale_date": r.sale_date, "total_quantity": r.total_quantity} for r in rows])


def persist_forecast_run(
    db: Session,
    result: ForecastResult,
    upload_job_id: Optional[int] = None,
) -> ForecastRun:
    """
    Save ForecastRun, ForecastPoints, and ForecastEvaluation to the database.
    """
    run = ForecastRun(
        upload_job_id=upload_job_id,
        product_id=result.product_id,
        city_name=result.city_name,
        model_name=result.model_name,
        horizon_days=result.horizon_days,
        train_start=result.train_start,
        train_end=result.train_end,
        val_start=result.val_start,
        val_end=result.val_end,
        status="complete",
        error_message=None,
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.flush()  # obtain run.id

    # Persist forecast points
    db_points = [
        ForecastPoint(
            run_id=run.id,
            forecast_date=p.forecast_date,
            yhat=p.yhat,
            yhat_lower=p.yhat_lower,
            yhat_upper=p.yhat_upper,
            actual=p.actual,
            is_future=p.is_future,
        )
        for p in result.points
    ]
    db.bulk_save_objects(db_points)

    # Persist evaluation metrics
    db_eval = ForecastEvaluation(
        run_id=run.id,
        mae=result.metrics.mae,
        rmse=result.metrics.rmse,
        mape=result.metrics.mape,
        wape=result.metrics.wape,
        n_eval_points=result.metrics.n_eval_points,
        computed_at=datetime.now(timezone.utc),
    )
    db.add(db_eval)
    db.commit()
    db.refresh(run)
    return run


def persist_failed_run(
    db: Session,
    product_id: str,
    city_name: str,
    model_name: str,
    horizon_days: int,
    error_message: str,
    upload_job_id: Optional[int] = None,
) -> ForecastRun:
    """Record a failed forecast run for traceability."""
    db.rollback()
    run = ForecastRun(
        upload_job_id=upload_job_id,
        product_id=product_id,
        city_name=city_name,
        model_name=model_name,
        horizon_days=horizon_days,
        status="failed",
        error_message=error_message,
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run
