"""
backend/schemas/forecast.py
----------------------------
Pydantic request / response schemas for the Demand Forecasting API.

Endpoints:
    POST /api/forecast/run     -> ForecastRunRequest -> ForecastBatchRunResponse
    GET  /api/forecast/{id}    -> ForecastRunDetail
    GET  /api/forecast/runs    -> ForecastRunListResponse
"""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


SUPPORTED_MODELS = ("naive", "moving_avg", "prophet")
SUPPORTED_HORIZONS = (7, 14, 30)


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class ForecastRunRequest(BaseModel):
    """
    Trigger demand forecasting.
    If product_ids is omitted or empty, the system defaults to the top 5 products.
    """
    upload_job_id: Optional[int] = Field(
        None,
        description="Optional upload job ID. If omitted, uses latest available daily demand.",
    )
    product_ids: Optional[List[str]] = Field(
        None,
        description="Product IDs to forecast. Defaults to top 5 products by quantity if omitted/empty.",
    )
    city_name: Optional[str] = Field(
        None,
        description="City scope for demand aggregation. If omitted, aggregates across all cities.",
    )
    models: List[str] = Field(
        default=["naive", "moving_avg", "prophet"],
        description="List of models to run: 'naive', 'moving_avg', 'prophet'",
    )
    horizon_days: int = Field(
        default=7,
        description="Forecast horizon: 7, 14, or 30 days.",
    )

    @field_validator("horizon_days")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if v not in SUPPORTED_HORIZONS:
            raise ValueError(f"horizon_days must be one of {SUPPORTED_HORIZONS}, got {v}")
        return v

    @field_validator("models")
    @classmethod
    def validate_models(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("models list cannot be empty")
        valid = set(SUPPORTED_MODELS)
        for m in v:
            if m not in valid:
                raise ValueError(f"Invalid model '{m}'. Supported models: {SUPPORTED_MODELS}")
        return list(dict.fromkeys(v))  # preserve order, remove duplicates


# ---------------------------------------------------------------------------
# Point & Evaluation Schemas
# ---------------------------------------------------------------------------

class ForecastPointSchema(BaseModel):
    """Single point in a forecast time series (historical test evaluation or future prediction)."""
    forecast_date: date
    yhat: float = Field(description="Point prediction (rounded/non-negative)")
    yhat_lower: Optional[float] = Field(None, description="80% confidence interval lower bound")
    yhat_upper: Optional[float] = Field(None, description="80% confidence interval upper bound")
    actual: Optional[float] = Field(None, description="Actual demand if within historical evaluation window")
    is_future: bool = Field(False, description="True if beyond historical data range")

    class Config:
        from_attributes = True


class EvaluationMetricsSchema(BaseModel):
    """Summary error metrics computed on chronological test split."""
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error (None if actuals contain 0)")
    wape: Optional[float] = Field(None, description="Weighted Absolute Percentage Error (robust to 0)")
    n_eval_points: int = Field(0, description="Count of evaluated points")

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Run Detail & Summary Schemas
# ---------------------------------------------------------------------------

class ForecastRunSummary(BaseModel):
    """Metadata summary of a forecast run with evaluation metrics."""
    id: int
    upload_job_id: Optional[int] = None
    product_id: str
    city_name: str
    model_name: str
    horizon_days: int
    train_start: Optional[date] = None
    train_end: Optional[date] = None
    val_start: Optional[date] = None
    val_end: Optional[date] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    evaluation: Optional[EvaluationMetricsSchema] = None

    class Config:
        from_attributes = True


class ForecastRunDetail(ForecastRunSummary):
    """Full forecast run including all prediction points."""
    points: List[ForecastPointSchema] = Field(default_factory=list)

    class Config:
        from_attributes = True


class ForecastBatchRunResponse(BaseModel):
    """Response returned when triggering forecast runs."""
    total_runs: int
    successful_runs: int
    failed_runs: int
    runs: List[ForecastRunSummary]


class ForecastRunListResponse(BaseModel):
    """Paginated list of forecast runs."""
    total: int
    page: int
    page_size: int
    results: List[ForecastRunSummary]
