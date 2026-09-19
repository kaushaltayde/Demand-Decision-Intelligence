"""
backend/api/forecast.py
-----------------------
FastAPI router for the Demand Forecasting & Evaluation Engine.

Endpoints:
    POST /api/forecast/run
        Trigger demand forecasting for specified products (or top 5 by default),
        models (naive, moving_avg, prophet), and horizon (7, 14, 30 days).
        Saves results and evaluation metrics in the database.

    GET  /api/forecast/{id}
        Retrieve full details for a forecast run, including all time series
        points (historical test predictions + future forecasts) and evaluation metrics.

    GET  /api/forecast/runs
        List historical forecast runs with optional filtering and pagination.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.db.models import ForecastEvaluation, ForecastPoint, ForecastRun
from backend.db.session import get_db
from backend.schemas.forecast import (
    EvaluationMetricsSchema,
    ForecastBatchRunResponse,
    ForecastPointSchema,
    ForecastRunDetail,
    ForecastRunListResponse,
    ForecastRunRequest,
    ForecastRunSummary,
)
from backend.services.forecaster import (
    fetch_product_demand_df,
    get_top_product_ids,
    persist_failed_run,
    persist_forecast_run,
    run_forecast,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _format_run_summary(run: ForecastRun) -> ForecastRunSummary:
    eval_schema = None
    if run.evaluation:
        eval_schema = EvaluationMetricsSchema(
            mae=float(run.evaluation.mae) if run.evaluation.mae is not None else None,
            rmse=float(run.evaluation.rmse) if run.evaluation.rmse is not None else None,
            mape=float(run.evaluation.mape) if run.evaluation.mape is not None else None,
            wape=float(run.evaluation.wape) if run.evaluation.wape is not None else None,
            n_eval_points=run.evaluation.n_eval_points,
        )

    return ForecastRunSummary(
        id=run.id,
        upload_job_id=run.upload_job_id,
        product_id=run.product_id,
        city_name=run.city_name,
        model_name=run.model_name,
        horizon_days=run.horizon_days,
        train_start=run.train_start,
        train_end=run.train_end,
        val_start=run.val_start,
        val_end=run.val_end,
        status=run.status,
        error_message=run.error_message,
        created_at=run.created_at,
        evaluation=eval_schema,
    )


# ---------------------------------------------------------------------------
# POST /api/forecast/run
# ---------------------------------------------------------------------------

@router.post(
    "/run",
    response_model=ForecastBatchRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger demand forecasting & evaluation runs",
    description=(
        "Generate demand forecasts for selected or top 5 products across specified models "
        "(naive, moving_avg, prophet) and horizon (7, 14, 30 days). "
        "Calculates evaluation metrics (MAE, RMSE, MAPE, WAPE) on chronological holdout splits "
        "and saves all forecasts and metrics in the database."
    ),
)
def run_forecasts(
    request: ForecastRunRequest,
    db: Session = Depends(get_db),
) -> ForecastBatchRunResponse:
    target_products = request.product_ids or []
    if not target_products:
        # Default to top 5 products by demand quantity
        target_products = get_top_product_ids(
            db, limit=5, upload_job_id=request.upload_job_id
        )

    if not target_products:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No products found to forecast. Please ensure daily demand data exists in the database "
                "or upload a sales file first."
            ),
        )

    city_name = request.city_name or "ALL"
    completed_summaries: List[ForecastRunSummary] = []
    success_count = 0
    fail_count = 0

    for product_id in target_products:
        demand_df = fetch_product_demand_df(
            db=db,
            product_id=product_id,
            city_name=city_name,
            upload_job_id=request.upload_job_id,
        )

        for model_name in request.models:
            if demand_df.empty:
                err_msg = f"No demand history found for product {product_id} (city={city_name})."
                failed_run = persist_failed_run(
                    db=db,
                    product_id=product_id,
                    city_name=city_name,
                    model_name=model_name,
                    horizon_days=request.horizon_days,
                    error_message=err_msg,
                    upload_job_id=request.upload_job_id,
                )
                completed_summaries.append(_format_run_summary(failed_run))
                fail_count += 1
                continue

            try:
                forecast_res = run_forecast(
                    df=demand_df,
                    product_id=product_id,
                    city_name=city_name,
                    horizon_days=request.horizon_days,
                    model_name=model_name,
                )
                db_run = persist_forecast_run(
                    db=db,
                    result=forecast_res,
                    upload_job_id=request.upload_job_id,
                )
                completed_summaries.append(_format_run_summary(db_run))
                success_count += 1
            except Exception as exc:
                db.rollback()
                logger.error(
                    "Error executing forecast for product %s, model %s: %s",
                    product_id,
                    model_name,
                    exc,
                    exc_info=True,
                )
                failed_run = persist_failed_run(
                    db=db,
                    product_id=product_id,
                    city_name=city_name,
                    model_name=model_name,
                    horizon_days=request.horizon_days,
                    error_message=str(exc),
                    upload_job_id=request.upload_job_id,
                )
                completed_summaries.append(_format_run_summary(failed_run))
                fail_count += 1

    return ForecastBatchRunResponse(
        total_runs=len(completed_summaries),
        successful_runs=success_count,
        failed_runs=fail_count,
        runs=completed_summaries,
    )


# ---------------------------------------------------------------------------
# GET /api/forecast/runs
# ---------------------------------------------------------------------------

@router.get(
    "/runs",
    response_model=ForecastRunListResponse,
    summary="List all forecast runs with filtering & pagination",
)
def list_forecast_runs(
    upload_job_id: Optional[int] = Query(None, description="Filter by upload job ID"),
    product_id: Optional[str] = Query(None, description="Filter by product ID"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    horizon_days: Optional[int] = Query(None, description="Filter by horizon"),
    status: Optional[str] = Query(None, description="Filter by status (complete, failed)"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> ForecastRunListResponse:
    q = db.query(ForecastRun)

    if upload_job_id is not None:
        q = q.filter(ForecastRun.upload_job_id == upload_job_id)
    if product_id:
        q = q.filter(ForecastRun.product_id == product_id)
    if model_name:
        q = q.filter(ForecastRun.model_name == model_name)
    if horizon_days is not None:
        q = q.filter(ForecastRun.horizon_days == horizon_days)
    if status:
        q = q.filter(ForecastRun.status == status)

    total = q.count()
    runs = (
        q.order_by(ForecastRun.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return ForecastRunListResponse(
        total=total,
        page=page,
        page_size=page_size,
        results=[_format_run_summary(r) for r in runs],
    )


# ---------------------------------------------------------------------------
# GET /api/forecast/{id}
# ---------------------------------------------------------------------------

@router.get(
    "/{id}",
    response_model=ForecastRunDetail,
    summary="Get forecast run details and prediction points",
    description="Returns run metadata, evaluation metrics, and full prediction series with confidence intervals.",
)
def get_forecast_run(
    id: int,
    db: Session = Depends(get_db),
) -> ForecastRunDetail:
    run = db.query(ForecastRun).filter(ForecastRun.id == id).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Forecast run with id {id} not found",
        )

    summary = _format_run_summary(run)

    points_q = (
        db.query(ForecastPoint)
        .filter(ForecastPoint.run_id == id)
        .order_by(ForecastPoint.forecast_date.asc())
        .all()
    )

    points = [
        ForecastPointSchema(
            forecast_date=p.forecast_date,
            yhat=float(p.yhat),
            yhat_lower=float(p.yhat_lower) if p.yhat_lower is not None else None,
            yhat_upper=float(p.yhat_upper) if p.yhat_upper is not None else None,
            actual=float(p.actual) if p.actual is not None else None,
            is_future=p.is_future,
        )
        for p in points_q
    ]

    return ForecastRunDetail(
        **summary.model_dump(),
        points=points,
    )
