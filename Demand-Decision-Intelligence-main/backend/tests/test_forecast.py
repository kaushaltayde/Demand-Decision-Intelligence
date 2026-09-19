"""
backend/tests/test_forecast.py
------------------------------
Unit and integration tests for Stage 5 Demand Forecasting & Evaluation Engine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from datetime import date, timedelta
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.forecaster import (
    calculate_metrics,
    split_train_test,
    regularize_time_series,
    forecast_naive,
    forecast_moving_average,
    forecast_prophet,
    run_forecast,
)


def _generate_synthetic_demand(n_days=60, base_qty=50.0):
    start = date(2022, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    # Seasonal + random pattern
    rng = np.random.default_rng(42)
    qtys = [max(1.0, base_qty + 10 * np.sin(2 * np.pi * i / 7) + rng.normal(0, 5)) for i in range(n_days)]
    return pd.DataFrame({"sale_date": dates, "total_quantity": qtys})


def test_calculate_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0])

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics.n_eval_points == 4
    assert abs(metrics.mae - 2.25) < 1e-2
    assert metrics.rmse > 0
    assert metrics.mape is not None and metrics.mape > 0
    assert metrics.wape is not None and metrics.wape > 0
    print("[PASS] test_calculate_metrics passed")


def test_calculate_metrics_with_zeros():
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([2.0, 8.0, 22.0])

    metrics = calculate_metrics(y_true, y_pred)
    # MAPE should be None when true values contain 0
    assert metrics.mape is None
    # WAPE is robust to zeros
    assert metrics.wape is not None and metrics.wape > 0
    assert metrics.mae > 0


def test_train_test_split_chronological():
    df = _generate_synthetic_demand(n_days=40)
    train_df, test_df = split_train_test(df, horizon_days=7)

    assert len(train_df) == 33
    assert len(test_df) == 7
    # Zero leakage: all train dates strictly before test dates
    assert train_df["sale_date"].max() < test_df["sale_date"].min()


def test_forecast_naive():
    df = _generate_synthetic_demand(n_days=30)
    train_df, test_df = split_train_test(df, horizon_days=7)
    points, metrics = forecast_naive(train_df, test_df, horizon_days=7)

    # 7 test points + 7 future points = 14 points
    assert len(points) == 14
    assert metrics.n_eval_points == 7
    assert metrics.mae is not None and metrics.mae >= 0

    for p in points:
        assert p.yhat >= 0
        assert p.yhat_lower is not None and p.yhat_upper is not None
        assert p.yhat_lower <= p.yhat <= p.yhat_upper or np.isclose(p.yhat_lower, p.yhat)
    print("[PASS] test_forecast_naive passed")


def test_forecast_moving_average():
    df = _generate_synthetic_demand(n_days=30)
    train_df, test_df = split_train_test(df, horizon_days=7)
    points, metrics = forecast_moving_average(train_df, test_df, horizon_days=7, window=7)

    assert len(points) == 14
    assert metrics.n_eval_points == 7
    assert metrics.mae is not None and metrics.mae >= 0

    for p in points:
        assert p.yhat >= 0
        assert p.yhat_lower <= p.yhat_upper
    print("[PASS] test_moving_average passed")


def test_forecast_prophet():
    df = _generate_synthetic_demand(n_days=45)
    train_df, test_df = split_train_test(df, horizon_days=7)
    points, metrics = forecast_prophet(train_df, test_df, horizon_days=7)

    assert len(points) == 14
    assert metrics.n_eval_points == 7
    assert metrics.mae is not None and metrics.mae >= 0
    assert metrics.rmse is not None and metrics.rmse >= 0
    print("[PASS] test_forecast_prophet passed")


def test_forecast_api_run_and_retrieve():
    client = TestClient(app)

    # Trigger forecast for top products
    payload = {
        "models": ["naive", "moving_avg"],
        "horizon_days": 7,
    }
    response = client.post("/api/forecast/run", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["total_runs"] > 0
    assert data["successful_runs"] > 0
    run_id = data["runs"][0]["id"]

    # Fetch run details
    detail_res = client.get(f"/api/forecast/{run_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["id"] == run_id
    assert len(detail["points"]) > 0
    assert detail["evaluation"] is not None

    # List runs
    list_res = client.get("/api/forecast/runs?page=1&page_size=10")
    assert list_res.status_code == 200
    runs_list = list_res.json()
    assert runs_list["total"] >= 1
    print("[PASS] test_forecast_api_run_and_retrieve passed")


if __name__ == "__main__":
    print("Running forecast tests...")
    test_calculate_metrics()
    test_train_test_split_chronological()
    test_forecast_naive()
    test_forecast_moving_average()
    test_forecast_prophet()
    test_forecast_api_run_and_retrieve()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
