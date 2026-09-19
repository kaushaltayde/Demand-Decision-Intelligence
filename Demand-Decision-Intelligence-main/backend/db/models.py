"""
backend/db/models.py
--------------------
SQLAlchemy ORM models for the Demand & Decision Intelligence System.

Tables created here:
  upload_jobs          – one row per file upload attempt
  sales_records        – validated/cleaned individual transaction rows
  daily_product_demand – aggregated daily demand (the ML team's input)
  forecast_runs        – one row per triggered forecast job
  forecast_points      – per-day predictions with confidence intervals
  forecast_evaluations – aggregated error metrics per run

Convention: snake_case column names (per CONVENTIONS.md).
"""

from datetime import datetime, date, timezone
from sqlalchemy import (
    Column, Integer, BigInteger, String, Numeric, Date, DateTime,
    Boolean, Text, ForeignKey, UniqueConstraint, Index,
)
from sqlalchemy.orm import relationship
from backend.db.session import Base


# ---------------------------------------------------------------------------
# UploadJob
# ---------------------------------------------------------------------------

class UploadJob(Base):
    """
    Tracks every file upload attempt.

    One row is created when a user submits a file to POST /api/demand/aggregate.
    Status progresses: pending → validating → aggregating → complete | failed.
    """

    __tablename__ = "upload_jobs"

    id = Column(Integer, primary_key=True, index=True)

    # File metadata
    original_filename = Column(String(255), nullable=False)
    file_size_bytes   = Column(BigInteger, nullable=True)

    # Row-level accounting
    total_rows_read    = Column(Integer, default=0, nullable=False)
    valid_rows         = Column(Integer, default=0, nullable=False)
    rejected_rows      = Column(Integer, default=0, nullable=False)
    duplicate_rows     = Column(Integer, default=0, nullable=False)

    # Missing-value fills (non-critical columns that were safely filled)
    rows_city_filled     = Column(Integer, default=0, nullable=False)
    rows_discount_filled = Column(Integer, default=0, nullable=False)
    rows_order_id_filled = Column(Integer, default=0, nullable=False)

    # Date coverage of the uploaded file
    upload_date_min = Column(Date, nullable=True)
    upload_date_max = Column(Date, nullable=True)

    # Job status
    status = Column(
        String(32),
        default="pending",
        nullable=False,
        comment="pending | validating | aggregating | complete | failed",
    )
    error_message = Column(Text, nullable=True)

    # Aggregation summary
    aggregated_rows = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    finished_at = Column(DateTime, nullable=True)

    # Relationships
    sales_records        = relationship("SalesRecord",        back_populates="upload_job", lazy="dynamic")
    daily_demand_records = relationship("DailyProductDemand", back_populates="upload_job", lazy="dynamic")

    def __repr__(self) -> str:
        return (
            f"<UploadJob id={self.id} file={self.original_filename!r} "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# SalesRecord
# ---------------------------------------------------------------------------

class SalesRecord(Base):
    """
    Stores individual validated/cleaned transaction rows after upload.

    These are the 'cleaned' rows that passed all mandatory checks.
    The demand aggregator reads from this table (or directly from the
    in-memory DataFrame if called in the same request pipeline).

    NOTE: This table is intentionally kept lean — it mirrors the raw
    column set exactly. The daily_product_demand table is the aggregated
    output for downstream ML use.
    """

    __tablename__ = "sales_records"

    id = Column(BigInteger, primary_key=True, index=True)

    upload_job_id = Column(
        Integer,
        ForeignKey("upload_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Core transaction fields (required — rows without these are rejected)
    sale_date           = Column(Date,          nullable=False)
    product_id          = Column(String(64),    nullable=False)
    procured_quantity   = Column(Numeric(12, 4), nullable=False)
    unit_selling_price  = Column(Numeric(12, 4), nullable=False)

    # Derived / optional fields (safely filled when missing)
    city_name              = Column(String(128), nullable=False, default="Unknown")
    total_discount_amount  = Column(Numeric(12, 4), nullable=False, default=0.0)
    order_id               = Column(String(64),  nullable=True)

    # Extra columns from the Flipkart dataset schema (stored if present)
    cart_id                      = Column(String(64),  nullable=True)
    dim_customer_key             = Column(String(64),  nullable=True)
    total_weighted_landing_price = Column(Numeric(14, 4), nullable=True)

    # Relationship
    upload_job = relationship("UploadJob", back_populates="sales_records")

    __table_args__ = (
        Index("ix_sales_records_sale_date_product", "sale_date", "product_id"),
        # NOTE: upload_job_id index is covered by index=True on the column above.
    )

    def __repr__(self) -> str:
        return (
            f"<SalesRecord id={self.id} date={self.sale_date} "
            f"product={self.product_id} qty={self.procured_quantity}>"
        )


# ---------------------------------------------------------------------------
# DailyProductDemand
# ---------------------------------------------------------------------------

class DailyProductDemand(Base):
    """
    Aggregated daily demand output — the primary table for ML forecasting.

    One row per (sale_date, product_id, city_name) combination within
    a given upload job.

    Column alignment with the ML team's existing CSV:
      date_          → sale_date
      product_id     → product_id
      city_name      → city_name
      daily_quantity → total_quantity
      daily_revenue  → revenue
      order_count    → order_count

    Additional columns added for ML feature use:
      avg_unit_price – revenue-weighted average unit price
      avg_discount   – average discount per transaction in the group
      is_zero_filled – True if this row was synthesised for a date gap
    """

    __tablename__ = "upload_daily_demand"

    id = Column(Integer, primary_key=True, index=True)

    upload_job_id = Column(
        Integer,
        ForeignKey("upload_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Aggregation key (date + product + city)
    sale_date   = Column(Date,       nullable=False)
    product_id  = Column(String(64), nullable=False)
    city_name   = Column(String(128), nullable=False)

    # Aggregated demand metrics
    total_quantity = Column(Integer,      nullable=False, default=0)
    revenue        = Column(Numeric(16, 2), nullable=False, default=0)
    avg_unit_price = Column(Numeric(12, 4), nullable=True)   # revenue-weighted avg
    avg_discount   = Column(Numeric(12, 4), nullable=True)   # simple average
    order_count    = Column(Integer,      nullable=False, default=0)

    # Zero-fill flag: True when this row fills a date gap (no actual sales)
    is_zero_filled = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship
    upload_job = relationship("UploadJob", back_populates="daily_demand_records")

    __table_args__ = (
        # Each (date, product, city) is unique within a job
        UniqueConstraint(
            "sale_date", "product_id", "city_name", "upload_job_id",
            name="uq_upload_daily_demand_key",
        ),
        Index("ix_upload_daily_demand_date_product", "sale_date", "product_id"),
        Index("ix_upload_daily_demand_upload_job",   "upload_job_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<DailyProductDemand date={self.sale_date} "
            f"product={self.product_id} city={self.city_name!r} "
            f"qty={self.total_quantity}>"
        )


# ---------------------------------------------------------------------------
# ForecastRun  (Stage 5)
# ---------------------------------------------------------------------------

class ForecastRun(Base):
    """
    One row per triggered forecast job.

    A single POST /api/forecast/run may create many ForecastRun rows —
    one for every (product_id × city_name × model_name × horizon_days)
    combination requested.

    Status lifecycle: pending → complete | failed
    """

    __tablename__ = "forecast_runs"

    id = Column(Integer, primary_key=True, index=True)

    # Source data reference
    upload_job_id = Column(
        Integer,
        ForeignKey("upload_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Forecast scope
    product_id  = Column(String(64),  nullable=False, index=True)
    city_name   = Column(String(128), nullable=False)

    # Model configuration
    model_name   = Column(
        String(32),
        nullable=False,
        comment="naive | moving_avg | prophet",
    )
    horizon_days = Column(Integer, nullable=False, comment="7, 14, or 30")

    # Chronological splits (dates, not datetimes)
    train_start = Column(Date, nullable=True)
    train_end   = Column(Date, nullable=True)
    val_start   = Column(Date, nullable=True)
    val_end     = Column(Date, nullable=True)

    # Job status
    status        = Column(String(16), default="pending", nullable=False,
                           comment="pending | complete | failed")
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    points     = relationship("ForecastPoint",      back_populates="run",
                              cascade="all, delete-orphan", lazy="dynamic")
    evaluation = relationship("ForecastEvaluation", back_populates="run",
                              uselist=False, cascade="all, delete-orphan")
    upload_job = relationship("UploadJob", foreign_keys=[upload_job_id])

    __table_args__ = (
        Index("ix_forecast_runs_product_model", "product_id", "model_name"),
    )

    def __repr__(self) -> str:
        return (
            f"<ForecastRun id={self.id} product={self.product_id!r} "
            f"model={self.model_name!r} horizon={self.horizon_days}d "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# ForecastPoint  (Stage 5)
# ---------------------------------------------------------------------------

class ForecastPoint(Base):
    """
    One row per (forecast_run, forecast_date).

    Stores the point prediction and 80 % confidence interval.
    For dates within the historical test window, `actual` is populated
    so evaluation metrics can be computed or re-computed.
    `is_future` is True for dates beyond the last observed data date.
    """

    __tablename__ = "forecast_points"

    id = Column(BigInteger, primary_key=True, index=True)

    run_id = Column(
        Integer,
        ForeignKey("forecast_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    forecast_date = Column(Date,           nullable=False)
    yhat          = Column(Numeric(14, 4), nullable=False, comment="point prediction")
    yhat_lower    = Column(Numeric(14, 4), nullable=True,  comment="80% CI lower")
    yhat_upper    = Column(Numeric(14, 4), nullable=True,  comment="80% CI upper")
    actual        = Column(Numeric(14, 4), nullable=True,
                           comment="actual demand (test period only; NULL for future)")
    is_future     = Column(Boolean, default=False, nullable=False,
                           comment="True when date is beyond available data")

    # Relationship
    run = relationship("ForecastRun", back_populates="points")

    __table_args__ = (
        UniqueConstraint("run_id", "forecast_date", name="uq_forecast_points_run_date"),
        Index("ix_forecast_points_run_date", "run_id", "forecast_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<ForecastPoint run={self.run_id} date={self.forecast_date} "
            f"yhat={self.yhat} actual={self.actual}>"
        )


# ---------------------------------------------------------------------------
# ForecastEvaluation  (Stage 5)
# ---------------------------------------------------------------------------

class ForecastEvaluation(Base):
    """
    Aggregated error metrics for a completed ForecastRun.

    One row per ForecastRun (1-to-1 relationship).

    Metrics:
        mae  – Mean Absolute Error
        rmse – Root Mean Squared Error
        mape – Mean Absolute Percentage Error (NULL when any actual == 0)
        wape – Weighted Absolute Percentage Error (always computed; robust to zeros)
              = sum(|actual - predicted|) / sum(actual)
    """

    __tablename__ = "forecast_evaluations"

    id = Column(Integer, primary_key=True, index=True)

    run_id = Column(
        Integer,
        ForeignKey("forecast_runs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    mae           = Column(Numeric(14, 4), nullable=True)
    rmse          = Column(Numeric(14, 4), nullable=True)
    mape          = Column(Numeric(10, 4), nullable=True,
                           comment="NULL when any actual=0 (undefined)")
    wape          = Column(Numeric(10, 4), nullable=True,
                           comment="Weighted APE; robust to zero actuals")
    n_eval_points = Column(Integer, nullable=False, default=0,
                           comment="number of test-period rows used in evaluation")

    computed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship
    run = relationship("ForecastRun", back_populates="evaluation")

    def __repr__(self) -> str:
        return (
            f"<ForecastEvaluation run={self.run_id} "
            f"mae={self.mae} rmse={self.rmse} wape={self.wape}>"
        )

