"""Application services connecting the dashboard to core logic."""

from hashlib import sha256

import streamlit as st
import pandas as pd

from src.config import (
    SUPPORTED_FORECAST_METHODS,
)
from src.dashboard.ml_pipeline import (
    run_ml_optimization_pipeline,
)
from src.dashboard.model_artifacts import (
    SUPPORTED_MACHINE_LEARNING_METHODS,
    load_model_artifact,
)
from src.data_ingestion.zip_loader import (
    load_project_data_from_zip,
)
from src.data_loader import (
    ProjectData,
    load_all_data,
)
from src.optimization_pipeline import (
    OptimizationPipelineResult,
    run_optimization_pipeline,
)


@st.cache_data(show_spinner=False)
def load_sample_project_data() -> ProjectData:
    """Load and cache the checked-in sample dataset."""

    return load_all_data()


@st.cache_data(show_spinner=False)
def load_uploaded_project_data(
    zip_bytes: bytes,
) -> ProjectData:
    """Load and cache a validated uploaded ZIP dataset."""

    return load_project_data_from_zip(
        zip_bytes
    )


@st.cache_data(show_spinner=False)
def run_pipeline(
    project_data: ProjectData,
    requested_horizon_days: int,
    forecast_method: str,
    optimizer_name: str,
    moving_average_window_days: int,
    dataset_fingerprint: str | None = None,
) -> OptimizationPipelineResult:
    """Run the correct pipeline for a forecast method."""

    if forecast_method in SUPPORTED_FORECAST_METHODS:
        return run_optimization_pipeline(
            project_data=project_data,
            requested_horizon_days=(
                requested_horizon_days
            ),
            forecast_method=forecast_method,
            optimizer_name=optimizer_name,
            moving_average_window_days=(
                moving_average_window_days
            ),
        )

    if (
        forecast_method
        in SUPPORTED_MACHINE_LEARNING_METHODS
    ):
        if dataset_fingerprint is None:
            raise ValueError(
                "dataset_fingerprint is required "
                "for machine-learning forecasts."
            )

        artifact = load_model_artifact(
            method=forecast_method,
            dataset_fingerprint=(
                dataset_fingerprint
            ),
        )

        return run_ml_optimization_pipeline(
            project_data=project_data,
            artifact=artifact,
            dataset_fingerprint=(
                dataset_fingerprint
            ),
            requested_horizon_days=(
                requested_horizon_days
            ),
            optimizer_name=optimizer_name,
        )

    raise ValueError(
        f"Unsupported forecast method: "
        f"{forecast_method}"
    )


def _prepare_frame_for_fingerprint(
    data: pd.DataFrame,
    sort_columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Create a deterministic DataFrame for hashing."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    normalized_data = data.copy()
    normalized_data.columns = (
        normalized_data.columns.astype(str)
    )

    if sort_columns is not None:
        missing_columns = (
            set(sort_columns)
            - set(normalized_data.columns)
        )

        if missing_columns:
            raise ValueError(
                "Fingerprint sort columns are missing: "
                f"{sorted(missing_columns)}"
            )

        normalized_data = (
            normalized_data
            .sort_values(
                by=list(sort_columns),
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
    else:
        normalized_data.index = (
            normalized_data.index.astype(str)
        )

        normalized_data = (
            normalized_data
            .sort_index()
            .sort_index(axis=1)
        )

    normalized_data = normalized_data.reindex(
        sorted(normalized_data.columns),
        axis=1,
    )

    return normalized_data


def create_project_data_fingerprint(
    project_data: ProjectData,
) -> str:
    """Create a deterministic identity for project data."""

    if not isinstance(project_data, ProjectData):
        raise TypeError(
            "project_data must be a ProjectData instance."
        )

    datasets = (
        (
            "stores",
            _prepare_frame_for_fingerprint(
                project_data.stores,
                sort_columns=("store_id",),
            ),
        ),
        (
            "products",
            _prepare_frame_for_fingerprint(
                project_data.products,
                sort_columns=("product_id",),
            ),
        ),
        (
            "sales",
            _prepare_frame_for_fingerprint(
                project_data.sales,
                sort_columns=(
                    "date",
                    "store_id",
                    "product_id",
                ),
            ),
        ),
        (
            "inventory",
            _prepare_frame_for_fingerprint(
                project_data.inventory,
                sort_columns=(
                    "store_id",
                    "product_id",
                ),
            ),
        ),
        (
            "distance_matrix",
            _prepare_frame_for_fingerprint(
                project_data.distance_matrix
            ),
        ),
        (
            "duration_matrix",
            _prepare_frame_for_fingerprint(
                project_data.duration_matrix
            ),
        ),
        (
            "transport_cost_matrix",
            _prepare_frame_for_fingerprint(
                project_data.transport_cost_matrix
            ),
        ),
    )

    fingerprint = sha256()

    for dataset_name, dataset in datasets:
        fingerprint.update(
            dataset_name.encode("utf-8")
        )

        fingerprint.update(
            "\x1f".join(
                dataset.columns
            ).encode("utf-8")
        )

        fingerprint.update(
            "\x1f".join(
                str(dtype)
                for dtype in dataset.dtypes
            ).encode("utf-8")
        )

        row_hashes = pd.util.hash_pandas_object(
            dataset,
            index=True,
            categorize=True,
        )

        fingerprint.update(
            row_hashes.to_numpy(
                dtype="uint64"
            ).tobytes()
        )

    return fingerprint.hexdigest()