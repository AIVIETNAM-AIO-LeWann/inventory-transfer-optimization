"""Manage Streamlit session state for the dashboard."""

import streamlit as st

from src.dashboard.constants import (
    ACTIVE_DATA_KEY,
    DATA_FINGERPRINT_KEY,
    DATA_SOURCE_KEY,
    MODEL_ARTIFACTS_KEY,
    MODEL_STATUS_KEY,
    MODEL_TRAINING_ERROR_KEY,
    MODEL_TRAINING_RESULT_KEY,
    PIPELINE_RESULT_KEY,
    PIPELINE_SETTINGS_KEY,
    UPLOADED_FILE_NAME_KEY,
)
from src.dashboard.model_artifacts import (
    ForecastModelArtifact,
    validate_model_artifact,
)
from src.data_loader import ProjectData
from src.optimization_pipeline import (
    OptimizationPipelineResult,
)


def activate_project_data(
    project_data: ProjectData,
    data_source: str,
    data_fingerprint: str,
    uploaded_file_name: str | None = None,
) -> None:
    """Store active data and invalidate stale output."""

    previous_fingerprint = (
        st.session_state.get(
            DATA_FINGERPRINT_KEY
        )
    )

    if previous_fingerprint != data_fingerprint:
        clear_pipeline_result()
        clear_model_training_state()

    st.session_state[
        ACTIVE_DATA_KEY
    ] = project_data

    st.session_state[
        DATA_SOURCE_KEY
    ] = data_source

    st.session_state[
        DATA_FINGERPRINT_KEY
    ] = data_fingerprint

    st.session_state[
        UPLOADED_FILE_NAME_KEY
    ] = uploaded_file_name


def get_active_project_data() -> ProjectData | None:
    """Return the active validated dataset."""

    return st.session_state.get(
        ACTIVE_DATA_KEY
    )


def get_data_fingerprint() -> str | None:
    """Return the active dataset fingerprint."""

    return st.session_state.get(
        DATA_FINGERPRINT_KEY
    )


def get_data_source() -> str | None:
    """Return the active dashboard data source."""

    data_source = st.session_state.get(
        DATA_SOURCE_KEY
    )

    if data_source is None:
        return None

    if not isinstance(data_source, str):
        raise TypeError(
            "Stored data source must be a string."
        )

    return data_source


def store_pipeline_result(
    result: OptimizationPipelineResult,
    settings: dict[str, str | int],
) -> None:
    """Store a completed pipeline result and settings."""

    if not isinstance(
        result,
        OptimizationPipelineResult,
    ):
        raise TypeError(
            "result must be an "
            "OptimizationPipelineResult."
        )

    if not isinstance(settings, dict):
        raise TypeError(
            "settings must be a dictionary."
        )

    st.session_state[
        PIPELINE_RESULT_KEY
    ] = result

    st.session_state[
        PIPELINE_SETTINGS_KEY
    ] = settings.copy()


def get_pipeline_result(
) -> OptimizationPipelineResult | None:
    """Return the most recent pipeline result."""

    return st.session_state.get(
        PIPELINE_RESULT_KEY
    )


def get_pipeline_settings(
) -> dict[str, str | int] | None:
    """Return settings used by the latest pipeline run."""

    settings = st.session_state.get(
        PIPELINE_SETTINGS_KEY
    )

    if settings is None:
        return None

    if not isinstance(settings, dict):
        raise TypeError(
            "Stored pipeline settings must "
            "be a dictionary."
        )

    return settings.copy()


def pipeline_settings_match(
    current_settings: dict[
        str,
        str | int,
    ],
) -> bool:
    """Check whether current controls match stored results."""

    if not isinstance(
        current_settings,
        dict,
    ):
        raise TypeError(
            "current_settings must be "
            "a dictionary."
        )

    stored_settings = (
        get_pipeline_settings()
    )

    if stored_settings is None:
        return False

    return (
        stored_settings
        == current_settings
    )


def clear_pipeline_result() -> None:
    """Remove the current pipeline output."""

    st.session_state.pop(
        PIPELINE_RESULT_KEY,
        None,
    )

    st.session_state.pop(
        PIPELINE_SETTINGS_KEY,
        None,
    )


def store_model_artifacts(
    artifacts: dict[
        str,
        ForecastModelArtifact,
    ],
) -> None:
    """Store newly trained model artifacts."""

    if not isinstance(artifacts, dict):
        raise TypeError(
            "artifacts must be a dictionary."
        )

    if not artifacts:
        raise ValueError(
            "artifacts must not be empty."
        )

    for method, artifact in artifacts.items():
        validate_model_artifact(
            artifact
        )

        if method != artifact.method:
            raise ValueError(
                "Artifact dictionary keys must "
                "match artifact methods."
            )

    st.session_state[
        MODEL_ARTIFACTS_KEY
    ] = artifacts.copy()

    st.session_state[
        MODEL_STATUS_KEY
    ] = {
        method: "ready"
        for method in artifacts
    }

    st.session_state[
        MODEL_TRAINING_RESULT_KEY
    ] = tuple(
        artifacts
    )

    st.session_state.pop(
        MODEL_TRAINING_ERROR_KEY,
        None,
    )


def get_model_artifacts(
) -> dict[str, ForecastModelArtifact]:
    """Return model artifacts trained in this session."""

    artifacts = st.session_state.get(
        MODEL_ARTIFACTS_KEY,
        {},
    )

    if not isinstance(artifacts, dict):
        raise TypeError(
            "Stored model artifacts must "
            "be a dictionary."
        )

    return artifacts


def get_model_artifact(
    method: str,
) -> ForecastModelArtifact | None:
    """Return one model artifact from session state."""

    return get_model_artifacts().get(
        method
    )


def store_model_training_error(
    error_message: str,
) -> None:
    """Store a model-training error message."""

    if not isinstance(error_message, str):
        raise TypeError(
            "error_message must be a string."
        )

    normalized_message = (
        error_message.strip()
    )

    if not normalized_message:
        raise ValueError(
            "error_message must not be empty."
        )

    st.session_state[
        MODEL_TRAINING_ERROR_KEY
    ] = normalized_message

    st.session_state[
        MODEL_STATUS_KEY
    ] = "failed"


def get_model_training_error() -> str | None:
    """Return the latest model-training error."""

    return st.session_state.get(
        MODEL_TRAINING_ERROR_KEY
    )


def clear_model_training_state() -> None:
    """Remove model state associated with old data."""

    for key in (
        MODEL_ARTIFACTS_KEY,
        MODEL_STATUS_KEY,
        MODEL_TRAINING_RESULT_KEY,
        MODEL_TRAINING_ERROR_KEY,
    ):
        st.session_state.pop(
            key,
            None,
        )


def clear_dashboard_state() -> None:
    """Remove all dashboard-owned session state."""

    for key in (
        ACTIVE_DATA_KEY,
        DATA_SOURCE_KEY,
        DATA_FINGERPRINT_KEY,
        UPLOADED_FILE_NAME_KEY,
        PIPELINE_RESULT_KEY,
        PIPELINE_SETTINGS_KEY,
        MODEL_ARTIFACTS_KEY,
        MODEL_STATUS_KEY,
        MODEL_TRAINING_RESULT_KEY,
        MODEL_TRAINING_ERROR_KEY,
    ):
        st.session_state.pop(
            key,
            None,
        )
