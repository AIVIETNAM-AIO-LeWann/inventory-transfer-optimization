"""Render dataset details and model management."""

import pandas as pd
import streamlit as st

from src.dashboard.components import (
    render_configuration_card,
    render_section_heading,
)
from src.dashboard.constants import (
    DATA_SOURCE_DISPLAY_NAMES,
    MACHINE_LEARNING_DISPLAY_NAMES,
    SAMPLE_DATA_SOURCE,
    SAMPLE_MODEL_DATASET_ID,
)
from src.dashboard.formatters import (
    dataframe_to_csv_bytes,
    format_decimal,
    format_integer,
)
from src.dashboard.model_artifacts import (
    SUPPORTED_MACHINE_LEARNING_METHODS,
    ForecastModelArtifact,
    load_model_artifact,
    model_artifact_exists,
)
from src.dashboard.model_training import (
    train_forecast_models,
)
from src.dashboard.services import (
    run_pipeline,
)
from src.dashboard.state import (
    get_data_fingerprint,
    get_model_artifacts,
    get_model_training_error,
    store_model_artifacts,
    store_model_training_error,
)
from src.data_loader import ProjectData


def create_dataset_summary(
    project_data: ProjectData,
) -> pd.DataFrame:
    """Create a row-count summary for all datasets."""

    return pd.DataFrame(
        [
            {
                "Dataset": "Stores",
                "Rows": len(project_data.stores),
            },
            {
                "Dataset": "Products",
                "Rows": len(project_data.products),
            },
            {
                "Dataset": "Sales",
                "Rows": len(project_data.sales),
            },
            {
                "Dataset": "Inventory",
                "Rows": len(project_data.inventory),
            },
            {
                "Dataset": "Distance Matrix",
                "Rows": len(
                    project_data.distance_matrix
                ),
            },
            {
                "Dataset": "Duration Matrix",
                "Rows": len(
                    project_data.duration_matrix
                ),
            },
            {
                "Dataset": "Transport Cost Matrix",
                "Rows": len(
                    project_data
                    .transport_cost_matrix
                ),
            },
        ]
    )


def load_available_model_artifacts(
    dataset_fingerprint: str | None,
) -> dict[str, ForecastModelArtifact]:
    """Load compatible artifacts from session or disk."""

    artifacts = (
        get_model_artifacts().copy()
    )

    if dataset_fingerprint is None:
        return artifacts

    for method in (
        SUPPORTED_MACHINE_LEARNING_METHODS
    ):
        if method in artifacts:
            continue

        if not model_artifact_exists(
            method=method,
            dataset_fingerprint=(
                dataset_fingerprint
            ),
        ):
            continue

        try:
            artifacts[method] = (
                load_model_artifact(
                    method=method,
                    dataset_fingerprint=(
                        dataset_fingerprint
                    ),
                )
            )
        except (
            OSError,
            TypeError,
            ValueError,
        ):
            continue

    return artifacts


def create_model_status_table(
    artifacts: dict[
        str,
        ForecastModelArtifact,
    ],
) -> pd.DataFrame:
    """Create model readiness and evaluation table."""

    rows: list[dict[str, str]] = [
        {
            "Method": "Historical Average",
            "Training": "Not required",
            "Status": "Ready",
            "Validation MAE": "N/A",
            "Validation RMSE": "N/A",
            "Test MAE": "N/A",
            "Test RMSE": "N/A",
        },
        {
            "Method": "Moving Average",
            "Training": "Not required",
            "Status": "Ready",
            "Validation MAE": "N/A",
            "Validation RMSE": "N/A",
            "Test MAE": "N/A",
            "Test RMSE": "N/A",
        },
    ]

    for method in (
        SUPPORTED_MACHINE_LEARNING_METHODS
    ):
        artifact = artifacts.get(method)

        if artifact is None:
            rows.append(
                {
                    "Method": (
                        MACHINE_LEARNING_DISPLAY_NAMES[
                            method
                        ]
                    ),
                    "Training": (
                        "Required per dataset"
                    ),
                    "Status": "Not trained",
                    "Validation MAE": "N/A",
                    "Validation RMSE": "N/A",
                    "Test MAE": "N/A",
                    "Test RMSE": "N/A",
                }
            )

            continue

        rows.append(
            {
                "Method": (
                    MACHINE_LEARNING_DISPLAY_NAMES[
                        method
                    ]
                ),
                "Training": "Completed",
                "Status": "Ready",
                "Validation MAE": (
                    format_decimal(
                        artifact.validation_mae,
                        decimal_places=4,
                    )
                ),
                "Validation RMSE": (
                    format_decimal(
                        artifact.validation_rmse,
                        decimal_places=4,
                    )
                ),
                "Test MAE": (
                    format_decimal(
                        artifact.test_mae,
                        decimal_places=4,
                    )
                ),
                "Test RMSE": (
                    format_decimal(
                        artifact.test_rmse,
                        decimal_places=4,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def render_model_training_section(
    project_data: ProjectData,
    dataset_fingerprint: str | None,
) -> dict[str, ForecastModelArtifact]:
    """Render controls for training forecasting models."""

    artifacts = load_available_model_artifacts(
        dataset_fingerprint=(
            dataset_fingerprint
        )
    )

    render_section_heading(
        title="Model Lab",
        description=(
            "Train dataset-specific forecasting models "
            "and save reusable artifacts."
        ),
    )

    with st.container(border=True):
        readiness_columns = st.columns(3)
        readiness_columns[0].metric(
            "Available ML Models",
            format_integer(len(artifacts)),
        )
        readiness_columns[1].metric(
            "Training Pending",
            format_integer(
                len(
                    SUPPORTED_MACHINE_LEARNING_METHODS
                )
                - len(artifacts)
            ),
        )
        readiness_columns[2].metric(
            "Artifact Compatibility",
            (
                "Dataset matched"
                if dataset_fingerprint is not None
                else "Unavailable"
            ),
        )

        st.caption(
            "Historical Average and Moving Average are "
            "always ready. Random Forest and AdaBoost "
            "must be trained for the active dataset."
        )

        selected_methods = st.multiselect(
            "Models to Train",
            options=(
                SUPPORTED_MACHINE_LEARNING_METHODS
            ),
            default=(
                SUPPORTED_MACHINE_LEARNING_METHODS
            ),
            format_func=lambda method: (
                MACHINE_LEARNING_DISPLAY_NAMES[
                    method
                ]
            ),
            help=(
                "Existing compatible artifacts are "
                "replaced when a model is retrained."
            ),
        )

        train_requested = st.button(
            "Train Selected Models",
            type="primary",
            icon=":material/model_training:",
            width="stretch",
            disabled=(
                not selected_methods
                or dataset_fingerprint is None
            ),
        )

    if train_requested:
        try:
            with st.spinner(
                "Preparing features and training "
                "forecasting models..."
            ):
                trained_artifacts = (
                    train_forecast_models(
                        project_data=project_data,
                        dataset_fingerprint=(
                            dataset_fingerprint
                        ),
                        methods=tuple(
                            selected_methods
                        ),
                        save_artifacts=True,
                    )
                )

            store_model_artifacts(
                trained_artifacts
            )

            run_pipeline.clear()

            st.success(
                "Model training completed. "
                "Compatible artifacts were saved."
            )

        except Exception as error:
            store_model_training_error(
                str(error)
            )

            st.error(
                "Model training could not "
                "be completed."
            )

            st.exception(error)

    training_error = (
        get_model_training_error()
    )

    if training_error is not None:
        st.warning(
            "Latest training error: "
            f"{training_error}"
        )

    return load_available_model_artifacts(
        dataset_fingerprint=dataset_fingerprint
    )


def render_feature_importance(
    artifacts: dict[
        str,
        ForecastModelArtifact,
    ],
) -> None:
    """Render feature importance for a trained model."""

    if not artifacts:
        st.info(
            "Train at least one machine-learning "
            "model to view feature importance."
        )
        return

    available_methods = tuple(
        method
        for method in (
            SUPPORTED_MACHINE_LEARNING_METHODS
        )
        if method in artifacts
    )

    selected_method = st.selectbox(
        "Feature Importance Model",
        options=available_methods,
        format_func=lambda method: (
            MACHINE_LEARNING_DISPLAY_NAMES[
                method
            ]
        ),
        key="feature_importance_model",
    )

    artifact = artifacts[
        selected_method
    ]

    importance_data = (
        artifact.feature_importance
        .head(20)
        .copy()
    )

    if importance_data.empty:
        st.info(
            "The selected model has no feature-importance "
            "records to display."
        )
        return

    top_feature = importance_data.iloc[0]
    importance_columns = st.columns(3)
    importance_columns[0].metric(
        "Top Feature",
        str(top_feature["feature"]),
    )
    importance_columns[1].metric(
        "Top Importance",
        format_decimal(
            top_feature["importance"],
            decimal_places=4,
        ),
    )
    importance_columns[2].metric(
        "Features Used",
        format_integer(
            len(artifact.feature_names)
        ),
    )

    st.caption(
        "Trained at "
        f"{artifact.trained_at} for the active "
        "dataset fingerprint."
    )

    st.bar_chart(
        importance_data.set_index(
            "feature"
        )["importance"],
        horizontal=True,
        color="#006D5B",
    )

    with st.expander(
        "View Complete Feature Importance"
    ):
        st.dataframe(
            artifact.feature_importance,
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Download Feature Importance CSV",
            data=dataframe_to_csv_bytes(
                artifact.feature_importance
            ),
            file_name=(
                f"{selected_method}_"
                "feature_importance.csv"
            ),
            mime="text/csv",
            icon=":material/download:",
        )


def render_source_data_previews(
    project_data: ProjectData,
) -> None:
    """Render expandable source-data previews."""

    recent_sales = (
        project_data.sales.sort_values(
            "date",
            ascending=False,
        )
    )

    source_datasets = (
        (
            "Stores",
            project_data.stores,
            "stores.csv",
            None,
        ),
        (
            "Products",
            project_data.products,
            "products.csv",
            None,
        ),
        (
            "Recent Sales",
            recent_sales,
            "sales.csv",
            1_000,
        ),
        (
            "Current Inventory",
            project_data.inventory,
            "inventory.csv",
            None,
        ),
        (
            "Distance Matrix",
            project_data.distance_matrix,
            "distance_matrix.csv",
            1_000,
        ),
        (
            "Duration Matrix",
            project_data.duration_matrix,
            "duration_matrix.csv",
            1_000,
        ),
        (
            "Transport Cost Matrix",
            project_data.transport_cost_matrix,
            "transport_cost_matrix.csv",
            1_000,
        ),
    )

    for title, data, file_name, preview_limit in (
        source_datasets
    ):
        preview_data = (
            data.head(preview_limit)
            if preview_limit is not None
            else data
        )

        with st.expander(
            f"{title} ({len(data):,} rows)"
        ):
            if (
                preview_limit is not None
                and len(data) > preview_limit
            ):
                st.caption(
                    f"Previewing the first "
                    f"{preview_limit:,} rows. The download "
                    "contains the complete dataset."
                )

            st.dataframe(
                preview_data,
                hide_index=True,
                width="stretch",
            )

            st.download_button(
                f"Download {title} CSV",
                data=dataframe_to_csv_bytes(data),
                file_name=file_name,
                mime="text/csv",
                icon=":material/download:",
                key=f"download_{file_name}",
            )


def render_data_models_tab(
    project_data: ProjectData,
    data_source: str,
    uploaded_file_name: str | None,
) -> None:
    """Render data details and model management."""

    st.subheader("Data & Models")
    st.caption(
        "Inspect the active dataset, manage compatible "
        "forecast models, and export source tables."
    )

    source_name = (
        DATA_SOURCE_DISPLAY_NAMES.get(
            data_source,
            data_source,
        )
    )

    dataset_fingerprint = (
        get_data_fingerprint()
    )
    model_dataset_fingerprint = (
        SAMPLE_MODEL_DATASET_ID
        if data_source == SAMPLE_DATA_SOURCE
        else dataset_fingerprint
    )
    sales_dates = pd.to_datetime(
        project_data.sales["date"],
        errors="coerce",
    )
    sales_coverage = (
        f"{sales_dates.min().date().isoformat()} to "
        f"{sales_dates.max().date().isoformat()}"
    )
    fingerprint_label = (
        f"{dataset_fingerprint[:12]}..."
        if dataset_fingerprint is not None
        else "Unavailable"
    )

    render_configuration_card(
        title="Active Dataset",
        items=(
            ("Source", source_name),
            (
                "Uploaded File",
                uploaded_file_name
                or "Not applicable",
            ),
            ("Sales Coverage", sales_coverage),
            (
                "Compatibility Key",
                fingerprint_label,
            ),
            (
                "Model Artifact Scope",
                model_dataset_fingerprint
                or "Unavailable",
            ),
        ),
    )

    render_section_heading(
        title="Data Snapshot",
        description=(
            "The active dataset contains seven validated "
            "tables used by forecasting and optimization."
        ),
    )

    data_metrics = st.columns(4)
    data_metrics[0].metric(
        "Stores",
        format_integer(len(project_data.stores)),
    )
    data_metrics[1].metric(
        "Products",
        format_integer(len(project_data.products)),
    )
    data_metrics[2].metric(
        "Sales Records",
        format_integer(len(project_data.sales)),
    )
    data_metrics[3].metric(
        "Inventory Records",
        format_integer(
            len(project_data.inventory)
        ),
    )

    st.dataframe(
        create_dataset_summary(
            project_data
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Dataset": st.column_config.TextColumn(
                "Dataset"
            ),
            "Rows": st.column_config.NumberColumn(
                "Rows",
                format="%,d",
            ),
        },
    )

    artifacts = (
        render_model_training_section(
            project_data=project_data,
            dataset_fingerprint=(
                model_dataset_fingerprint
            ),
        )
    )

    render_section_heading(
        title="Forecast Model Readiness",
        description=(
            "Compare training requirements and evaluation "
            "metrics for every forecast method."
        ),
    )

    st.dataframe(
        create_model_status_table(
            artifacts
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Method": "Forecast Method",
            "Training": "Training Requirement",
            "Status": "Readiness",
        },
    )

    render_section_heading(
        title="Feature Importance",
        description=(
            "Understand which time-series and business "
            "features influence a trained model most."
        ),
    )

    render_feature_importance(
        artifacts
    )

    st.info(
        "Optuna hyperparameter tuning and SHAP model "
        "explanations are planned for the next stage."
    )

    render_section_heading(
        title="Source Data",
        description=(
            "Preview validated input tables or download "
            "their complete CSV contents."
        ),
    )

    render_source_data_previews(
        project_data
    )
