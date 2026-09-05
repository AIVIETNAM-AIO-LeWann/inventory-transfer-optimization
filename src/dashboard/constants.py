"""Shared constants for the Streamlit dashboard."""

from src.config import (
    GENETIC_ALGORITHM_OPTIMIZER,
    GREEDY_OPTIMIZER,
    HISTORICAL_AVERAGE_METHOD,
    INTER_CITY_ROUTE,
    INTRA_CITY_ROUTE,
    LINEAR_PROGRAMMING_OPTIMIZER,
    MOVING_AVERAGE_METHOD,
    REQUIRED_DATA_FILES,
    SUPPORTED_FORECAST_METHODS,
)
from src.dashboard.model_artifacts import (
    ADABOOST_METHOD,
    RANDOM_FOREST_METHOD,
    SUPPORTED_MACHINE_LEARNING_METHODS,
)
from src.inventory_analyzer import (
    BALANCED_STATUS,
    EXCESS_STATUS,
    SHORTAGE_STATUS,
)


# =========================================================
# APPLICATION
# =========================================================

APP_TITLE = "Inventory Transfer Optimization"

APP_DESCRIPTION = (
    "Forecast demand, analyze inventory imbalances, "
    "and create inventory transfer plans."
)

APP_PAGE_ICON = ":material/inventory_2:"

APP_LAYOUT = "wide"


# =========================================================
# DATA SOURCES
# =========================================================

SAMPLE_DATA_SOURCE = "sample"
UPLOADED_DATA_SOURCE = "uploaded"

DATA_SOURCE_DISPLAY_NAMES = {
    SAMPLE_DATA_SOURCE: "Sample Dataset",
    UPLOADED_DATA_SOURCE: "Upload ZIP Dataset",
}

REQUIRED_UPLOAD_FILENAMES = tuple(
    data_file.name
    for data_file in REQUIRED_DATA_FILES
)

MAX_UPLOAD_SIZE_MB = 100
MAX_UNCOMPRESSED_UPLOAD_SIZE_MB = 500


# =========================================================
# FORECAST METHODS
# =========================================================

FORECAST_METHOD_DISPLAY_NAMES = {
    HISTORICAL_AVERAGE_METHOD: (
        "Historical Average"
    ),
    MOVING_AVERAGE_METHOD: (
        "Moving Average"
    ),
    RANDOM_FOREST_METHOD: (
        "Random Forest"
    ),
    ADABOOST_METHOD: (
        "AdaBoost"
    ),
}

MACHINE_LEARNING_DISPLAY_NAMES = {
    RANDOM_FOREST_METHOD: "Random Forest",
    ADABOOST_METHOD: "AdaBoost",
}

DASHBOARD_FORECAST_METHODS = (
    *SUPPORTED_FORECAST_METHODS,
    *SUPPORTED_MACHINE_LEARNING_METHODS,
)


# =========================================================
# OPTIMIZATION ALGORITHMS
# =========================================================

OPTIMIZER_DISPLAY_NAMES = {
    GREEDY_OPTIMIZER: "Greedy",
    LINEAR_PROGRAMMING_OPTIMIZER: (
        "Linear Programming"
    ),
    GENETIC_ALGORITHM_OPTIMIZER: (
        "Genetic Algorithm"
    ),
}


# =========================================================
# DASHBOARD NAVIGATION
# =========================================================

OVERVIEW_PAGE = "overview"
FORECAST_PAGE = "demand_forecast"
INVENTORY_PAGE = "inventory_health"
TRANSFER_PAGE = "transfer_plan"
NETWORK_MAP_PAGE = "network_map"
DATA_MODELS_PAGE = "data_models"

DASHBOARD_PAGES = (
    OVERVIEW_PAGE,
    FORECAST_PAGE,
    INVENTORY_PAGE,
    TRANSFER_PAGE,
    NETWORK_MAP_PAGE,
    DATA_MODELS_PAGE,
)

DASHBOARD_PAGE_DISPLAY_NAMES = {
    OVERVIEW_PAGE: "📊 Overview",
    FORECAST_PAGE: "📈 Demand Forecast",
    INVENTORY_PAGE: "🏥 Inventory Health",
    TRANSFER_PAGE: "🚚 Transfer Plan",
    NETWORK_MAP_PAGE: "🗺️ Network Map",
    DATA_MODELS_PAGE: "⚙️ Data & Models",
}


# =========================================================
# COLOR PALETTE
# =========================================================

PRIMARY_COLOR = "#006D5B"
SECONDARY_COLOR = "#344054"

BACKGROUND_COLOR = "#F7F8FA"
CARD_BACKGROUND_COLOR = "#FFFFFF"

TEXT_COLOR = "#101828"
MUTED_TEXT_COLOR = "#667085"

SUCCESS_COLOR = "#12B76A"
WARNING_COLOR = "#F79009"
ERROR_COLOR = "#D92D20"
INFO_COLOR = "#2E90FA"

STATUS_COLORS = {
    SHORTAGE_STATUS: ERROR_COLOR,
    BALANCED_STATUS: SUCCESS_COLOR,
    EXCESS_STATUS: INFO_COLOR,
}

STATUS_MARKER_COLORS = {
    SHORTAGE_STATUS: "red",
    BALANCED_STATUS: "green",
    EXCESS_STATUS: "blue",
}

STATUS_DISPLAY_NAMES = {
    SHORTAGE_STATUS: "Shortage",
    BALANCED_STATUS: "Balanced",
    EXCESS_STATUS: "Excess",
}

ROUTE_COLORS = {
    INTRA_CITY_ROUTE: INFO_COLOR,
    INTER_CITY_ROUTE: WARNING_COLOR,
}

ROUTE_DISPLAY_NAMES = {
    INTRA_CITY_ROUTE: "Intra-city",
    INTER_CITY_ROUTE: "Inter-city",
}


# =========================================================
# MAP SETTINGS
# =========================================================

DEFAULT_MAP_ZOOM = 5
DEFAULT_MAX_DISPLAYED_ROUTES = 100

STORE_MARKER_ICON = "shopping-cart"
STORE_MARKER_ICON_PREFIX = "glyphicon"


# =========================================================
# SESSION STATE KEYS
# =========================================================

ACTIVE_DATA_KEY = "active_project_data"
DATA_SOURCE_KEY = "selected_data_source"
DATA_SOURCE_SELECTION_KEY = (
    "data_source_selection"
)
DATA_FINGERPRINT_KEY = "data_fingerprint"
UPLOADED_FILE_NAME_KEY = (
    "uploaded_file_name"
)

PIPELINE_RESULT_KEY = "pipeline_result"
PIPELINE_SETTINGS_KEY = "pipeline_settings"

DATA_VALIDATION_KEY = (
    "data_validation_result"
)

MODEL_STATUS_KEY = "model_status"
MODEL_TRAINING_RESULT_KEY = (
    "model_training_result"
)
MODEL_ARTIFACTS_KEY = "model_artifacts"
MODEL_TRAINING_ERROR_KEY = (
    "model_training_error"
)
MODEL_TRAINING_REQUEST_KEY = (
    "model_training_request"
)
DASHBOARD_PAGE_SELECTION_KEY = (
    "dashboard_page_selection"
)