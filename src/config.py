"""Shared configuration for the Inventory Transfer Optimization System."""

from pathlib import Path


# =========================================================
# 1. PROJECT PATHS
# =========================================================

# config.py is located inside the src/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
LOGS_DIR = PROJECT_ROOT / "logs"


# =========================================================
# 2. INPUT DATA FILES
# =========================================================

STORES_FILE = DATA_DIR / "stores.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"
SALES_FILE = DATA_DIR / "sales_data.csv"
INVENTORY_FILE = DATA_DIR / "inventory_data.csv"
DISTANCE_MATRIX_FILE = DATA_DIR / "distance_matrix.csv"
DURATION_MATRIX_FILE = DATA_DIR / "duration_matrix.csv"
TRANSPORT_COST_MATRIX_FILE = DATA_DIR / "transport_cost_matrix.csv"

# Used by the data validator to check required input files
REQUIRED_DATA_FILES = (
    STORES_FILE,
    PRODUCTS_FILE,
    SALES_FILE,
    INVENTORY_FILE,
    DISTANCE_MATRIX_FILE,
    DURATION_MATRIX_FILE,
    TRANSPORT_COST_MATRIX_FILE,
)


# =========================================================
# 3. RESULT FILES
# =========================================================

INVENTORY_ANALYSIS_FILE = (
    RESULTS_DIR / "inventory_analysis.csv"
)

ROUTE_ANALYSIS_FILE = (
    RESULTS_DIR / "route_analysis.csv"
)

GREEDY_TRANSFER_PLAN_FILE = (
    RESULTS_DIR / "greedy_transfer_plan.csv"
)

BEST_TRANSFER_PLAN_FILE = (
    RESULTS_DIR / "best_transfer_plan.csv"
)

ALGORITHM_COMPARISON_FILE = (
    RESULTS_DIR / "algorithm_comparison.csv"
)

LINEAR_PROGRAMMING_TRANSFER_PLAN_FILE = (
    RESULTS_DIR / "linear_programming_transfer_plan.csv"
)

GENETIC_ALGORITHM_TRANSFER_PLAN_FILE = (
    RESULTS_DIR / "genetic_algorithm_transfer_plan.csv"
)

DAILY_DEMAND_FORECAST_FILE = (
    RESULTS_DIR / "daily_demand_forecast.csv"
)

FORECAST_BACKTEST_RESULTS_FILE = (
    RESULTS_DIR / "forecast_backtest_results.csv"
)

FORECAST_BACKTEST_SUMMARY_FILE = (
    RESULTS_DIR / "forecast_backtest_summary.csv"
)

# =========================================================
# 4. RANDOM SEED
# =========================================================

# Ensures reproducible sample data and optimization results
RANDOM_SEED = 2026


# =========================================================
# 5. STORE GENERATION
# =========================================================

STORE_CITIES = {
    "Hanoi": {
        "lat_range": (20.90, 21.10),
        "lon_range": (105.70, 105.90),
        "count": 7,
    },
    "Da Nang": {
        "lat_range": (16.00, 16.10),
        "lon_range": (108.20, 108.30),
        "count": 5,
    },
    "Ho Chi Minh City": {
        "lat_range": (10.70, 10.90),
        "lon_range": (106.60, 106.80),
        "count": 8,
    },
}

# Calculate the total number of stores from the city configuration
NUM_STORES = sum(
    city_config["count"]
    for city_config in STORE_CITIES.values()
)


# =========================================================
# 6. PRODUCT GENERATION
# =========================================================

NUM_PRODUCTS = 30

PRODUCT_CATEGORIES = (
    "Electronics",
    "Clothing",
    "Home Goods",
    "Food",
    "Beauty",
)

MIN_PRODUCT_COST = 20_000
MAX_PRODUCT_COST = 2_000_000

# Product selling prices are generated using these profit margins
MIN_PROFIT_MARGIN = 0.10
MAX_PROFIT_MARGIN = 0.50


# =========================================================
# 7. SALES GENERATION
# =========================================================

SALES_START_DATE = "2026-01-01"
SALES_DAYS = 365

# Average daily demand range for generated sales data
MIN_DAILY_DEMAND = 1
MAX_DAILY_DEMAND = 20

# Sales may increase during weekends
WEEKEND_SALES_MULTIPLIER = 1.20


# =========================================================
# 8. INVENTORY SETTINGS
# =========================================================

# Inventory below this threshold is considered a shortage
MIN_INVENTORY_DAYS = 7

# Normal inventory level used when generating sample data
TARGET_INVENTORY_DAYS = 14

# Inventory above this threshold is considered excess
MAX_INVENTORY_DAYS = 21

# Ratios of store-product pairs intentionally assigned to each status
SHORTAGE_RATIO = 0.20
EXCESS_RATIO = 0.20


# =========================================================
# 9. DISTANCE AND TRANSPORT COST
# =========================================================

# Earth radius used by the Haversine distance formula
EARTH_RADIUS_KM = 6371.0

# Routing method: "osrm" or "haversine"
ROUTING_METHOD = "osrm"

# Public OSRM Table API used for development
OSRM_TABLE_URL = (
    "https://router.project-osrm.org/"
    "table/v1/driving"
)

OSRM_REQUEST_TIMEOUT_SECONDS = 60

# Used to estimate duration if OSRM is unavailable
FALLBACK_AVERAGE_SPEED_KPH = 50.0

# Cost of transporting one product unit for one kilometer
TRANSPORT_COST_PER_KM_PER_UNIT = 100.0

# Minimum transportation cost for one product unit
MIN_TRANSPORT_COST_PER_UNIT = 500.0

# Maximum allowed transfer distance
MAX_TRANSFER_DISTANCE_KM = 1_600.0

# =========================================================
# 10. TRANSFER ROUTE POLICY
# =========================================================

INTRA_CITY_ROUTE = "intra_city"
INTER_CITY_ROUTE = "inter_city"

# Additional time for preparation, dispatch, and receiving
INTRA_CITY_HANDLING_TIME_MINUTES = 30
INTER_CITY_HANDLING_TIME_MINUTES = 6 * 60

# Maximum total lead time
INTRA_CITY_MAX_LEAD_TIME_MINUTES = 3 * 60
INTER_CITY_MAX_LEAD_TIME_MINUTES = 36 * 60


# =========================================================
# 11. RULE-BASED OPTIMIZATION SETTINGS
# =========================================================

# These weights must add up to 1.0
DISTANCE_WEIGHT = 0.40
EXCESS_WEIGHT = 0.30
SHORTAGE_WEIGHT = 0.30


# =========================================================
# 12. LINEAR PROGRAMMING SETTINGS
# =========================================================

# Penalty applied to each unit of unresolved shortage
UNMET_SHORTAGE_PENALTY_PER_UNIT = 1_000_000.0


# =========================================================
# 13. GENETIC ALGORITHM SETTINGS
# =========================================================

GA_POPULATION_SIZE = 100
GA_GENERATIONS = 50
GA_CROSSOVER_PROBABILITY = 0.70
GA_MUTATION_PROBABILITY = 0.10
GA_TOURNAMENT_SIZE = 3

# =========================================================
# 14. DEMAND FORECASTING SETTINGS
# =========================================================

FORECAST_BACKTEST_FOLDS = 5
BACKTEST_MIN_TRAINING_DAYS = 90

HISTORICAL_AVERAGE_METHOD = "historical_average"
MOVING_AVERAGE_METHOD = "moving_average"

SUPPORTED_FORECAST_METHODS = (
    HISTORICAL_AVERAGE_METHOD,
    MOVING_AVERAGE_METHOD,
)

DEFAULT_FORECAST_METHOD = (
    HISTORICAL_AVERAGE_METHOD
)

MIN_FORECAST_HORIZON_DAYS = 1
MAX_FORECAST_HORIZON_DAYS = 14

DEFAULT_FORECAST_HORIZON_DAYS = 7

SHORT_TERM_REPLENISHMENT_DAYS = (
    MIN_INVENTORY_DAYS
)

LONG_TERM_REPLENISHMENT_DAYS = (
    TARGET_INVENTORY_DAYS
)

FORECAST_HORIZON_PRESETS = (
    1,
    5,
    7,
    14,
)

MOVING_AVERAGE_WINDOW_DAYS = 7

FORECAST_EVALUATION_HORIZONS = (
    1,
    7,
    14,
)

FORECAST_LAG_DAYS = (
    1,
    7,
    14,
)

FORECAST_ROLLING_WINDOWS = (
    7,
    14,
    28,
)

MODEL_TRAIN_RATIO = 0.70
MODEL_VALIDATION_RATIO = 0.15
MODEL_TEST_RATIO = 0.15


# =========================================================
# 15. OPTIMIZATION PIPELINE SETTINGS
# =========================================================

GREEDY_OPTIMIZER = "greedy"
LINEAR_PROGRAMMING_OPTIMIZER = "linear_programming"
GENETIC_ALGORITHM_OPTIMIZER = "genetic_algorithm"

SUPPORTED_OPTIMIZERS = (
    GREEDY_OPTIMIZER,
    LINEAR_PROGRAMMING_OPTIMIZER,
    GENETIC_ALGORITHM_OPTIMIZER,
)

DEFAULT_OPTIMIZER = GREEDY_OPTIMIZER

# =========================================================
# 16. HELPER FUNCTIONS
# =========================================================

def ensure_project_directories() -> dict[str, Path]:
    """Create all required project directories if they do not exist."""

    directories = {
        "data": DATA_DIR,
        "results": RESULTS_DIR,
        "visualizations": VISUALIZATIONS_DIR,
        "logs": LOGS_DIR,
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def get_data_file_path(filename: str) -> Path:
    """Return the full path of a data file."""
    return DATA_DIR / filename


def get_results_file_path(filename: str) -> Path:
    """Return the full path of a result file."""
    return RESULTS_DIR / filename


def get_ga_config() -> dict[str, int | float]:
    """Return Genetic Algorithm settings as a dictionary."""

    return {
        "population_size": GA_POPULATION_SIZE,
        "generations": GA_GENERATIONS,
        "crossover_probability": GA_CROSSOVER_PROBABILITY,
        "mutation_probability": GA_MUTATION_PROBABILITY,
        "tournament_size": GA_TOURNAMENT_SIZE,
    }


def validate_config() -> None:
    """Validate the project configuration values."""

    if NUM_STORES <= 0:
        raise ValueError("NUM_STORES must be greater than zero.")

    if NUM_PRODUCTS <= 0:
        raise ValueError("NUM_PRODUCTS must be greater than zero.")

    if SALES_DAYS <= 0:
        raise ValueError("SALES_DAYS must be greater than zero.")

    if not (
        MIN_INVENTORY_DAYS
        < TARGET_INVENTORY_DAYS
        < MAX_INVENTORY_DAYS
    ):
        raise ValueError(
            "Inventory thresholds must satisfy: "
            "MIN_INVENTORY_DAYS "
            "< TARGET_INVENTORY_DAYS "
            "< MAX_INVENTORY_DAYS."
        )

    if not 0 <= SHORTAGE_RATIO <= 1:
        raise ValueError(
            "SHORTAGE_RATIO must be between 0 and 1."
        )

    if not 0 <= EXCESS_RATIO <= 1:
        raise ValueError(
            "EXCESS_RATIO must be between 0 and 1."
        )

    if SHORTAGE_RATIO + EXCESS_RATIO > 1:
        raise ValueError(
            "The sum of SHORTAGE_RATIO and EXCESS_RATIO "
            "must not exceed 1."
        )

    rule_weight_total = (
        DISTANCE_WEIGHT
        + EXCESS_WEIGHT
        + SHORTAGE_WEIGHT
    )

    if abs(rule_weight_total - 1.0) > 1e-9:
        raise ValueError(
            "Rule-based optimization weights must add up to 1."
        )

    if TRANSPORT_COST_PER_KM_PER_UNIT < 0:
        raise ValueError(
            "TRANSPORT_COST_PER_KM_PER_UNIT must not be negative."
        )

    if MIN_TRANSPORT_COST_PER_UNIT < 0:
        raise ValueError(
            "MIN_TRANSPORT_COST_PER_UNIT must not be negative."
        )

    if MAX_TRANSFER_DISTANCE_KM <= 0:
        raise ValueError(
            "MAX_TRANSFER_DISTANCE_KM must be greater than zero."
        )

    for city, city_config in STORE_CITIES.items():
        if city_config["count"] <= 0:
            raise ValueError(
                f"The store count for {city} must be greater than zero."
            )

    if ROUTING_METHOD not in {"osrm", "haversine"}:
        raise ValueError(
            "ROUTING_METHOD must be either "
            "'osrm' or 'haversine'."
        )

    if OSRM_REQUEST_TIMEOUT_SECONDS <= 0:
        raise ValueError(
            "OSRM_REQUEST_TIMEOUT_SECONDS "
            "must be greater than zero."
        )

    if FALLBACK_AVERAGE_SPEED_KPH <= 0:
        raise ValueError(
            "FALLBACK_AVERAGE_SPEED_KPH "
            "must be greater than zero."
        )
    
    if INTRA_CITY_HANDLING_TIME_MINUTES < 0:
        raise ValueError(
            "INTRA_CITY_HANDLING_TIME_MINUTES "
            "must not be negative."
        )

    if INTER_CITY_HANDLING_TIME_MINUTES < 0:
        raise ValueError(
            "INTER_CITY_HANDLING_TIME_MINUTES "
            "must not be negative."
        )

    if INTRA_CITY_MAX_LEAD_TIME_MINUTES <= 0:
        raise ValueError(
            "INTRA_CITY_MAX_LEAD_TIME_MINUTES "
            "must be greater than zero."
        )

    if INTER_CITY_MAX_LEAD_TIME_MINUTES <= 0:
        raise ValueError(
            "INTER_CITY_MAX_LEAD_TIME_MINUTES "
            "must be greater than zero."
        )

    if (
        INTRA_CITY_HANDLING_TIME_MINUTES
        >= INTRA_CITY_MAX_LEAD_TIME_MINUTES
    ):
        raise ValueError(
            "Intra-city handling time must be lower "
            "than the maximum intra-city lead time."
        )

    if (
        INTER_CITY_HANDLING_TIME_MINUTES
        >= INTER_CITY_MAX_LEAD_TIME_MINUTES
    ):
        raise ValueError(
            "Inter-city handling time must be lower "
            "than the maximum inter-city lead time."
        )

    if not SUPPORTED_FORECAST_METHODS:
        raise ValueError(
            "SUPPORTED_FORECAST_METHODS "
            "must not be empty."
        )

    if (
        DEFAULT_FORECAST_METHOD
        not in SUPPORTED_FORECAST_METHODS
    ):
        raise ValueError(
            "DEFAULT_FORECAST_METHOD must exist in "
            "SUPPORTED_FORECAST_METHODS."
        )

    if MIN_FORECAST_HORIZON_DAYS <= 0:
        raise ValueError(
            "MIN_FORECAST_HORIZON_DAYS "
            "must be greater than zero."
        )

    if (
        MAX_FORECAST_HORIZON_DAYS
        < MIN_FORECAST_HORIZON_DAYS
    ):
        raise ValueError(
            "MAX_FORECAST_HORIZON_DAYS must be "
            "greater than or equal to "
            "MIN_FORECAST_HORIZON_DAYS."
        )

    if not (
        MIN_FORECAST_HORIZON_DAYS
        <= DEFAULT_FORECAST_HORIZON_DAYS
        <= MAX_FORECAST_HORIZON_DAYS
    ):
        raise ValueError(
            "DEFAULT_FORECAST_HORIZON_DAYS must be "
            "within the allowed forecast range."
        )

    if not (
        MIN_FORECAST_HORIZON_DAYS
        <= SHORT_TERM_REPLENISHMENT_DAYS
        < LONG_TERM_REPLENISHMENT_DAYS
        == MAX_FORECAST_HORIZON_DAYS
    ):
        raise ValueError(
            "Replenishment horizons must satisfy: "
            "minimum forecast horizon <= short-term "
            "horizon < long-term horizon == maximum "
            "forecast horizon."
        )

    if not FORECAST_HORIZON_PRESETS:
        raise ValueError(
            "FORECAST_HORIZON_PRESETS "
            "must not be empty."
        )

    invalid_presets = [
        horizon
        for horizon in FORECAST_HORIZON_PRESETS
        if not (
            MIN_FORECAST_HORIZON_DAYS
            <= horizon
            <= MAX_FORECAST_HORIZON_DAYS
        )
    ]

    if invalid_presets:
        raise ValueError(
            "Forecast horizon presets are outside "
            f"the allowed range: {invalid_presets}"
        )

    if MOVING_AVERAGE_WINDOW_DAYS <= 0:
        raise ValueError(
            "MOVING_AVERAGE_WINDOW_DAYS "
            "must be greater than zero."
        )

    if not SUPPORTED_OPTIMIZERS:
        raise ValueError(
        "SUPPORTED_OPTIMIZERS must not be empty."
        )

    if DEFAULT_OPTIMIZER not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            "DEFAULT_OPTIMIZER must exist in "
            "SUPPORTED_OPTIMIZERS."
        )

    if not FORECAST_EVALUATION_HORIZONS:
        raise ValueError(
            "FORECAST_EVALUATION_HORIZONS "
            "must not be empty."
        )

    invalid_evaluation_horizons = [
        horizon
        for horizon in FORECAST_EVALUATION_HORIZONS
        if not (
            MIN_FORECAST_HORIZON_DAYS
            <= horizon
            <= MAX_FORECAST_HORIZON_DAYS
        )
    ]

    if invalid_evaluation_horizons:
        raise ValueError(
            "Forecast evaluation horizons are "
            "outside the allowed range: "
            f"{invalid_evaluation_horizons}"
        )
    if (
    not isinstance(FORECAST_BACKTEST_FOLDS, int)
    or FORECAST_BACKTEST_FOLDS <= 0
    ):
        raise ValueError(
            "FORECAST_BACKTEST_FOLDS must be "
            "a positive integer."
        )

    if (
        not isinstance(BACKTEST_MIN_TRAINING_DAYS, int)
        or BACKTEST_MIN_TRAINING_DAYS <= 0
    ):
        raise ValueError(
            "BACKTEST_MIN_TRAINING_DAYS must be "
            "a positive integer."
        )

    if (
        BACKTEST_MIN_TRAINING_DAYS
        < MOVING_AVERAGE_WINDOW_DAYS
    ):
        raise ValueError(
            "BACKTEST_MIN_TRAINING_DAYS must be "
            "greater than or equal to "
            "MOVING_AVERAGE_WINDOW_DAYS."
        )
    if not FORECAST_LAG_DAYS:
        raise ValueError(
            "FORECAST_LAG_DAYS must not be empty."
        )

    invalid_lag_days = [
        lag_day
        for lag_day in FORECAST_LAG_DAYS
        if (
            not isinstance(lag_day, int)
            or lag_day <= 0
        )
    ]

    if invalid_lag_days:
        raise ValueError(
            "Forecast lag days must be positive "
            f"integers: {invalid_lag_days}"
        )

    if (
        len(set(FORECAST_LAG_DAYS))
        != len(FORECAST_LAG_DAYS)
    ):
        raise ValueError(
            "FORECAST_LAG_DAYS must not contain "
            "duplicate values."
        )

    if not FORECAST_ROLLING_WINDOWS:
        raise ValueError(
            "FORECAST_ROLLING_WINDOWS "
            "must not be empty."
        )

    invalid_rolling_windows = [
        window
        for window in FORECAST_ROLLING_WINDOWS
        if (
            not isinstance(window, int)
            or window <= 0
        )
    ]

    if invalid_rolling_windows:
        raise ValueError(
            "Forecast rolling windows must be "
            "positive integers: "
            f"{invalid_rolling_windows}"
        )

    if (
        len(set(FORECAST_ROLLING_WINDOWS))
        != len(FORECAST_ROLLING_WINDOWS)
    ):
        raise ValueError(
            "FORECAST_ROLLING_WINDOWS must not "
            "contain duplicate values."
        )

    model_split_ratios = (
        MODEL_TRAIN_RATIO,
        MODEL_VALIDATION_RATIO,
        MODEL_TEST_RATIO,
    )

    invalid_model_split_ratios = [
        ratio
        for ratio in model_split_ratios
        if (
            isinstance(ratio, bool)
            or not isinstance(
                ratio,
                (int, float),
            )
            or ratio <= 0
            or ratio >= 1
        )
    ]

    if invalid_model_split_ratios:
        raise ValueError(
            "Model split ratios must be numbers "
            "between zero and one."
        )

    if abs(sum(model_split_ratios) - 1.0) > 1e-9:
        raise ValueError(
            "MODEL_TRAIN_RATIO, "
            "MODEL_VALIDATION_RATIO, and "
            "MODEL_TEST_RATIO must sum to 1.0."
        )


if __name__ == "__main__":
    validate_config()
    directories = ensure_project_directories()

    print("Configuration is valid.")
    print(f"Total number of stores: {NUM_STORES}")
    print(f"Project root: {PROJECT_ROOT}")

    for directory_name, directory_path in directories.items():
        print(f"{directory_name}: {directory_path}")
