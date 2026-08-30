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

INVENTORY_ANALYSIS_FILE = RESULTS_DIR / "inventory_analysis.csv"
BEST_TRANSFER_PLAN_FILE = RESULTS_DIR / "best_transfer_plan.csv"
ALGORITHM_COMPARISON_FILE = RESULTS_DIR / "algorithm_comparison.csv"


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
SALES_DAYS = 90

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
UNMET_SHORTAGE_PENALTY_PER_UNIT = 100_000.0


# =========================================================
# 13. GENETIC ALGORITHM SETTINGS
# =========================================================

GA_POPULATION_SIZE = 100
GA_GENERATIONS = 50
GA_CROSSOVER_PROBABILITY = 0.70
GA_MUTATION_PROBABILITY = 0.10
GA_TOURNAMENT_SIZE = 3


# =========================================================
# 14. HELPER FUNCTIONS
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


if __name__ == "__main__":
    validate_config()
    directories = ensure_project_directories()

    print("Configuration is valid.")
    print(f"Total number of stores: {NUM_STORES}")
    print(f"Project root: {PROJECT_ROOT}")

    for directory_name, directory_path in directories.items():
        print(f"{directory_name}: {directory_path}")
