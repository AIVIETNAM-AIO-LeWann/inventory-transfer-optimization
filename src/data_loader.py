"""Load and validate project data from CSV files."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    DISTANCE_MATRIX_FILE,
    DURATION_MATRIX_FILE,
    INVENTORY_FILE,
    PRODUCTS_FILE,
    SALES_FILE,
    STORES_FILE,
    TRANSPORT_COST_MATRIX_FILE,
)
from src.validator import (
    validate_inventory,
    validate_products,
    validate_route_matrix,
    validate_sales,
    validate_stores,
)


@dataclass
class ProjectData:
    """Store all validated project datasets."""

    stores: pd.DataFrame
    products: pd.DataFrame
    sales: pd.DataFrame
    inventory: pd.DataFrame
    distance_matrix: pd.DataFrame
    duration_matrix: pd.DataFrame
    transport_cost_matrix: pd.DataFrame


def read_csv_file(
    file_path: str | Path,
    dataset_name: str,
    **read_options: object,
) -> pd.DataFrame:
    """Read a CSV file and provide clear error messages."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file was not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{dataset_name} path is not a file: {path}"
        )

    try:
        data = pd.read_csv(
            path,
            **read_options,
        )
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"{dataset_name} file is empty: {path}"
        ) from error
    except pd.errors.ParserError as error:
        raise ValueError(
            f"{dataset_name} file cannot be parsed: {path}"
        ) from error

    return data


def load_stores(
    file_path: str | Path = STORES_FILE,
) -> pd.DataFrame:
    """Load and validate store data."""

    stores = read_csv_file(
        file_path=file_path,
        dataset_name="stores",
    )

    validate_stores(stores)

    return stores


def load_products(
    file_path: str | Path = PRODUCTS_FILE,
) -> pd.DataFrame:
    """Load and validate product data."""

    products = read_csv_file(
        file_path=file_path,
        dataset_name="products",
    )

    validate_products(products)

    return products


def load_sales(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    file_path: str | Path = SALES_FILE,
) -> pd.DataFrame:
    """Load and validate sales data."""

    sales = read_csv_file(
        file_path=file_path,
        dataset_name="sales",
        parse_dates=["date"],
    )

    validate_sales(
        sales=sales,
        stores=stores,
        products=products,
    )

    return sales


def load_inventory(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    file_path: str | Path = INVENTORY_FILE,
) -> pd.DataFrame:
    """Load and validate inventory data."""

    inventory = read_csv_file(
        file_path=file_path,
        dataset_name="inventory",
        parse_dates=["last_updated"],
    )

    validate_inventory(
        inventory=inventory,
        stores=stores,
        products=products,
    )

    return inventory


def load_route_matrix(
    stores: pd.DataFrame,
    file_path: str | Path,
    matrix_name: str,
) -> pd.DataFrame:
    """Load and validate a store-to-store route matrix."""

    matrix = read_csv_file(
        file_path=file_path,
        dataset_name=matrix_name,
        index_col="store_id",
    )

    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)

    validate_route_matrix(
        matrix=matrix,
        stores=stores,
        matrix_name=matrix_name,
    )

    return matrix


def load_all_data() -> ProjectData:
    """Load and validate all project datasets."""

    stores = load_stores()
    products = load_products()

    sales = load_sales(
        stores=stores,
        products=products,
    )

    inventory = load_inventory(
        stores=stores,
        products=products,
    )

    distance_matrix = load_route_matrix(
        stores=stores,
        file_path=DISTANCE_MATRIX_FILE,
        matrix_name="distance_matrix",
    )

    duration_matrix = load_route_matrix(
        stores=stores,
        file_path=DURATION_MATRIX_FILE,
        matrix_name="duration_matrix",
    )

    transport_cost_matrix = load_route_matrix(
        stores=stores,
        file_path=TRANSPORT_COST_MATRIX_FILE,
        matrix_name="transport_cost_matrix",
    )

    return ProjectData(
        stores=stores,
        products=products,
        sales=sales,
        inventory=inventory,
        distance_matrix=distance_matrix,
        duration_matrix=duration_matrix,
        transport_cost_matrix=transport_cost_matrix,
    )


def print_data_summary(
    project_data: ProjectData,
) -> None:
    """Print the shape of every loaded dataset."""

    print("All project data was loaded successfully.")
    print(f"Stores: {project_data.stores.shape}")
    print(f"Products: {project_data.products.shape}")
    print(f"Sales: {project_data.sales.shape}")
    print(f"Inventory: {project_data.inventory.shape}")
    print(
        "Distance matrix: "
        f"{project_data.distance_matrix.shape}"
    )
    print(
        "Duration matrix: "
        f"{project_data.duration_matrix.shape}"
    )
    print(
        "Transport cost matrix: "
        f"{project_data.transport_cost_matrix.shape}"
    )


def main() -> None:
    """Load all project data and print a summary."""

    project_data = load_all_data()
    print_data_summary(project_data)


if __name__ == "__main__":
    main()