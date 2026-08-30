"""Generate every sample dataset required by the project."""

from pathlib import Path

from src.config import (
    DISTANCE_MATRIX_FILE,
    DURATION_MATRIX_FILE,
    ROUTING_METHOD,
    TRANSPORT_COST_MATRIX_FILE,
    ensure_project_directories,
    validate_config,
)
from src.data_generator.distance_calculator import (
    generate_route_matrices,
    generate_transport_cost_matrix,
    save_matrix,
)
from src.data_generator.inventory_generator import (
    generate_inventory,
    save_inventory,
)
from src.data_generator.product_generator import (
    generate_products,
    save_products,
)
from src.data_generator.sales_generator import (
    generate_sales,
    save_sales,
)
from src.data_generator.store_generator import (
    generate_stores,
    save_stores,
)


def generate_all_data(
    routing_method: str = ROUTING_METHOD,
) -> tuple[dict[str, Path], str]:
    """
    Generate, validate, and save all sample project data.

    Data is generated in memory before files are saved so every
    dependent generator receives the exact output of the previous one.

    Args:
        routing_method:
            Routing method used to generate distance and duration
            matrices. Supported values are "osrm" and "haversine".

    Returns:
        A dictionary of generated file paths and the routing source used.
    """

    validate_config()
    ensure_project_directories()

    stores = generate_stores()
    products = generate_products()

    sales = generate_sales(
        stores=stores,
        products=products,
    )

    inventory = generate_inventory(
        sales=sales,
    )

    (
        distance_matrix,
        duration_matrix,
        routing_source,
    ) = generate_route_matrices(
        stores=stores,
        routing_method=routing_method,
    )

    transport_cost_matrix = generate_transport_cost_matrix(
        distance_matrix=distance_matrix,
    )

    output_paths = {
        "stores": save_stores(stores),
        "products": save_products(products),
        "sales": save_sales(sales),
        "inventory": save_inventory(inventory),
        "distance_matrix": save_matrix(
            distance_matrix,
            DISTANCE_MATRIX_FILE,
        ),
        "duration_matrix": save_matrix(
            duration_matrix,
            DURATION_MATRIX_FILE,
        ),
        "transport_cost_matrix": save_matrix(
            transport_cost_matrix,
            TRANSPORT_COST_MATRIX_FILE,
        ),
    }

    return output_paths, routing_source


def print_generation_summary(
    output_paths: dict[str, Path],
    routing_source: str,
) -> None:
    """Print a summary of generated files."""

    print("All sample data was generated successfully.")
    print(f"Routing source: {routing_source}")

    for dataset_name, output_path in output_paths.items():
        print(f"{dataset_name}: {output_path}")


def main() -> None:
    """Run the complete sample-data generation pipeline."""

    output_paths, routing_source = generate_all_data()

    print_generation_summary(
        output_paths=output_paths,
        routing_source=routing_source,
    )


if __name__ == "__main__":
    main()
