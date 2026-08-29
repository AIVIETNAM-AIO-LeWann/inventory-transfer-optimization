"""Generate sample product data for the inventory optimization system."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    MAX_PRODUCT_COST,
    MAX_PROFIT_MARGIN,
    MIN_PRODUCT_COST,
    MIN_PROFIT_MARGIN,
    NUM_PRODUCTS,
    PRODUCTS_FILE,
    PRODUCT_CATEGORIES,
    RANDOM_SEED,
    validate_config,
)


CATEGORY_PRODUCTS = {
    "Electronics": (
        "Wireless Earbuds",
        "Bluetooth Speaker",
        "Gaming Mouse",
        "Mechanical Keyboard",
        "Portable SSD",
        "Power Bank",
        "Smart Watch",
        "Webcam",
        "USB-C Hub",
        "Wi-Fi Router",
    ),
    "Clothing": (
        "Oxford Shirt",
        "Polo Shirt",
        "Crewneck T-Shirt",
        "Denim Jeans",
        "Casual Trousers",
        "Lightweight Jacket",
        "Summer Dress",
        "Knit Cardigan",
        "Running Shoes",
        "Canvas Sneakers",
    ),
    "Home Goods": (
        "Frying Pan",
        "Stainless Steel Pot",
        "Dinnerware Set",
        "Kitchen Knife",
        "Cutting Board",
        "Electric Kettle",
        "Table Lamp",
        "Throw Blanket",
        "Bed Sheet Set",
        "Storage Basket",
    ),
    "Food": (
        "Jasmine Rice",
        "Instant Noodles",
        "Fish Sauce",
        "Soy Sauce",
        "Vietnamese Coffee",
        "Green Tea",
        "Rice Crackers",
        "Dried Mango",
        "Coconut Water",
        "Rice Paper",
    ),
    "Beauty": (
        "Facial Cleanser",
        "Vitamin C Serum",
        "Moisturizing Cream",
        "Sunscreen SPF 50",
        "Sheet Mask Set",
        "Shampoo",
        "Conditioner",
        "Body Wash",
        "Body Lotion",
        "Lipstick",
    ),
}


CATEGORY_BRANDS = {
    "Electronics": (
        "NovaTech",
        "Vertex",
        "Lumina",
        "Nexa",
        "Orbit",
    ),
    "Clothing": (
        "Northline",
        "Urban Thread",
        "Everyday",
        "Blue River",
        "Motion",
    ),
    "Home Goods": (
        "HomeNest",
        "Living Plus",
        "Bright Home",
        "Daily Living",
        "Cozy Space",
    ),
    "Food": (
        "Mekong Foods",
        "Golden Harvest",
        "Fresh Day",
        "Green Farm",
        "Viet Taste",
    ),
    "Beauty": (
        "Pure Glow",
        "Bloom Care",
        "Lumi Beauty",
        "Natural Touch",
        "Daily Skin",
    ),
}


# Cost ranges represent product acquisition costs in VND.
CATEGORY_COST_RANGES = {
    "Electronics": (500_000, 2_000_000),
    "Clothing": (100_000, 800_000),
    "Home Goods": (150_000, 1_200_000),
    "Food": (20_000, 300_000),
    "Beauty": (100_000, 1_000_000),
}


PRODUCT_COLUMNS = (
    "product_id",
    "product_name",
    "category",
    "cost",
    "price",
)


def calculate_category_counts( num_products: int, ) -> dict[str, int]:
    """Distribute products approximately equally across categories."""

    if num_products <= 0:
        raise ValueError(
            "num_products must be greater than zero."
        )

    categories = list(PRODUCT_CATEGORIES)

    if not categories:
        raise ValueError(
            "PRODUCT_CATEGORIES must not be empty."
        )

    products_per_category, remaining_products = divmod( num_products, len(categories), )

    category_counts = { category: products_per_category for category in categories }

    for category in categories[:remaining_products]:
        category_counts[category] += 1

    return category_counts


def validate_product_generation_settings( num_products: int, ) -> None:
    """Validate settings required by the product generator."""

    category_counts = calculate_category_counts(num_products)

    required_categories = set(PRODUCT_CATEGORIES)

    missing_product_lists = (
        required_categories - set(CATEGORY_PRODUCTS)
    )

    missing_brand_lists = (
        required_categories - set(CATEGORY_BRANDS)
    )

    missing_cost_ranges = (
        required_categories - set(CATEGORY_COST_RANGES)
    )

    if missing_product_lists:
        raise ValueError(
            "Missing product names for categories: "
            f"{sorted(missing_product_lists)}"
        )

    if missing_brand_lists:
        raise ValueError(
            "Missing brands for categories: "
            f"{sorted(missing_brand_lists)}"
        )

    if missing_cost_ranges:
        raise ValueError(
            "Missing cost ranges for categories: "
            f"{sorted(missing_cost_ranges)}"
        )

    for category in PRODUCT_CATEGORIES:
        product_names = CATEGORY_PRODUCTS[category]
        brands = CATEGORY_BRANDS[category]
        minimum_cost, maximum_cost = CATEGORY_COST_RANGES[category]

        if not product_names:
            raise ValueError(
                f"{category} must contain at least one product name."
            )

        if not brands:
            raise ValueError(
                f"{category} must contain at least one brand."
            )

        if minimum_cost < MIN_PRODUCT_COST:
            raise ValueError(
                f"The minimum cost for {category} is below "
                "MIN_PRODUCT_COST."
            )

        if maximum_cost > MAX_PRODUCT_COST:
            raise ValueError(
                f"The maximum cost for {category} exceeds "
                "MAX_PRODUCT_COST."
            )

        if minimum_cost >= maximum_cost:
            raise ValueError(
                f"Invalid cost range for {category}."
            )

        possible_names = len(product_names) * len(brands)
        required_count = category_counts[category]

        if required_count > possible_names:
            raise ValueError(
                f"{category} requires {required_count} products, "
                f"but only {possible_names} unique combinations "
                "are available."
            )


def generate_products( num_products: int = NUM_PRODUCTS, seed: int = RANDOM_SEED, ) -> pd.DataFrame:
    """
    Generate sample product data.

    Args:
        num_products:
            Total number of products to generate.
        seed:
            Random seed used for reproducibility.

    Returns:
        A DataFrame containing generated product information.
    """

    validate_config()
    validate_product_generation_settings(num_products)

    rng = np.random.default_rng(seed)
    category_counts = calculate_category_counts(num_products)

    product_records: list[dict[str, object]] = []

    for category, product_count in category_counts.items():
        product_names = CATEGORY_PRODUCTS[category]
        brands = CATEGORY_BRANDS[category]
        minimum_cost, maximum_cost = CATEGORY_COST_RANGES[category]

        possible_full_names = [
            f"{brand} {product_name}"
            for brand in brands
            for product_name in product_names
        ]

        selected_indexes = rng.choice(
            len(possible_full_names),
            size=product_count,
            replace=False,
        )

        for selected_index in selected_indexes:
            product_name = possible_full_names[int(selected_index)]

            # Generate costs in multiples of 1,000 VND.
            cost = int(
                rng.integers(
                    minimum_cost // 1_000,
                    maximum_cost // 1_000 + 1,
                )
            ) * 1_000

            profit_margin = rng.uniform(
                MIN_PROFIT_MARGIN,
                MAX_PROFIT_MARGIN,
            )

            # Round the selling price up to the nearest 1,000 VND.
            price = int(
                np.ceil(
                    cost * (1 + profit_margin) / 1_000
                )
            ) * 1_000

            product_records.append(
                {
                    "product_name": product_name,
                    "category": category,
                    "cost": cost,
                    "price": price,
                }
            )

    # Mix categories while keeping the result reproducible.
    rng.shuffle(product_records)

    for product_number, product_record in enumerate(
        product_records,
        start=1,
    ):
        product_record["product_id"] = f"P{product_number:03d}"

    products = pd.DataFrame(
        product_records,
        columns=PRODUCT_COLUMNS,
    )

    validate_generated_products(
        products,
        expected_count=num_products,
    )

    return products


def validate_generated_products( products: pd.DataFrame, expected_count: int = NUM_PRODUCTS, ) -> None:
    """Validate a generated product DataFrame."""

    missing_columns = set(PRODUCT_COLUMNS) - set(products.columns)

    if missing_columns:
        raise ValueError(
            "Generated product data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(products) != expected_count:
        raise ValueError(
            f"Expected {expected_count} products, "
            f"but generated {len(products)}."
        )

    if products["product_id"].duplicated().any():
        raise ValueError(
            "Generated product IDs must be unique."
        )

    if products["product_name"].duplicated().any():
        raise ValueError(
            "Generated product names must be unique."
        )

    invalid_categories = (
        set(products["category"]) - set(PRODUCT_CATEGORIES)
    )

    if invalid_categories:
        raise ValueError(
            "Generated products contain invalid categories: "
            f"{sorted(invalid_categories)}"
        )

    if (products["cost"] <= 0).any():
        raise ValueError(
            "Product costs must be greater than zero."
        )

    if (products["price"] <= products["cost"]).any():
        raise ValueError(
            "Product prices must be greater than product costs."
        )

    if products[list(PRODUCT_COLUMNS)].isna().any().any():
        raise ValueError(
            "Generated product data must not contain missing values."
        )


def save_products( products: pd.DataFrame, output_path: str | Path = PRODUCTS_FILE,) -> Path:
    """Validate and save product data to a CSV file."""

    validate_generated_products(
        products,
        expected_count=len(products),
    )

    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    products.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
    )

    return destination.resolve()


def main() -> None:
    """Generate, validate, and save sample product data."""

    products = generate_products()
    output_path = save_products(products)

    print(f"Generated {len(products)} products.")
    print(f"Saved product data to: {output_path}")
    print()
    print(products.to_string(index=False))


if __name__ == "__main__":
    main()