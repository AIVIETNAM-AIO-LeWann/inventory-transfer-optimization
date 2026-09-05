"""Load and validate a complete project dataset from ZIP bytes."""

from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo

import pandas as pd

from src.dashboard.constants import (
    MAX_UNCOMPRESSED_UPLOAD_SIZE_MB,
    MAX_UPLOAD_SIZE_MB,
    REQUIRED_UPLOAD_FILENAMES,
)
from src.data_loader import ProjectData
from src.validator import validate_all_data


DATASET_FILE_NAMES = {
    "stores": "stores.csv",
    "products": "products.csv",
    "sales": "sales_data.csv",
    "inventory": "inventory_data.csv",
    "distance_matrix": "distance_matrix.csv",
    "duration_matrix": "duration_matrix.csv",
    "transport_cost_matrix": (
        "transport_cost_matrix.csv"
    ),
}


def validate_zip_bytes(
    zip_bytes: bytes,
    max_upload_size_mb: int = MAX_UPLOAD_SIZE_MB,
) -> None:
    """Validate the type and compressed size of an upload."""

    if not isinstance(zip_bytes, bytes):
        raise TypeError(
            "zip_bytes must be bytes."
        )

    if not zip_bytes:
        raise ValueError(
            "The uploaded ZIP file is empty."
        )

    if (
        isinstance(max_upload_size_mb, bool)
        or not isinstance(max_upload_size_mb, int)
        or max_upload_size_mb <= 0
    ):
        raise ValueError(
            "max_upload_size_mb must be a positive integer."
        )

    maximum_size_bytes = (
        max_upload_size_mb * 1024 * 1024
    )

    if len(zip_bytes) > maximum_size_bytes:
        raise ValueError(
            "The uploaded ZIP file exceeds the "
            f"{max_upload_size_mb} MB limit."
        )


def normalize_zip_member_name(
    member_name: str,
) -> str:
    """Return a safe root-level filename from a ZIP member."""

    if not isinstance(member_name, str):
        raise TypeError(
            "ZIP member names must be strings."
        )

    normalized_name = member_name.replace(
        chr(92),
        "/",
    )
    member_path = PurePosixPath(normalized_name)

    if (
        not normalized_name
        or member_path.is_absolute()
        or len(member_path.parts) != 1
        or ".." in member_path.parts
    ):
        raise ValueError(
            "Every CSV must be stored at the root "
            "of the ZIP file."
        )

    return member_path.name


def validate_archive_members(
    members: list[ZipInfo],
    max_uncompressed_size_mb: int = (
        MAX_UNCOMPRESSED_UPLOAD_SIZE_MB
    ),
) -> dict[str, ZipInfo]:
    """Validate archive contents and return required members."""

    if (
        isinstance(max_uncompressed_size_mb, bool)
        or not isinstance(
            max_uncompressed_size_mb,
            int,
        )
        or max_uncompressed_size_mb <= 0
    ):
        raise ValueError(
            "max_uncompressed_size_mb must be a "
            "positive integer."
        )

    file_members = [
        member
        for member in members
        if not member.is_dir()
    ]

    if not file_members:
        raise ValueError(
            "The uploaded ZIP file contains no files."
        )

    normalized_members: dict[str, ZipInfo] = {}

    for member in file_members:
        normalized_name = normalize_zip_member_name(
            member.filename
        )

        if member.flag_bits & 0x1:
            raise ValueError(
                "Encrypted ZIP files are not supported."
            )

        if normalized_name in normalized_members:
            raise ValueError(
                "The uploaded ZIP contains duplicate "
                f"files named {normalized_name}."
            )

        normalized_members[normalized_name] = member

    expected_names = set(
        REQUIRED_UPLOAD_FILENAMES
    )
    provided_names = set(
        normalized_members
    )
    missing_names = sorted(
        expected_names - provided_names
    )
    unexpected_names = sorted(
        provided_names - expected_names
    )

    if missing_names:
        raise ValueError(
            "The uploaded ZIP is missing required files: "
            f"{missing_names}"
        )

    if unexpected_names:
        raise ValueError(
            "The uploaded ZIP contains unexpected files: "
            f"{unexpected_names}"
        )

    total_uncompressed_size = sum(
        member.file_size
        for member in file_members
    )
    maximum_size_bytes = (
        max_uncompressed_size_mb
        * 1024
        * 1024
    )

    if total_uncompressed_size > maximum_size_bytes:
        raise ValueError(
            "The uncompressed dataset exceeds the "
            f"{max_uncompressed_size_mb} MB limit."
        )

    return {
        name: normalized_members[name]
        for name in REQUIRED_UPLOAD_FILENAMES
    }


def read_csv_member(
    archive: ZipFile,
    member: ZipInfo,
    dataset_name: str,
    **read_options: object,
) -> pd.DataFrame:
    """Read one CSV member and provide a clear error message."""

    try:
        with archive.open(member) as csv_file:
            return pd.read_csv(
                csv_file,
                **read_options,
            )
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"{dataset_name} CSV is empty."
        ) from error
    except (
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        raise ValueError(
            f"{dataset_name} CSV cannot be parsed."
        ) from error


def load_project_data_from_zip(
    zip_bytes: bytes,
) -> ProjectData:
    """Load, normalize, and validate seven CSVs from a ZIP file."""

    validate_zip_bytes(zip_bytes)

    try:
        with ZipFile(BytesIO(zip_bytes)) as archive:
            members = validate_archive_members(
                archive.infolist()
            )

            stores = read_csv_member(
                archive=archive,
                member=members[
                    DATASET_FILE_NAMES["stores"]
                ],
                dataset_name="stores",
            )
            products = read_csv_member(
                archive=archive,
                member=members[
                    DATASET_FILE_NAMES["products"]
                ],
                dataset_name="products",
            )
            sales = read_csv_member(
                archive=archive,
                member=members[
                    DATASET_FILE_NAMES["sales"]
                ],
                dataset_name="sales",
                parse_dates=["date"],
            )
            inventory = read_csv_member(
                archive=archive,
                member=members[
                    DATASET_FILE_NAMES["inventory"]
                ],
                dataset_name="inventory",
                parse_dates=["last_updated"],
            )
            distance_matrix = read_csv_member(
                archive=archive,
                member=members[
                    DATASET_FILE_NAMES[
                        "distance_matrix"
                    ]
                ],
                dataset_name="distance_matrix",
                index_col="store_id",
            )
            duration_matrix = read_csv_member(
                archive=archive,
                member=members[
                    DATASET_FILE_NAMES[
                        "duration_matrix"
                    ]
                ],
                dataset_name="duration_matrix",
                index_col="store_id",
            )
            transport_cost_matrix = read_csv_member(
                archive=archive,
                member=members[
                    DATASET_FILE_NAMES[
                        "transport_cost_matrix"
                    ]
                ],
                dataset_name=(
                    "transport_cost_matrix"
                ),
                index_col="store_id",
            )
    except BadZipFile as error:
        raise ValueError(
            "The uploaded file is not a valid ZIP archive."
        ) from error

    for matrix in (
        distance_matrix,
        duration_matrix,
        transport_cost_matrix,
    ):
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)

    validate_all_data(
        stores=stores,
        products=products,
        sales=sales,
        inventory=inventory,
        distance_matrix=distance_matrix,
        duration_matrix=duration_matrix,
        transport_cost_matrix=(
            transport_cost_matrix
        ),
    )

    return ProjectData(
        stores=stores,
        products=products,
        sales=sales,
        inventory=inventory,
        distance_matrix=distance_matrix,
        duration_matrix=duration_matrix,
        transport_cost_matrix=(
            transport_cost_matrix
        ),
    )
