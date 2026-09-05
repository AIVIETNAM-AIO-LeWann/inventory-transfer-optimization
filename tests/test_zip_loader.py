"""Tests for loading project datasets from ZIP files."""

from io import BytesIO
from zipfile import ZipFile, ZipInfo

import pandas as pd
import pytest

from src.config import REQUIRED_DATA_FILES
from src.data_ingestion.zip_loader import (
    load_project_data_from_zip,
    normalize_zip_member_name,
    validate_archive_members,
    validate_zip_bytes,
)
from src.data_loader import ProjectData


def create_project_zip(
    omitted_name: str | None = None,
    extra_name: str | None = None,
    nested: bool = False,
) -> bytes:
    """Create ZIP bytes from the checked-in project CSVs."""

    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, mode="w") as archive:
        for file_path in REQUIRED_DATA_FILES:
            if file_path.name == omitted_name:
                continue

            archive_name = file_path.name

            if nested:
                archive_name = (
                    f"dataset/{archive_name}"
                )

            archive.writestr(
                archive_name,
                file_path.read_bytes(),
            )

        if extra_name is not None:
            archive.writestr(
                extra_name,
                b"extra",
            )

    return zip_buffer.getvalue()


def test_load_valid_project_zip() -> None:
    """A complete valid ZIP should create ProjectData."""

    project_data = load_project_data_from_zip(
        create_project_zip()
    )

    assert isinstance(
        project_data,
        ProjectData,
    )
    assert not project_data.stores.empty
    assert not project_data.products.empty
    assert not project_data.sales.empty
    assert not project_data.inventory.empty
    assert pd.api.types.is_datetime64_any_dtype(
        project_data.sales["date"]
    )
    assert pd.api.types.is_datetime64_any_dtype(
        project_data.inventory["last_updated"]
    )


@pytest.mark.parametrize(
    "zip_bytes",
    [
        b"",
        bytearray(b"not bytes"),
    ],
)
def test_validate_zip_bytes_rejects_invalid_input(
    zip_bytes,
) -> None:
    """Empty and non-bytes uploads should fail."""

    expected_exception = (
        ValueError
        if isinstance(zip_bytes, bytes)
        else TypeError
    )

    with pytest.raises(expected_exception):
        validate_zip_bytes(zip_bytes)


def test_validate_zip_bytes_rejects_large_upload() -> None:
    """Compressed uploads above the configured limit should fail."""

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):
        validate_zip_bytes(
            b"0" * (1024 * 1024 + 1),
            max_upload_size_mb=1,
        )


@pytest.mark.parametrize(
    "member_name",
    [
        "dataset/stores.csv",
        "../stores.csv",
        "/stores.csv",
        "dataset\\stores.csv",
    ],
)
def test_nested_zip_member_is_rejected(
    member_name: str,
) -> None:
    """CSV files must be stored at the ZIP root."""

    with pytest.raises(
        ValueError,
        match="root",
    ):
        normalize_zip_member_name(
            member_name
        )


def test_missing_required_file_is_rejected() -> None:
    """A ZIP missing one required CSV should fail."""

    with pytest.raises(
        ValueError,
        match="missing required files",
    ):
        load_project_data_from_zip(
            create_project_zip(
                omitted_name="stores.csv"
            )
        )


def test_unexpected_file_is_rejected() -> None:
    """A ZIP with unsupported files should fail."""

    with pytest.raises(
        ValueError,
        match="unexpected files",
    ):
        load_project_data_from_zip(
            create_project_zip(
                extra_name="notes.txt"
            )
        )


def test_nested_archive_is_rejected() -> None:
    """A dataset folder inside the ZIP should fail."""

    with pytest.raises(
        ValueError,
        match="root",
    ):
        load_project_data_from_zip(
            create_project_zip(nested=True)
        )


def test_invalid_zip_is_rejected() -> None:
    """Non-ZIP bytes should produce a clear error."""

    with pytest.raises(
        ValueError,
        match="valid ZIP",
    ):
        load_project_data_from_zip(
            b"not-a-zip-file"
        )


def test_duplicate_archive_member_is_rejected() -> None:
    """Duplicate filenames should be rejected."""

    members = [
        ZipInfo("stores.csv"),
        ZipInfo("stores.csv"),
    ]

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_archive_members(members)


def test_uncompressed_size_limit_is_enforced() -> None:
    """Declared uncompressed content above the limit should fail."""

    members = []

    for file_path in REQUIRED_DATA_FILES:
        member = ZipInfo(file_path.name)
        member.file_size = 200_000
        members.append(member)

    with pytest.raises(
        ValueError,
        match="uncompressed dataset",
    ):
        validate_archive_members(
            members=members,
            max_uncompressed_size_mb=1,
        )
