import polars as pl

from MEDS_transforms import EXTRACT_CONFIG_YAML, MANDATORY_COLUMNS, MANDATORY_TYPES

# We set this equality explicitly here so linting does not remove an apparently "unused" import if we just
# rename with "as" during the import.
CONFIG_YAML = EXTRACT_CONFIG_YAML

# TODO(mmd): This should really somehow be pulled from MEDS.
MEDS_METADATA_MANDATORY_TYPES = {
    "code": pl.String,
    "description": pl.String,
    "parent_codes": pl.List(pl.String),
}

MEDS_DATA_MANDATORY_TYPES = {c: MANDATORY_TYPES[c] for c in MANDATORY_COLUMNS}
