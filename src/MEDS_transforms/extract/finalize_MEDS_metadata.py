#!/usr/bin/env python
"""Utilities for finalizing the metadata files for extracted MEDS datasets."""

import json
from pathlib import Path

import hydra
import jsonschema
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from meds import __version__ as MEDS_VERSION
from meds import (
    code_metadata_schema,
    dataset_metadata_schema,
    held_out_split,
    patient_split_schema,
    train_split,
    tuning_split,
)
from omegaconf import DictConfig

from MEDS_transforms.extract import CONFIG_YAML, MEDS_METADATA_MANDATORY_TYPES
from MEDS_transforms.utils import stage_init


def get_and_validate_code_metadata_schema(code_metadata: pl.DataFrame, do_retype: bool = True) -> pa.Table:
    """Validates the schema of the code metadata DataFrame.

    Args:
        code_metadata: The code metadata DataFrame to validate.

    Returns:
        pa.Table: The validated code metadata DataFrame, with columns re-typed as needed.

    Raises:
        ValueError: if do_retype is False and the code metadata DataFrame is not schema compliant.

    Examples:
        >>> df = pl.DataFrame({})
        >>> get_and_validate_code_metadata_schema(df, do_retype=False) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: Code metadata DataFrame must have a 'code' column of type String.
                    Code metadata DataFrame must have a 'description' column of type String.
                    Code metadata DataFrame must have a 'parent_codes' column of type List(String).
        >>> get_and_validate_code_metadata_schema(df)
        pyarrow.Table
        code: string
        description: string
        parent_codes: list<item: string>
          child 0, item: string
        ----
        code: [[null]]
        description: [[null]]
        parent_codes: [[null]]
        >>> df = pl.DataFrame({"code": ["A"], "description": [1], "parent_codes": [3.2], "foo": [34.2]})
        >>> get_and_validate_code_metadata_schema(df, do_retype=False) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: Code metadata 'description' column must be of type String. Got Int64.
                    Code metadata 'parent_codes' column must be of type List(String). Got Float64.
        >>> get_and_validate_code_metadata_schema(df)
        pyarrow.Table
        code: string
        description: string
        parent_codes: list<item: string>
          child 0, item: string
        foo: double
        ----
        code: [["A"]]
        description: [["1"]]
        parent_codes: [[["3.2"]]]
        foo: [[34.2]]
    """

    schema = code_metadata.schema
    errors = []
    for col, dtype in MEDS_METADATA_MANDATORY_TYPES.items():
        if col in schema and schema[col] != dtype:
            if do_retype:
                code_metadata = code_metadata.with_columns(pl.col(col).cast(dtype, strict=False))
            else:
                errors.append(f"Code metadata '{col}' column must be of type {dtype}. Got {schema[col]}.")
        elif col not in schema:
            if do_retype:
                code_metadata = code_metadata.with_columns(pl.lit(None, dtype=dtype).alias(col))
            else:
                errors.append(f"Code metadata DataFrame must have a '{col}' column of type {dtype}.")

    if errors:
        raise ValueError("\n".join(errors))

    additional_cols = [col for col in schema if col not in MEDS_METADATA_MANDATORY_TYPES]

    if additional_cols:
        extra_schema = code_metadata.head(1).select(additional_cols).to_arrow().schema
        code_metadata_properties = list(zip(extra_schema.names, extra_schema.types))
        code_metadata = code_metadata.select(*MEDS_METADATA_MANDATORY_TYPES.keys(), *additional_cols)
    else:
        code_metadata = code_metadata.select(*MEDS_METADATA_MANDATORY_TYPES.keys())
        code_metadata_properties = []

    validated_schema = code_metadata_schema(code_metadata_properties)
    return code_metadata.to_arrow().cast(validated_schema)


@hydra.main(version_base=None, config_path=str(CONFIG_YAML.parent), config_name=CONFIG_YAML.stem)
def main(cfg: DictConfig):
    """Writes out schema compliant MEDS metadata files for the extracted dataset.

    In particular, this script ensures that
    (1) a compliant `metadata/codes.parquet` file exists that has the mandatory columns
      - `code` (string)
      - `description` (string)
      - `parent_codes` (list of strings)
    (2) a `metadata/dataset.json` file exists that has the keys
      - `dataset_name` (string)
      - `dataset_version` (string)
      - `etl_name` (string)
      - `etl_version` (string)
      - `meds_version` (string)
    (3) a `metadata/patient_splits.parquet` file exists that has the mandatory columns
      - `patient_id` (Int64)
      - `split` (string)

    This stage *_should almost always be the last metadata stage in an extraction pipeline._*

    All arguments are specified through the command line into the `cfg` object through Hydra.

    The `cfg.stage_cfg` object is a special key that is imputed by OmegaConf to contain the stage-specific
    configuration arguments based on the global, pipeline-level configuration file. It cannot be overwritten
    directly on the command line, but can be overwritten implicitly by overwriting components of the
    `stage_configs.extract_code_metadata` key.

    Args:
        stage_configs.finalize_MEDS_data.do_retype: Whether the script should throw an error or attempt to
            cast columns to the correct type if they are not already of the correct type. Defaults to `True`.
            May not work properly with other default aspects of the MEDS_Extract pipeline if set to `False`.
        etl_metadata.dataset_name: The name of the dataset being extracted.
        etl_metadata.dataset_version: The version of the dataset being extracted.
    """

    if cfg.worker != 0:
        logger.info("Non-zero worker found in reduce-only stage. Exiting")
        return

    _, _, input_metadata_dir = stage_init(cfg)
    output_metadata_dir = Path(cfg.stage_cfg.reducer_output_dir)

    output_code_metadata_fp = output_metadata_dir / "codes.parquet"
    dataset_metadata_fp = output_metadata_dir / "dataset.json"
    patient_splits_fp = output_metadata_dir / "patient_splits.parquet"

    for out_fp in [output_code_metadata_fp, dataset_metadata_fp, patient_splits_fp]:
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        if out_fp.exists() and cfg.do_overwrite:
            out_fp.unlink()
        elif out_fp.exists() and not cfg.do_overwrite:
            raise FileExistsError(f"Output file already exists at {str(out_fp.resolve())}")

    # Code metadata validation
    logger.info("Validating code metadata")
    input_code_metadata_fp = input_metadata_dir / "codes.parquet"
    if input_code_metadata_fp.exists():
        logger.info(f"Reading code metadata from {str(input_code_metadata_fp.resolve())}")
        code_metadata = pl.read_parquet(input_code_metadata_fp, use_pyarrow=True)
        final_metadata_tbl = get_and_validate_code_metadata_schema(
            code_metadata, do_retype=cfg.stage_cfg.do_retype
        )
    else:
        logger.info(f"No code metadata found at {str(input_code_metadata_fp)}. Making empty metadata file.")
        codes_schema = code_metadata_schema()
        final_metadata_tbl = pa.Table.from_pylist([], schema=codes_schema)

    logger.info(f"Writing finalized metadata df to {str(output_code_metadata_fp.resolve())}")
    pq.write_table(final_metadata_tbl, output_code_metadata_fp)

    # Dataset metadata creation
    logger.info("Creating dataset metadata")

    dataset_metadata = {
        "dataset_name": cfg.etl_metadata.dataset_name,
        "dataset_version": str(cfg.etl_metadata.dataset_version),
        "etl_name": cfg.etl_metadata.package_name,
        "etl_version": str(cfg.etl_metadata.package_version),
        "meds_version": MEDS_VERSION,
    }
    jsonschema.validate(instance=dataset_metadata, schema=dataset_metadata_schema)

    logger.info(f"Writing finalized dataset metadata to {str(dataset_metadata_fp.resolve())}")
    dataset_metadata_fp.write_text(json.dumps(dataset_metadata))

    # Split creation
    shards_map_fp = Path(cfg.shards_map_fp)
    logger.info("Creating patient splits from {str(shards_map_fp.resolve())}")
    shards_map = json.loads(shards_map_fp.read_text())
    patient_splits = []
    seen_splits = {train_split: 0, tuning_split: 0, held_out_split: 0}
    for shard, patient_ids in shards_map.items():
        split = "/".join(shard.split("/")[:-1])

        if split not in seen_splits:
            seen_splits[split] = 0
        seen_splits[split] += len(patient_ids)

        patient_splits.extend([{"patient_id": pid, "split": split} for pid in patient_ids])

    for split, cnt in seen_splits.items():
        if cnt:
            logger.info(f"Split {split} has {cnt} patients")
        else:
            logger.warning(f"Split {split} not found in shards map")

    patient_splits_tbl = pa.Table.from_pylist(patient_splits, schema=patient_split_schema)
    logger.info(f"Writing finalized patient splits to {str(patient_splits_fp.resolve())}")
    pq.write_table(patient_splits_tbl, patient_splits_fp)


if __name__ == "__main__":
    main()
