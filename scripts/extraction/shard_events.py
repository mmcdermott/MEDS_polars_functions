#!/usr/bin/env python

import copy
import gzip
import random
from collections.abc import Sequence
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import (
    get_shard_prefix,
    hydra_loguru_init,
    is_col_field,
    parse_col_field,
    write_lazyframe,
)

ROW_IDX_NAME = "__row_idx__"
META_KEYS = {"timestamp_format"}


def kwargs_strs(kwargs: dict) -> str:
    return "\n".join([f"  * {k}={v}" for k, v in kwargs.items()])


def scan_with_row_idx(fp: Path, columns: Sequence[str], **scan_kwargs) -> pl.LazyFrame:
    """Scans a file with a row index column added.

    Note that we don't put ``row_index_name=ROW_IDX_NAME`` in the kwargs because it is not well supported in
    polars currently, pending https://github.com/pola-rs/polars/issues/15730. Instead, we add it at the end,
    which seems to work.
    """

    kwargs = {**scan_kwargs}
    match "".join(fp.suffixes).lower():
        case ".csv.gz":
            if columns:
                kwargs["columns"] = columns

            logger.debug(
                f"Reading {str(fp.resolve())} as compressed CSV with kwargs:\n{kwargs_strs(kwargs)}."
            )
            logger.warning("Reading compressed CSV files may be slow and limit parallelizability.")
            with gzip.open(fp, mode="rb") as f:
                return pl.read_csv(f, **kwargs).with_row_index(ROW_IDX_NAME).lazy()
        case ".csv":
            logger.debug(f"Reading {str(fp.resolve())} as CSV with kwargs:\n{kwargs_strs(kwargs)}.")
            df = pl.scan_csv(fp, **kwargs)
        case ".parquet":
            if "infer_schema_length" in kwargs:
                infer_schema_length = kwargs.pop("infer_schema_length")
                logger.info(f"Ignoring infer_schema_length={infer_schema_length} for Parquet files.")

            logger.debug(f"Reading {str(fp.resolve())} as Parquet with kwargs:\n{kwargs_strs(kwargs)}.")
            df = pl.scan_parquet(fp, **kwargs)
        case _:
            raise ValueError(f"Unsupported file type: {fp.suffix}")

    if columns:
        columns = [*columns]
        logger.debug(f"Selecting columns: {columns}")
        df = df.select(columns)

    df = df.with_row_index(ROW_IDX_NAME)

    logger.debug(f"Returning df with columns: {', '.join(df.columns)}")
    return df


def retrieve_columns(event_conversion_cfg: DictConfig) -> dict[str, list[str]]:
    """Extracts and organizes column names from configuration for a list of files.

    This function processes each file specified in the 'files' list, reading the
    event conversion configurations that are specific to each file based on its
    stem (filename without the extension). It compiles a list of column names
    needed for each file from the configuration, which includes both general
    columns like row index and patient ID, as well as specific columns defined
    for medical events and timestamps formatted in a special 'col(column_name)' syntax.

    Args:
        event_conversion_cfg (DictConfig): A dictionary configuration where
            each key is a filename stem and each value is a dictionary containing
            configuration details for different codes or events, specifying
            which columns to retrieve for each file.

    Returns:
        dict[Path, list[str]]: A dictionary mapping each file path to a list
        of unique column names necessary for processing the file.
        The list of columns includes generic columns and those specified in the 'event_conversion_cfg'.

    Examples:
        >>> cfg = DictConfig({
        ...     "patient_id_col": "patient_id_global",
        ...     "hosp/patients": {
        ...         "eye_color": {
        ...             "code": ["EYE_COLOR", "col(eye_color)"], "timestamp": None, "mod": "mod_col"
        ...         },
        ...         "height": {
        ...             "code": "HEIGHT", "timestamp": None, "numerical_value": "height"
        ...         }
        ...     },
        ...     "icu/chartevents": {
        ...         "patient_id_col": "patient_id_icu",
        ...         "heart_rate": {
        ...             "code": "HEART_RATE", "timestamp": "charttime", "numerical_value": "HR"
        ...         },
        ...         "lab": {
        ...             "code": ["col(itemid)", "col(valueuom)"],
        ...             "timestamp": "charttime",
        ...             "numerical_value": "valuenum",
        ...             "text_value": "value",
        ...             "mod": "mod_lab",
        ...         }
        ...     },
        ...     "icu/meds": {
        ...         "med": {"code": "col(medication)", "timestamp": "medtime"}
        ...     }
        ... })
        >>> retrieve_columns(cfg) # doctest: +NORMALIZE_WHITESPACE
        {'hosp/patients': ['eye_color', 'height', 'mod_col', 'patient_id_global'],
         'icu/chartevents': ['HR', 'charttime', 'itemid', 'mod_lab', 'patient_id_icu', 'value', 'valuenum',
                             'valueuom'],
         'icu/meds': ['medication', 'medtime', 'patient_id_global']}
        >>> cfg = DictConfig({
        ...     "subjects": {
        ...         "patient_id_col": "MRN",
        ...         "eye_color": {"code": ["col(eye_color)"], "timestamp": None},
        ...     },
        ...     "labs": {"lab": {"code": "col(labtest)", "timestamp": "charttime"}},
        ... })
        >>> retrieve_columns(cfg)
        {'subjects': ['MRN', 'eye_color'], 'labs': ['charttime', 'labtest', 'patient_id']}
    """

    event_conversion_cfg = copy.deepcopy(event_conversion_cfg)

    # Initialize a dictionary to store file paths as keys and lists of column names as values.
    prefix_to_columns = {}

    default_patient_id_col = event_conversion_cfg.pop("patient_id_col", "patient_id")
    for input_prefix, event_cfgs in event_conversion_cfg.items():
        input_patient_id_column = event_cfgs.pop("patient_id_col", default_patient_id_col)

        prefix_to_columns[input_prefix] = {input_patient_id_column}

        for event_cfg in event_cfgs.values():
            # If the config has a 'code' key and it contains column fields, parse and add them.
            for key, value in event_cfg.items():
                if key in META_KEYS:
                    continue

                if value is None:
                    # None can be used to indicate a null timestamp, which has no associated column.
                    continue

                if isinstance(value, str):
                    value = [value]

                for field in value:
                    if is_col_field(field):
                        prefix_to_columns[input_prefix].add(parse_col_field(field))
                    elif key == "code":
                        # strings in the "code" fields are literals, not columns
                        continue
                    else:
                        prefix_to_columns[input_prefix].add(field)

    # Return things in sorted order for determinism.
    return {k: list(sorted(v)) for k, v in prefix_to_columns.items()}


def filter_to_row_chunk(df: pl.LazyFrame, start: int, end: int) -> pl.LazyFrame:
    return df.filter(pl.col(ROW_IDX_NAME).is_between(start, end, closed="left")).drop(ROW_IDX_NAME)


@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Runs the input data re-sharding process. Can be parallelized across output shards.

    Output shards are simply row-chunks of the input data. There is no randomization or re-ordering of the
    input data. Read contention on the input files may render additional parallelism beyond one worker per
    input file ineffective.
    """
    hydra_loguru_init()

    raw_cohort_dir = Path(cfg.raw_cohort_dir)
    MEDS_cohort_dir = Path(cfg.MEDS_cohort_dir)
    row_chunksize = cfg.row_chunksize

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")
    logger.info(f"Reading event conversion config from {event_conversion_cfg_fp} to identify needed columns.")
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)

    prefix_to_columns = retrieve_columns(event_conversion_cfg)

    seen_files = set()
    input_files_to_subshard = []
    for fmt in ["parquet", "csv", "csv.gz"]:
        files_in_fmt = set(list(raw_cohort_dir.glob(f"*.{fmt}")) + list(raw_cohort_dir.glob(f"**/*.{fmt}")))
        for f in files_in_fmt:
            if get_shard_prefix(raw_cohort_dir, f) in seen_files:
                logger.warning(f"Skipping {f} as it has already been added in a preferred format.")
                continue
            elif get_shard_prefix(raw_cohort_dir, f) not in prefix_to_columns:
                logger.warning(f"Skipping {f} as it is not specified in the event conversion configuration.")
                continue
            else:
                input_files_to_subshard.append(f)
                seen_files.add(get_shard_prefix(raw_cohort_dir, f))

    random.shuffle(input_files_to_subshard)

    subsharding_files_strs = "\n".join([f"  * {str(fp.resolve())}" for fp in input_files_to_subshard])
    logger.info(
        f"Starting event sub-sharding. Sub-sharding {len(input_files_to_subshard)} files:\n"
        f"{subsharding_files_strs}"
    )
    logger.info(
        f"Will read raw data from {str(raw_cohort_dir.resolve())}/$IN_FILE.parquet and write sub-sharded "
        f"data to {str(MEDS_cohort_dir.resolve())}/sub_sharded/$IN_FILE/$ROW_START-$ROW_END.parquet"
    )

    start = datetime.now()
    for input_file in input_files_to_subshard:
        columns = prefix_to_columns[get_shard_prefix(raw_cohort_dir, input_file)]

        out_dir = MEDS_cohort_dir / "sub_sharded" / get_shard_prefix(raw_cohort_dir, input_file)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing {input_file} to {out_dir}.")

        logger.info(f"Performing preliminary read of {str(input_file.resolve())} to determine row count.")
        df = scan_with_row_idx(input_file, columns=columns, infer_schema_length=cfg["infer_schema_length"])

        row_count = df.select(pl.len()).collect().item()

        if row_count == 0:
            logger.warning(
                f"File {str(input_file.resolve())} reports "
                f"`df.select(pl.len()).collect().item()={row_count}`. Trying to debug"
            )
            logger.warning(f"Columns: {', '.join(df.columns)}")
            logger.warning(f"First 10 rows:\n{df.head(10).collect()}")
            logger.warning(f"Last 10 rows:\n{df.tail(10).collect()}")
            raise ValueError(
                f"File {str(input_file.resolve())} has no rows! If this is not an error, exclude it from "
                f"the event conversion configuration in {str(event_conversion_cfg_fp.resolve())}."
            )

        logger.info(f"Read {row_count} rows from {str(input_file.resolve())}.")

        row_shards = list(range(0, row_count, row_chunksize))
        random.shuffle(row_shards)
        logger.info(f"Splitting {input_file} into {len(row_shards)} row-chunks of size {row_chunksize}.")

        for i, st in enumerate(row_shards):
            end = min(st + row_chunksize, row_count)
            out_fp = out_dir / f"[{st}-{end}).parquet"

            compute_fn = partial(filter_to_row_chunk, start=st, end=end)
            logger.info(
                f"Writing file {i+1}/{len(row_shards)}: {input_file} row-chunk [{st}-{end}) to {out_fp}."
            )
            rwlock_wrap(
                input_file,
                out_fp,
                partial(scan_with_row_idx, columns=columns, infer_schema_length=cfg["infer_schema_length"]),
                write_lazyframe,
                compute_fn,
                do_overwrite=cfg.do_overwrite,
            )
    end = datetime.now()
    logger.info(f"Sub-sharding completed in {datetime.now() - start}")


if __name__ == "__main__":
    main()
