#!/usr/bin/env python
"""Utilities for extracting code metadata about the codes produced for the MEDS events."""

import copy
import random
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_transforms.extract import CONFIG_YAML, MEDS_METADATA_MANDATORY_TYPES
from MEDS_transforms.extract.convert_to_sharded_events import get_code_expr
from MEDS_transforms.extract.utils import get_supported_fp
from MEDS_transforms.mapreduce.mapper import rwlock_wrap
from MEDS_transforms.parser import cfg_to_expr
from MEDS_transforms.utils import stage_init, write_lazyframe


def extract_metadata(
    metadata_df: pl.LazyFrame, event_cfg: dict[str, str | None], allowed_codes: list | None = None
) -> pl.LazyFrame:
    """Extracts a single metadata dataframe block for an event configuration from the raw metadata.

    Args:
        df: The raw metadata DataFrame. Mandatory columns are determined by the `event_cfg` configuration
            dictionary.
        event_cfg: A dictionary containing the configuration for the event. This must contain the critical
            `"code"` key alongside a mandatory `_metadata` block, which must contain some columns that should
            be extracted from the metadata to link to the code.
            The `"code"` key must contain either (1) a string literal representing the code for the event or
            (2) the name of a column in the raw data from which the code should be extracted. In the latter
            case, the column name should be enclosed in `col()` function call syntax--e.g.,
            `col(my_code_column)`. Note there are no quotes used inside the `col()` function syntax.

    Returns:
        A DataFrame containing the metadata extracted and linked to appropriately constructed code strings for
        the event configuration. The output DataFrame will contain at least two columns: `"code"` and whatever
        metadata column is specified for extraction in the metadata block. The output dataframe will not
        necessarily be unique by code if the input metadata is not unique by code.

    Raises:
        KeyError: If the event configuration dictionary is missing the `"code"` or `"_metadata"` keys or if
            the `"_metadata_"` key is empty or if columns referenced by the event configuration dictionary are
            not found in the raw metadata.

    Examples:
        >>> extract_metadata(pl.DataFrame(), {})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain 'code' key. Got: []."
        >>> extract_metadata(pl.DataFrame(), {"code": "test"})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain a non-empty '_metadata' key. Got: [code]."
        >>> raw_metadata = pl.DataFrame({
        ...     "code": ["A", "B", "C", "D", "E"],
        ...     "code_modifier": ["1", "2", "3", "4", "5"],
        ...     "name": ["Code A-1", "B-2", "C with 3", "D, but 4", None],
        ...     "priority": [1, 2, 3, 4, 5],
        ... })
        >>> event_cfg = {
        ...     "code": ["FOO", "col(code)", "col(code_modifier)"],
        ...     "_metadata": {"desc": "name"},
        ... }
        >>> extract_metadata(raw_metadata, event_cfg)
        shape: (4, 2)
        ┌───────────┬──────────┐
        │ code      ┆ desc     │
        │ ---       ┆ ---      │
        │ str       ┆ str      │
        ╞═══════════╪══════════╡
        │ FOO//A//1 ┆ Code A-1 │
        │ FOO//B//2 ┆ B-2      │
        │ FOO//C//3 ┆ C with 3 │
        │ FOO//D//4 ┆ D, but 4 │
        └───────────┴──────────┘
        >>> extract_metadata(raw_metadata, event_cfg, allowed_codes=["FOO//A//1", "FOO//C//3"])
        shape: (2, 2)
        ┌───────────┬──────────┐
        │ code      ┆ desc     │
        │ ---       ┆ ---      │
        │ str       ┆ str      │
        ╞═══════════╪══════════╡
        │ FOO//A//1 ┆ Code A-1 │
        │ FOO//C//3 ┆ C with 3 │
        └───────────┴──────────┘
        >>> extract_metadata(raw_metadata.drop("code_modifier"), event_cfg)
        Traceback (most recent call last):
            ...
        KeyError: "Columns {'code_modifier'} not found in metadata columns: ['code', 'name', 'priority']"
        >>> extract_metadata(raw_metadata, ['foo'])
        Traceback (most recent call last):
            ...
        TypeError: Event configuration must be a dictionary. Got: <class 'list'> ['foo'].

    You can also manipulate the columns in more complex ways when assigning metadata from the input source,
    and mandatory MEDS metadata columns will be cast to the correct types:
        >>> raw_metadata = pl.DataFrame({
        ...     "code": ["A", "A", "C", "D"],
        ...     "code_modifier": ["1", "1", "2", "3"],
        ...     "code_modifier_2": ["1", "2", "3", "4"],
        ...     "title": ["A-1-1", "A-1-2", "C-2-3", None],
        ...     "special_title": ["used", None, None, None],
        ... })
        >>> event_cfg = {
        ...     "code": ["FOO", "col(code)", "col(code_modifier)"],
        ...     "_metadata": {
        ...         "description": ["special_title", "title"],
        ...         "parent_codes": [
        ...             {"OUT_VAL/{code_modifier}/2": {"code_modifier_2": "2"}},
        ...             {"OUT_VAL_for_3/{code_modifier}": {"code_modifier_2": "3"}},
        ...             {
        ...                 "matcher": {"code_modifier_2": "4"},
        ...                 "output": {"literal": "expanded form"},
        ...             },
        ...         ],
        ...     },
        ... }
        >>> extract_metadata(raw_metadata, event_cfg)
        shape: (4, 3)
        ┌───────────┬─────────────┬─────────────────────┐
        │ code      ┆ description ┆ parent_codes        │
        │ ---       ┆ ---         ┆ ---                 │
        │ str       ┆ str         ┆ list[str]           │
        ╞═══════════╪═════════════╪═════════════════════╡
        │ FOO//A//1 ┆ used        ┆ null                │
        │ FOO//A//1 ┆ A-1-2       ┆ ["OUT_VAL/1/2"]     │
        │ FOO//C//2 ┆ C-2-3       ┆ ["OUT_VAL_for_3/2"] │
        │ FOO//D//3 ┆ null        ┆ ["expanded form"]   │
        └───────────┴─────────────┴─────────────────────┘
    """
    event_cfg = copy.deepcopy(event_cfg)

    if not isinstance(event_cfg, (dict, DictConfig)):
        raise TypeError(f"Event configuration must be a dictionary. Got: {type(event_cfg)} {event_cfg}.")

    if "code" not in event_cfg:
        raise KeyError(
            "Event configuration dictionary must contain 'code' key. "
            f"Got: [{', '.join(event_cfg.keys())}]."
        )
    if "_metadata" not in event_cfg or not event_cfg["_metadata"]:
        raise KeyError(
            "Event configuration dictionary must contain a non-empty '_metadata' key. "
            f"Got: [{', '.join(event_cfg.keys())}]."
        )

    df_select_exprs = {}
    final_cols = []
    needed_cols = set()
    for out_col, in_cfg in event_cfg["_metadata"].items():
        in_expr, needed = cfg_to_expr(in_cfg)
        df_select_exprs[out_col] = in_expr
        final_cols.append(out_col)
        needed_cols.update(needed)

    code_expr, _, needed_code_cols = get_code_expr(event_cfg.pop("code"))

    columns = metadata_df.collect_schema().names()
    missing_cols = (needed_cols | needed_code_cols) - set(columns) - set(final_cols)
    if missing_cols:
        raise KeyError(f"Columns {missing_cols} not found in metadata columns: {columns}")

    for col in needed_code_cols:
        if col not in df_select_exprs:
            df_select_exprs[col] = pl.col(col)

    metadata_df = metadata_df.select(**df_select_exprs).with_columns(code=code_expr)

    if allowed_codes:
        metadata_df = metadata_df.filter(pl.col("code").is_in(allowed_codes))

    metadata_df = metadata_df.filter(~pl.all_horizontal(*[pl.col(c).is_null() for c in final_cols]))

    for mandatory_col, mandatory_type in MEDS_METADATA_MANDATORY_TYPES.items():
        if mandatory_col not in final_cols:
            continue

        if metadata_df.schema[mandatory_col] is not mandatory_type:
            logger.warning(f"Metadata column '{mandatory_col}' must be of type {mandatory_type}. Casting.")
            metadata_df = metadata_df.with_columns(pl.col(mandatory_col).cast(mandatory_type), strict=False)

    return metadata_df.unique(maintain_order=True).select("code", *final_cols)


def extract_all_metadata(
    metadata_df: pl.LazyFrame, event_cfgs: list[dict], allowed_codes: list | None = None
) -> pl.LazyFrame:
    """Extracts all metadata for a list of event configurations.

    Args:
        metadata_df: The raw metadata DataFrame. Mandatory columns are determined by the `event_cfg`
            configurations.
        event_cfgs: A list of event configuration dictionaries. Each dictionary must contain the code
            and metadata elements.
        allowed_codes: A list of codes to allow in the output metadata. If None, all codes are allowed.

    Returns:
        A unified DF containing all metadata for all event configurations.

    Examples:
        >>> raw_metadata = pl.DataFrame({
        ...     "code": ["A", "B", "C", "D"],
        ...     "code_modifier": ["1", "2", "3", "4"],
        ...     "name": ["Code A-1", "B-2", "C with 3", "D, but 4"],
        ...     "priority": [1, 2, 3, 4],
        ... })
        >>> event_cfg_1 = {
        ...     "code": ["FOO", "col(code)", "col(code_modifier)"],
        ...     "_metadata": {"desc": "name"},
        ... }
        >>> event_cfg_2 = {
        ...     "code": ["BAR", "col(code)", "col(code_modifier)"],
        ...     "_metadata": {"desc2": "name"},
        ... }
        >>> event_cfgs = [event_cfg_1, event_cfg_2]
        >>> extract_all_metadata(raw_metadata, event_cfgs, allowed_codes=["FOO//A//1", "BAR//B//2"])
        shape: (2, 3)
        ┌───────────┬──────────┬───────┐
        │ code      ┆ desc     ┆ desc2 │
        │ ---       ┆ ---      ┆ ---   │
        │ str       ┆ str      ┆ str   │
        ╞═══════════╪══════════╪═══════╡
        │ FOO//A//1 ┆ Code A-1 ┆ null  │
        │ BAR//B//2 ┆ null     ┆ B-2   │
        └───────────┴──────────┴───────┘
    """

    all_metadata = []
    for event_cfg in event_cfgs:
        all_metadata.append(extract_metadata(metadata_df, event_cfg, allowed_codes=allowed_codes))

    return pl.concat(all_metadata, how="diagonal_relaxed").unique(maintain_order=True)


def get_events_and_metadata_by_metadata_fp(event_configs: dict | DictConfig) -> dict[str, dict[str, dict]]:
    """Reformats the event conversion config to map metadata file input prefixes to linked event configs.

    Args:
        event_configs: The event conversion configuration dictionary.

    Returns:
        A dictionary keyed by metadata input file prefix mapping to a dictionary of event configurations that
        link to that metadata prefix.

    Examples:
        >>> event_configs = {
        ...     "patient_id_col": "MRN",
        ...     "icu/procedureevents": {
        ...         "patient_id_col": "subject_id",
        ...         "start": {
        ...             "code": ["PROCEDURE", "START", "col(itemid)"],
        ...             "_metadata": {
        ...                 "proc_datetimeevents": {"desc": ["omop_concept_name", "label"]},
        ...                 "proc_itemid": {"desc": ["omop_concept_name", "label"]},
        ...             },
        ...         },
        ...         "end": {
        ...             "code": ["PROCEDURE", "END", "col(itemid)"],
        ...             "_metadata": {
        ...                 "proc_datetimeevents": {"desc": ["omop_concept_name", "label"]},
        ...                 "proc_itemid": {"desc": ["omop_concept_name", "label"]},
        ...             },
        ...         },
        ...     },
        ...     "icu/inputevents": {
        ...         "event": {
        ...             "code": ["INFUSION", "col(itemid)"],
        ...             "_metadata": {
        ...                 "inputevents_to_rxnorm": {"desc": "{label}", "itemid": "{foo}"}
        ...             },
        ...         },
        ...     },
        ... }
        >>> get_events_and_metadata_by_metadata_fp(event_configs) # doctest: +NORMALIZE_WHITESPACE
        {'proc_datetimeevents': [{'code': ['PROCEDURE', 'START', 'col(itemid)'],
                                  '_metadata': {'desc': ['omop_concept_name', 'label']}},
                                 {'code': ['PROCEDURE', 'END', 'col(itemid)'],
                                  '_metadata': {'desc': ['omop_concept_name', 'label']}}],
         'proc_itemid':         [{'code': ['PROCEDURE', 'START', 'col(itemid)'],
                                  '_metadata': {'desc': ['omop_concept_name', 'label']}},
                                 {'code': ['PROCEDURE', 'END', 'col(itemid)'],
                                  '_metadata': {'desc': ['omop_concept_name', 'label']}}],
         'inputevents_to_rxnorm': [{'code': ['INFUSION', 'col(itemid)'],
                                    '_metadata': {'desc': '{label}', 'itemid': '{foo}'}}]}
        >>> no_metadata_event_configs = {
        ...     "icu/procedureevents": {
        ...         "start": {"code": ["PROCEDURE", "START", "col(itemid)"]},
        ...         "end": {"code": ["PROCEDURE", "END", "col(itemid)"]},
        ...     },
        ...     "icu/inputevents": {
        ...         "event": {"code": ["INFUSION", "col(itemid)"]},
        ...     },
        ... }
        >>> get_events_and_metadata_by_metadata_fp(no_metadata_event_configs)
        {}
    """

    out = {}

    for file_pfx, event_cfgs_for_pfx in event_configs.items():
        if file_pfx == "patient_id_col":
            continue

        for event_key, event_cfg in event_cfgs_for_pfx.items():
            if event_key == "patient_id_col":
                continue

            for metadata_pfx, metadata_cfg in event_cfg.get("_metadata", {}).items():
                if metadata_pfx not in out:
                    out[metadata_pfx] = []
                out[metadata_pfx].append({"code": event_cfg["code"], "_metadata": metadata_cfg})

    return out


@hydra.main(version_base=None, config_path=str(CONFIG_YAML.parent), config_name=CONFIG_YAML.stem)
def main(cfg: DictConfig):
    """Extracts any dataset-specific metadata and adds it to any existing code metadata file.

    This script can extract arbitrary, code-linked metadata columns from input mappings and add them to the
    `metadata/codes.parquet` file. The metadata columns are extracted from the raw metadata files using a
    parsing DSL that is specified in the `event_conversion_config_fp` file. See `parser.py` for more details
    on this DSL.

    Note that there are two sentinel columns in the output metadata that have certain mandates for MEDS
    compliance: The `description` column and the `parent_codes` column. The `description` column must be a
    string, and if there are multiple matches in the extracted metadata for a code, in this script they will
    be concatenated into a single string with the `description_separator` string. The `parent_codes` column
    must be a list of strings, each formatted as an OMOP vocabulary name, followed by a "/", followed by the
    OMOP concept code. This column is used to link codes to their parent codes in the OMOP vocabulary.

    All arguments are specified through the command line into the `cfg` object through Hydra.

    The `cfg.stage_cfg` object is a special key that is imputed by OmegaConf to contain the stage-specific
    configuration arguments based on the global, pipeline-level configuration file. It cannot be overwritten
    directly on the command line, but can be overwritten implicitly by overwriting components of the
    `stage_configs.extract_code_metadata` key.

    Args:
        stage_configs.extract_code_metadata.description_separator: If there are multiple metadata matches for
            a row, this string will be used as a separator to join the matches for the sentinel
            `"description"` column into a single string in the output metadata, per compliance with the MEDS
            schema.
    """

    stage_input_dir, partial_metadata_dir, _, _ = stage_init(cfg)
    raw_input_dir = Path(cfg.input_dir)

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")

    logger.info(f"Reading event conversion config from {event_conversion_cfg_fp}")
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    logger.info(f"Event conversion config:\n{OmegaConf.to_yaml(event_conversion_cfg)}")

    partial_metadata_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(event_conversion_cfg, partial_metadata_dir / "event_conversion_config.yaml")

    events_and_metadata_by_metadata_fp = get_events_and_metadata_by_metadata_fp(event_conversion_cfg)
    event_metadata_configs = list(events_and_metadata_by_metadata_fp.items())
    random.shuffle(event_metadata_configs)

    # Load all codes
    all_codes = (
        pl.scan_parquet(stage_input_dir / "**/*.parquet")
        .select(pl.col("code").unique())
        .collect()
        .get_column("code")
        .to_list()
    )

    all_out_fps = []
    for input_prefix, event_metadata_cfgs in event_metadata_configs:
        event_metadata_cfgs = copy.deepcopy(event_metadata_cfgs)

        metadata_fp, read_fn = get_supported_fp(raw_input_dir, input_prefix)
        if metadata_fp.suffix != ".parquet":
            read_fn = partial(read_fn, infer_schema_length=999999999)
        out_fp = partial_metadata_dir / f"{input_prefix}.parquet"
        logger.info(f"Extracting metadata from {metadata_fp} and saving to {out_fp}")

        compute_fn = partial(extract_all_metadata, event_cfgs=event_metadata_cfgs, allowed_codes=all_codes)

        rwlock_wrap(metadata_fp, out_fp, read_fn, write_lazyframe, compute_fn, do_overwrite=cfg.do_overwrite)
        all_out_fps.append(out_fp)

    logger.info("Extracted metadata for all events. Merging.")

    if cfg.worker != 0:
        logger.info("Code metadata extraction completed. Exiting")
        return

    logger.info("Starting reduction process")

    while not all(fp.exists() for fp in all_out_fps):
        missing_files_str = "\n".join(f"  - {str(fp.resolve())}" for fp in all_out_fps if not fp.exists())
        logger.info("Waiting to begin reduction for all files to be written...\n" f"{missing_files_str}")
        time.sleep(cfg.polling_time)

    start = datetime.now()
    logger.info("All map shards complete! Starting code metadata reduction computation.")

    def reducer_fn(*dfs):
        return pl.concat(dfs, how="diagonal_relaxed").unique(maintain_order=True)

    reduced = reducer_fn(*[pl.scan_parquet(fp, glob=False) for fp in all_out_fps])
    join_cols = ["code", *cfg.get("code_modifier_cols", [])]
    reduced_cols = reduced.collect_schema().names()
    metadata_cols = [c for c in reduced_cols if c not in join_cols]

    n_unique_obs = reduced.select(pl.n_unique(*join_cols)).collect().item()
    n_rows = reduced.select(pl.count()).collect().item()
    logger.info(f"Collected metadata for {n_unique_obs} unique codes among {n_rows} total observations.")

    if n_unique_obs != n_rows:
        aggs = {c: pl.col(c) for c in metadata_cols if c not in MEDS_METADATA_MANDATORY_TYPES}
        if "description" in metadata_cols:
            separator = cfg.stage_cfg.description_separator
            aggs["description"] = pl.col("description").str.concat(separator)
        if "parent_codes" in metadata_cols:
            aggs["parent_codes"] = pl.col("parent_codes").explode()

        reduced = reduced.group_by(join_cols).agg(**aggs)

    reduced = reduced.collect()

    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)
    old_metadata_fp = metadata_input_dir / "codes.parquet"

    if old_metadata_fp.exists():
        logger.info(f"Joining to existing code metadata at {str(old_metadata_fp.resolve())}")
        existing = pl.read_parquet(old_metadata_fp, use_pyarrow=True)
        reduced = existing.join(reduced, on=join_cols, how="full", coalesce=True)

    reducer_fp = Path(cfg.stage_cfg.reducer_output_dir) / "codes.parquet"
    reducer_fp.parent.mkdir(parents=True, exist_ok=True)
    reduced.write_parquet(reducer_fp, use_pyarrow=True)
    logger.info(f"Finished reduction in {datetime.now() - start}")


if __name__ == "__main__":
    main()
