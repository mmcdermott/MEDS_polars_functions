"""Core utilities for MEDS pipelines built with these tools."""

import inspect
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_transforms import __package_name__ as package_name
from MEDS_transforms import __version__ as package_version


def get_smallest_valid_uint_type(num: int | float | pl.Expr) -> pl.DataType:
    """Returns the smallest valid unsigned integral type for an ID variable with `num` unique options.

    Args:
        num: The number of IDs that must be uniquely expressed.

    Raises:
        ValueError: If there is no unsigned int type big enough to express the passed number of ID
            variables.

    Examples:
        >>> get_smallest_valid_uint_type(num=1)
        UInt8
        >>> get_smallest_valid_uint_type(num=2**8-1)
        UInt16
        >>> get_smallest_valid_uint_type(num=2**16-1)
        UInt32
        >>> get_smallest_valid_uint_type(num=2**32-1)
        UInt64
        >>> get_smallest_valid_uint_type(num=2**64-1)
        Traceback (most recent call last):
            ...
        ValueError: Value is too large to be expressed as an int!
    """
    if num >= (2**64) - 1:
        raise ValueError("Value is too large to be expressed as an int!")
    if num >= (2**32) - 1:
        return pl.UInt64
    elif num >= (2**16) - 1:
        return pl.UInt32
    elif num >= (2**8) - 1:
        return pl.UInt16
    else:
        return pl.UInt8


def write_lazyframe(df: pl.LazyFrame, out_fp: Path) -> None:
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    df.write_parquet(out_fp, use_pyarrow=True)


def stage_init(cfg: DictConfig) -> tuple[Path, Path, Path]:
    """Initializes the stage by logging the configuration and the stage-specific paths.

    Args:
        cfg: The global configuration object, which should have a ``cfg.stage_cfg`` attribute containing the
            stage specific configuration.

    Returns: The data input directory, stage output directory, and metadata input directory.
    """
    hydra_loguru_init()

    logger.info(
        f"Running {current_script_name()} with the following configuration:\n{OmegaConf.to_yaml(cfg)}"
    )

    input_dir = Path(cfg.stage_cfg.data_input_dir)
    output_dir = Path(cfg.stage_cfg.output_dir)
    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)

    def chk(x: Path):
        return "✅" if x.exists() else "❌"

    paths_strs = [
        f"  - {k}: {chk(v)} {str(v.resolve())}"
        for k, v in {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "metadata_input_dir": metadata_input_dir,
        }.items()
    ]

    logger_strs = [
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}",
        "Paths: (checkbox indicates if it exists)",
    ]
    logger.debug("\n".join(logger_strs + paths_strs))

    return input_dir, output_dir, metadata_input_dir


def get_package_name() -> str:
    """Returns the name of the python package running this pipeline."""
    return package_name


def get_package_version() -> str:
    """Returns the version of the python package running this pipeline."""
    return package_version


def get_script_docstring() -> str:
    """Returns the docstring of the main function of the script from which this function was called."""

    main_module = sys.modules["__main__"]
    func = getattr(main_module, "main", None)
    if func and callable(func):
        return inspect.getdoc(func) or ""
    return ""


def current_script_name() -> str:
    """Returns the name of the module that called this function."""

    main_module = sys.modules["__main__"]
    main_func = getattr(main_module, "main", None)
    if main_func and callable(main_func):
        func_module = main_func.__module__
        if func_module == "__main__":
            return Path(sys.argv[0]).stem
        else:
            return func_module.split(".")[-1]

    logger.warning("Can't find main function in __main__ module. Using sys.argv[0] as a fallback.")
    return Path(sys.argv[0]).stem


def is_metadata_stage(stage: dict[str, Any] | DictConfig) -> bool:
    """Determines if a stage is a metadata stage either by explicit argument or via the "aggregations" key.

    Args:
        stage: The stage configuration dictionary.

    Returns:
        bool: True if the stage is a metadata stage, False otherwise.

    Raises:
        TypeError: If the "is_metadata" key is present but not a boolean or `None`.

    Examples:
        >>> is_metadata_stage({"aggregations": ["foo"]})
        True
        >>> is_metadata_stage({"is_metadata": True})
        True
        >>> is_metadata_stage({"is_metadata": False})
        False
        >>> is_metadata_stage({"is_metadata": None})
        False
        >>> is_metadata_stage({"output_dir": "/a/b/metadata"})
        False
        >>> is_metadata_stage({})
        False
        >>> is_metadata_stage(DictConfig({"aggregations": ["foo"]}))
        True
        >>> is_metadata_stage(DictConfig({"is_metadata": 32}))
        Traceback (most recent call last):
            ...
        TypeError: If specified manually, is_metadata must be a boolean. Got 32
    """
    if "is_metadata" in stage:
        if not isinstance(stage["is_metadata"], (bool, type(None))):
            raise TypeError(
                f"If specified manually, is_metadata must be a boolean. Got {stage['is_metadata']}"
            )
        return bool(stage["is_metadata"])
    else:
        return "aggregations" in stage


def populate_stage(
    stage_name: str,
    input_dir: str,
    cohort_dir: str,
    stages: list[str],
    stage_configs: dict[str, dict],
    pre_parsed_stages: dict[str, dict] | None = None,
) -> dict:
    """Populates a stage in the stages configuration with inferred stage parameters.

    Infers and adds (unless already present, in which case the provided value is used) the following
    parameters to the stage configuration:
      - `is_metadata`: Whether the stage is a metadata stage, which is determined to be `False` if the stage
        does not have an `aggregations` parameter.
      - `data_input_dir`: The input directory for the stage (either the global input directory or the previous
        data stage's output directory).
      - `metadata_input_dir`: The input directory for the stage (either the global input directory or the
        previous metadata stage's output directory).
      - `output_dir`: The output directory for the stage (the cohort directory with the stage name appended).

    Args:
        stage_name: The name of the stage to populate.
        input_dir: The global input directory.
        cohort_dir: The cohort directory into which this overall pipeline is writing data.
        stages: The names of the stages processed by this pipeline in order.
        stage_configs: The raw, unresolved stage configuration dictionaries for any stages with specific
            arguments, keyed by stage name.
        pre_parsed_stages: The stages configuration dictionaries (resolved), keyed by stage name. If
            specified, the function will not re-resolve the stages in this list.

    Returns:
        dict: The populated stage configuration.

    Raises:
        ValueError: If the stage is not present in the stages configuration.

    Examples:
        >>> from omegaconf import DictConfig
        >>> root_config = DictConfig({
        ...     "input_dir": "/a/b",
        ...     "cohort_dir": "/c/d",
        ...     "stages": ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"],
        ...     "stage_configs": {
        ...         "stage2": {"is_metadata": True},
        ...         "stage3": {"is_metadata": None, "output_dir": "/g/h"},
        ...         "stage4": {"data_input_dir": "/e/f"},
        ...         "stage5": {"aggregations": ["foo"], "train_only": None},
        ...     },
        ... })
        >>> args = [root_config[k] for k in ["input_dir", "cohort_dir", "stages", "stage_configs"]]
        >>> populate_stage("stage1", *args) # doctest: +NORMALIZE_WHITESPACE
        {'is_metadata': False, 'data_input_dir': '/a/b/data', 'metadata_input_dir': '/a/b/metadata',
         'output_dir': '/c/d/stage1', 'reducer_output_dir': None}
        >>> populate_stage("stage2", *args) # doctest: +NORMALIZE_WHITESPACE
        {'is_metadata': True, 'data_input_dir': '/c/d/stage1', 'metadata_input_dir': '/a/b/metadata',
         'output_dir': '/c/d/stage2', 'reducer_output_dir': '/c/d/stage2', 'train_only': True}
        >>> populate_stage("stage3", *args) # doctest: +NORMALIZE_WHITESPACE
        {'is_metadata': None, 'output_dir': '/g/h', 'data_input_dir': '/c/d/stage1',
         'metadata_input_dir': '/c/d/stage2', 'reducer_output_dir': None}
        >>> populate_stage("stage4", *args) # doctest: +NORMALIZE_WHITESPACE
        {'data_input_dir': '/e/f', 'is_metadata': False,
         'metadata_input_dir': '/c/d/stage2', 'output_dir': '/c/d/stage4', 'reducer_output_dir': None}
        >>> populate_stage("stage5", *args) # doctest: +NORMALIZE_WHITESPACE
        {'aggregations': ['foo'], 'train_only': None, 'is_metadata': True, 'data_input_dir': '/c/d/stage4',
         'metadata_input_dir': '/c/d/stage2', 'output_dir': '/c/d/stage5',
         'reducer_output_dir': '/c/d/metadata'}
        >>> populate_stage("stage6", *args) # doctest: +NORMALIZE_WHITESPACE
        {'is_metadata': False, 'data_input_dir': '/c/d/stage4',
         'metadata_input_dir': '/c/d/stage5', 'output_dir': '/c/d/data', 'reducer_output_dir': None}
        >>> populate_stage("stage7", *args) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: 'stage7' is not a valid stage name. Options are: stage1, stage2, stage3, stage4, stage5,
            stage6
        >>> root_config = DictConfig({
        ...     "input_dir": "/a/b",
        ...     "cohort_dir": "/c/d",
        ...     "stages": ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"],
        ...     "stage_configs": {"stage2": {"is_metadata": 34}},
        ... })
        >>> args = [root_config[k] for k in ["input_dir", "cohort_dir", "stages", "stage_configs"]]
        >>> populate_stage("stage2", *args)
        Traceback (most recent call last):
            ...
        TypeError: If specified manually, is_metadata must be a boolean. Got 34
    """

    if stage_name not in stages:
        raise ValueError(f"'{stage_name}' is not a valid stage name. Options are: {', '.join(stages)}")

    if pre_parsed_stages is None:
        pre_parsed_stages = {}

    stage = None
    prior_data_stage = None
    prior_metadata_stage = None
    for s in stages:
        if s == stage_name:
            stage = stage_configs.get(s, {})
            break
        elif s in pre_parsed_stages:
            s_resolved = pre_parsed_stages[s]
        else:
            s_resolved = populate_stage(s, input_dir, cohort_dir, stages, stage_configs, pre_parsed_stages)

        pre_parsed_stages[s] = s_resolved
        if s_resolved["is_metadata"]:
            prior_metadata_stage = s_resolved
        else:
            prior_data_stage = s_resolved

    # First, we set the stage's input directories. It needs to be able to read both data shards and reduced
    # metadata files. These will either be pulled from the overall pipeline input directories or from the
    # relevant preceding stage's output directories, as appropriate.
    is_first_data_stage = prior_data_stage is None
    is_first_metadata_stage = prior_metadata_stage is None

    pipeline_input_data_dir = str(Path(input_dir) / "data")
    pipeline_input_metadata_dir = str(Path(input_dir) / "metadata")

    if is_first_data_stage:
        default_data_input_dir = pipeline_input_data_dir
    else:
        default_data_input_dir = prior_data_stage["output_dir"]

    if is_first_metadata_stage:
        default_metadata_input_dir = pipeline_input_metadata_dir
    else:
        default_metadata_input_dir = prior_metadata_stage["output_dir"]

    # Now, we need to set output directories. The output directory for the stage will either be a stage
    # specific output directory, or, for the last data or metadata stages, respectively, will be the global
    # pipeline output data shard or metadata directories, as appropriate.
    is_metadata = is_metadata_stage(stage)

    stage_index = stages.index(stage_name)

    is_later_data_stage = False
    is_later_metadata_stage = False
    for i in range(stage_index + 1, len(stages)):
        stage_i = stage_configs.get(stages[i], {})
        if is_metadata_stage(stage_i):
            is_later_metadata_stage = True
        else:
            is_later_data_stage = True

    is_last_data_stage = (not is_later_data_stage) and (not is_metadata)
    is_last_metadata_stage = (not is_later_metadata_stage) and is_metadata

    cohort_dir = Path(cohort_dir)

    if is_last_data_stage:
        default_mapper_output_dir = str(cohort_dir / "data")
        default_reducer_output_dir = None
    elif is_last_metadata_stage:
        default_mapper_output_dir = str(cohort_dir / stage_name)
        default_reducer_output_dir = str(cohort_dir / "metadata")
    elif is_metadata:
        default_mapper_output_dir = str(cohort_dir / stage_name)
        default_reducer_output_dir = str(cohort_dir / stage_name)
    else:
        default_mapper_output_dir = str(cohort_dir / stage_name)
        default_reducer_output_dir = None

    inferred_keys = {
        "is_metadata": is_metadata,
        "data_input_dir": default_data_input_dir,
        "metadata_input_dir": default_metadata_input_dir,
        "output_dir": default_mapper_output_dir,
        "reducer_output_dir": default_reducer_output_dir,
    }

    if is_metadata:
        inferred_keys["train_only"] = True

    out = {**stage}
    for key, val in inferred_keys.items():
        if key not in out:
            out[key] = val

    return out


OmegaConf.register_new_resolver("get_script_docstring", get_script_docstring, replace=False)
OmegaConf.register_new_resolver("current_script_name", current_script_name, replace=False)
OmegaConf.register_new_resolver("populate_stage", populate_stage, replace=False)
OmegaConf.register_new_resolver("get_package_version", get_package_version, replace=False)
OmegaConf.register_new_resolver("get_package_name", get_package_name, replace=False)


def hydra_loguru_init() -> None:
    """Adds loguru output to the logs that hydra scrapes.

    Must be called from a hydra main!
    """
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logfile_name = hydra.core.hydra_config.HydraConfig.get().job.name
    logger.add(os.path.join(hydra_path, f"{logfile_name}.log"))


def get_shard_prefix(base_path: Path, fp: Path) -> str:
    """Extracts the shard prefix from a file path by removing the raw_cohort_dir.

    Args:
        base_path: The base path to remove.
        fp: The file path to extract the shard prefix from.

    Returns:
        The shard prefix (the file path relative to the base path with the suffix removed).

    Examples:
        >>> get_shard_prefix(Path("/a/b/c"), Path("/a/b/c/d.parquet"))
        'd'
        >>> get_shard_prefix(Path("/a/b/c"), Path("/a/b/c/d/e.csv.gz"))
        'd/e'
    """

    relative_path = fp.relative_to(base_path)
    relative_parent = relative_path.parent
    file_name = relative_path.name.split(".")[0]

    return str(relative_parent / file_name)


def is_col_field(field: str | None) -> bool:
    """Checks if a string field is formatted as "col(column_name)".

    This format is used to denote a column in a Polars DataFrame in the event conversion configuration.

    Args:
        field (str | None): The field to check.

    Returns:
        bool: True if the field is formatted as "col(column_name)", False otherwise.

    Examples:
        >>> is_col_field("col(patient_id)")
        True
        >>> is_col_field("col(patient_id")
        False
        >>> is_col_field("patient_id)")
        False
        >>> is_col_field("column(patient_id)")
        False
        >>> is_col_field("patient_id")
        False
        >>> is_col_field(None)
        False
    """
    if field is None:
        return False
    return field.startswith("col(") and field.endswith(")")


def parse_col_field(field: str) -> str:
    """Extracts the actual column name from a string formatted as "col(column_name)".

    Args:
        field (str): A string formatted as "col(column_name)".

    Raises:
        ValueError: If the input string does not match the expected format.

    Examples:
        >>> parse_col_field("col(patient_id)")
        'patient_id'
        >>> parse_col_field("col(patient_id")
        Traceback (most recent call last):
        ...
        ValueError: Invalid column field: col(patient_id
        >>> parse_col_field("column(patient_id)")
        Traceback (most recent call last):
        ...
        ValueError: Invalid column field: column(patient_id)
    """
    if not is_col_field(field):
        raise ValueError(f"Invalid column field: {field}")
    return field[4:-1]
