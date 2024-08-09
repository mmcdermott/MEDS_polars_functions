"""Basic utilities for parallelizable map operations on sharded MEDS datasets with caching and locking."""

import inspect
from collections.abc import Callable, Generator
from datetime import datetime
from enum import Enum, auto
from functools import partial, wraps
from pathlib import Path
from typing import Any, NotRequired, TypedDict, TypeVar

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, ListConfig

from ..extract.parser import is_matcher, matcher_to_expr
from ..utils import stage_init, write_lazyframe
from .utils import rwlock_wrap, shard_iterator

DF_T = TypeVar("DF_T")

COMPUTE_FN_T = Callable[[DF_T], DF_T]
COMPUTE_FN_UNBOUND_T = Callable[..., DF_T]
COMPUTE_FNTR_T = Callable[..., COMPUTE_FN_T]
ANY_COMPUTE_FN_T = COMPUTE_FN_T | COMPUTE_FN_UNBOUND_T | COMPUTE_FNTR_T


class ComputeFnArgs(TypedDict, total=False):
    df: NotRequired[DF_T]
    cfg: NotRequired[DictConfig]
    stage_cfg: NotRequired[DictConfig]
    code_modifiers: NotRequired[list[str]]
    code_metadata: NotRequired[pl.DataFrame]


SHARD_GEN_T = Generator[tuple[Path, Path], None, None]
SHARD_ITR_FNTR_T = Callable[[DictConfig], SHARD_GEN_T]


class ComputeFnType(Enum):
    """Stores the three types of compute functions that can be used in this utility.

    Attributes:
        DIRECT: A direct compute function that maps a DataFrame to a DataFrame.
        UNBOUND: An unbound compute function that needs to be called with additional parameters in addition to
            the input DataFrame, but still returns a DataFrame.
        FUNCTOR: A functor that needs to be called with non "df" args to return a direct compute function.
    """

    DIRECT = auto()
    UNBOUND = auto()
    FUNCTOR = auto()


def compute_fn_type(compute_fn: ANY_COMPUTE_FN_T) -> ComputeFnType | None:
    """Determine the type of a compute function, or return None if the type matches no compute function type.

    Returns the type of the compute function:
      - ComputeFnType.DIRECT if the compute function is a direct compute function -- i.e., if it takes only a
        single parameter named ``"df"`` and if the return annotation (if one exists) is not a
        `collections.abc.Callable` type annotation.
      - ComputeFnType.UNBOUND if the compute function is an unbound compute function -- i.e., if it takes a
        ``"df"`` parameter and at least one other parameter among the set of allowed additional or functor
        parameters, and if the return annotation (if one exists) is not a `collections.abc.Callable` type
        annotation.
      - ComputeFnType.FUNCTOR if the compute function is a functor that returns a direct compute function when
        called with the appropriate parameters -- i.e., if it takes no ``"df"`` parameter, if all parameters
        are among the allowed functor parameters, and if the return annotation (if one exists) is a
        `collections.abc.Callable` type annotation.

    Allowed functor parameters are:
      - "cfg" for the DictConfig configuration object.
      - "stage_cfg" for the DictConfig stage configuration object.
      - "code_modifiers" for a list of code modifier columns.
      - "code_metadata" for a Polars DataFrame of code metadata

    Args:
        compute_fn: The compute function to check.

    Examples:
        >>> def compute_fn(df: pl.DataFrame) -> pl.DataFrame: return df
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.DIRECT: 1>
        >>> def compute_fn(df: pl.DataFrame) -> dict[Any, list[Any]]: return df.to_dict(as_series=False)
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.DIRECT: 1>
        >>> def compute_fn(df: pl.DataFrame): return None
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.DIRECT: 1>
        >>> def compute_fn(foo: pl.DataFrame): return None
        >>> compute_fn_type(compute_fn) is None
        True
        >>> def compute_fn(df: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame: return df
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.UNBOUND: 2>
        >>> def compute_fn(df: pl.DataFrame, foo: DictConfig) -> pl.DataFrame: return df
        >>> compute_fn_type(compute_fn) is None
        True
        >>> def compute_fn(df: pl.LazyFrame, cfg: DictConfig): return df
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.UNBOUND: 2>
        >>> def compute_fn(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]: return lambda df: df
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.FUNCTOR: 3>
        >>> def compute_fn(cfg: DictConfig): return lambda df: df
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.FUNCTOR: 3>
        >>> def compute_fn(df: pl.DataFrame, cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
        ...     return lambda df: df
        >>> compute_fn_type(compute_fn) is None
        True
    """
    sig = inspect.signature(compute_fn)

    allowed_params = {*ComputeFnArgs.__required_keys__, *ComputeFnArgs.__optional_keys__}
    all_params_allowed = all(param in allowed_params for param in sig.parameters.keys())
    if not all_params_allowed:
        return None

    has_df_param = "df" in sig.parameters
    has_only_df_param = has_df_param and (len(sig.parameters) == 1)
    has_return_annotation = sig.return_annotation.__name__ != "_empty"
    has_callable_return = sig.return_annotation.__name__ == "Callable"

    if has_only_df_param:
        if has_return_annotation:
            return ComputeFnType.DIRECT if not has_callable_return else None
        return ComputeFnType.DIRECT
    elif has_df_param:
        if has_return_annotation:
            return ComputeFnType.UNBOUND if not has_callable_return else None
        return ComputeFnType.UNBOUND
    else:
        if has_return_annotation:
            return ComputeFnType.FUNCTOR if has_callable_return else None
        return ComputeFnType.FUNCTOR


def identity_fn(df: Any) -> Any:
    """A "null" compute function that returns the input DataFrame as is.

    Args:
        df: The input DataFrame.

    Returns:
        The input DataFrame.

    Examples:
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> (identity_fn(df) == df).select(pl.all_horizontal(pl.all().all())).item()
        True
    """

    return df


def read_and_filter_fntr(filter_expr: pl.Expr, read_fn: Callable[[Path], DF_T]) -> Callable[[Path], DF_T]:
    """Create a function that reads a DataFrame from a file and filters it based on a given expression.

    This is specified as a functor in this way to allow it to modify arbitrary other read functions for use in
    different mapreduce pipelines.

    Args:
        filter_expr: The filter expression to apply to the DataFrame.
        read_fn: The read function to use to read the DataFrame.

    Returns:
        A function that reads a DataFrame from a file and filters it based on the given expression.

    Examples:
        >>> dfs = {
        ...     "df1": pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        ...     "df2": pl.DataFrame({"a": [4, 5, 6], "b": [7, 8, 9]})
        ... }
        >>> read_fn = lambda key: dfs[key]
        >>> fn = read_and_filter_fntr((pl.col("a") % 2) == 0, read_fn)
        >>> fn("df1")
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 5   │
        └─────┴─────┘
        >>> fn("df2")
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 4   ┆ 7   │
        │ 6   ┆ 9   │
        └─────┴─────┘
        >>> fn = read_and_filter_fntr((pl.col("b") % 2) == 0, read_fn)
        >>> fn("df1")
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 3   ┆ 6   │
        └─────┴─────┘
        >>> fn("df2")
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 5   ┆ 8   │
        └─────┴─────┘
    """

    def read_and_filter(in_fp: Path) -> DF_T:
        return read_fn(in_fp).filter(filter_expr)

    return read_and_filter


MATCH_REVISE_KEY = "_match_revise"
MATCHER_KEY = "_matcher"


def is_match_revise(stage_cfg: DictConfig) -> bool:
    """Check if the stage configuration is in a match and revise format.

    Examples:
        >>> is_match_revise(DictConfig({"_match_revise": []}))
        False
        >>> is_match_revise(DictConfig({"_match_revise": [{"_matcher": {"code": "CODE//TEMP"}}]}))
        True
        >>> is_match_revise(DictConfig({"foo": "bar"}))
        False
    """
    return bool(stage_cfg.get(MATCH_REVISE_KEY, False))


def validate_match_revise(stage_cfg: DictConfig):
    """Validate that the stage configuration is in a match and revise format.

    Examples:
        >>> validate_match_revise(DictConfig({"foo": []}))
        Traceback (most recent call last):
            ...
        ValueError: Stage configuration must contain a _match_revise key
        >>> validate_match_revise(DictConfig({"_match_revise": "foo"}))
        Traceback (most recent call last):
            ...
        ValueError: Match revise options must be a list, got <class 'str'>
        >>> validate_match_revise(DictConfig({"_match_revise": [1]}))
        Traceback (most recent call last):
            ...
        ValueError: Match revise config 0 must be a dict, got <class 'int'>
        >>> validate_match_revise(DictConfig({"_match_revise": [{"_matcher": {"foo": "bar"}}, 1]}))
        Traceback (most recent call last):
            ...
        ValueError: Match revise config 1 must be a dict, got <class 'int'>
        >>> validate_match_revise(DictConfig({"_match_revise": [{"foo": "bar"}]}))
        Traceback (most recent call last):
            ...
        ValueError: Match revise config 0 must contain a _matcher key
        >>> validate_match_revise(DictConfig({"_match_revise": [{"_matcher": {32: "bar"}}]}))
        Traceback (most recent call last):
            ...
        ValueError: Match revise config 0 must contain a valid matcher in _matcher
        >>> validate_match_revise(DictConfig({"_match_revise": [{"_matcher": {"code": "CODE//TEMP"}}]}))
    """

    if MATCH_REVISE_KEY not in stage_cfg:
        raise ValueError(f"Stage configuration must contain a {MATCH_REVISE_KEY} key")

    match_revise_options = stage_cfg[MATCH_REVISE_KEY]
    if not isinstance(match_revise_options, (list, ListConfig)):
        raise ValueError(f"Match revise options must be a list, got {type(match_revise_options)}")

    for i, match_revise_cfg in enumerate(match_revise_options):
        if not isinstance(match_revise_cfg, (dict, DictConfig)):
            raise ValueError(f"Match revise config {i} must be a dict, got {type(match_revise_cfg)}")

        if MATCHER_KEY not in match_revise_cfg:
            raise ValueError(f"Match revise config {i} must contain a {MATCHER_KEY} key")

        if not is_matcher(match_revise_cfg[MATCHER_KEY]):
            raise ValueError(f"Match revise config {i} must contain a valid matcher in {MATCHER_KEY}")


def match_revise_fntr(cfg: DictConfig, stage_cfg: DictConfig, compute_fn: ANY_COMPUTE_FN_T) -> COMPUTE_FN_T:
    """A functor that creates a match & revise compute function based on the given configuration.

    Stage configurations for match & revise must be in a match and revise format. Consider the below example,
    showing the ``stage_cfg`` object in ``yaml`` format:

        ```yaml
        global_arg_1: "foo"
        _match_revise:
          - _matcher: {code: "CODE//BAR"}
            local_arg_1: "bar"
          - _matcher: {code: "CODE//BAZ"}
            local_arg_1: "baz"
        ```

    This configuration will create a match & revise compute function that will filter the input DataFrame for
    rows that match the ``CODE//BAR`` code and apply the compute function with the ``local_arg_1=bar``
    parameter, and then filter the input DataFrame for rows that match the ``CODE//BAZ`` code and apply the
    compute function with the ``local_arg_1=baz`` parameter. Both of these local compute functions will be
    applied to the input DataFrame in sequence, and the resulting DataFrames will be concatenated alongside
    any of the dataframe that matches no matcher (which will be left unmodified) and merged in a sorted way
    that respects the ``patient_id``, ``time`` ordering first, then the order of the match & revise blocks
    themselves, then the order of the rows in each match & revise block output. Each local compute function
    will also use the ``global_arg_1=foo`` parameter.

    Args:
        cfg: The DictConfig configuration object.
        stage_cfg: The DictConfig stage configuration object. This stage configuration must be in a match and
            revise format, meaning it must have a key ``"_match_revise"`` that contains a list of local match
            & revise configurations. Each local match & revise configuration must contain a key ``"_matcher"``
            which links to the matcher configuration to use to filter the input DataFrame for the local
            compute execution, and all other keys are local configuration parameters to be used in the local
            compute execution.
        compute_fn: The compute function to bind to the match & revise configuration local arguments.

    Returns:
        A function that applies the match & revise compute function to the input DataFrame.

    Raises:
        ValueError: If the stage configuration is not in a match and revise format.

    Examples:
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 2, 2, 2],
        ...     "time": [1, 2, 2, 1, 1, 2],
        ...     "initial_idx": [0, 1, 2, 3, 4, 5],
        ...     "code": ["FINAL", "CODE//TEMP_2", "CODE//TEMP_1", "FINAL", "CODE//TEMP_2", "CODE//TEMP_1"]
        ... })
        >>> def compute_fn(df: pl.DataFrame, stage_cfg: DictConfig) -> pl.DataFrame:
        ...     return df.with_columns(
        ...         pl.col("code").str.slice(0, len("CODE//")) +
        ...         stage_cfg.local_code_mid + "//" + stage_cfg.global_code_end
        ...     )
        >>> stage_cfg = DictConfig({
        ...     "global_code_end": "foo",
        ...     "_match_revise": [
        ...         {"_matcher": {"code": "CODE//TEMP_1"}, "local_code_mid": "bar"},
        ...         {"_matcher": {"code": "CODE//TEMP_2"}, "local_code_mid": "baz"}
        ...     ]
        ... })
        >>> cfg = DictConfig({"stage_cfg": stage_cfg})
        >>> match_revise_fn = match_revise_fntr(cfg, stage_cfg, compute_fn)
        >>> match_revise_fn(df.lazy()).collect()
        shape: (6, 4)
        ┌────────────┬──────┬─────────────┬────────────────┐
        │ patient_id ┆ time ┆ initial_idx ┆ code           │
        │ ---        ┆ ---  ┆ ---         ┆ ---            │
        │ i64        ┆ i64  ┆ i64         ┆ str            │
        ╞════════════╪══════╪═════════════╪════════════════╡
        │ 1          ┆ 1    ┆ 0           ┆ FINAL          │
        │ 1          ┆ 2    ┆ 2           ┆ CODE//bar//foo │
        │ 1          ┆ 2    ┆ 1           ┆ CODE//baz//foo │
        │ 2          ┆ 1    ┆ 4           ┆ CODE//baz//foo │
        │ 2          ┆ 1    ┆ 3           ┆ FINAL          │
        │ 2          ┆ 2    ┆ 5           ┆ CODE//bar//foo │
        └────────────┴──────┴─────────────┴────────────────┘
        >>> stage_cfg = DictConfig({
        ...     "global_code_end": "foo",
        ...     "_match_revise": [
        ...         {"_matcher": {"code": "CODE//TEMP_2"}, "local_code_mid": "bizz"},
        ...         {"_matcher": {"code": "CODE//TEMP_1"}, "local_code_mid": "foo", "global_code_end": "bar"},
        ...     ]
        ... })
        >>> cfg = DictConfig({"stage_cfg": stage_cfg})
        >>> match_revise_fn = match_revise_fntr(cfg, stage_cfg, compute_fn)
        >>> match_revise_fn(df.lazy()).collect()
        shape: (6, 4)
        ┌────────────┬──────┬─────────────┬─────────────────┐
        │ patient_id ┆ time ┆ initial_idx ┆ code            │
        │ ---        ┆ ---  ┆ ---         ┆ ---             │
        │ i64        ┆ i64  ┆ i64         ┆ str             │
        ╞════════════╪══════╪═════════════╪═════════════════╡
        │ 1          ┆ 1    ┆ 0           ┆ FINAL           │
        │ 1          ┆ 2    ┆ 1           ┆ CODE//bizz//foo │
        │ 1          ┆ 2    ┆ 2           ┆ CODE//foo//bar  │
        │ 2          ┆ 1    ┆ 4           ┆ CODE//bizz//foo │
        │ 2          ┆ 1    ┆ 3           ┆ FINAL           │
        │ 2          ┆ 2    ┆ 5           ┆ CODE//foo//bar  │
        └────────────┴──────┴─────────────┴─────────────────┘
        >>> stage_cfg = DictConfig({
        ...     "global_code_end": "foo", "_match_revise": [{"_matcher": {"missing": "CODE//TEMP_2"}}]
        ... })
        >>> cfg = DictConfig({"stage_cfg": stage_cfg})
        >>> match_revise_fn = match_revise_fntr(cfg, stage_cfg, compute_fn)
        >>> match_revise_fn(df.lazy()).collect() # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: Missing needed columns {'missing'} for local matcher 0:
            [(col("missing")) == (String(CODE//TEMP_2))].all_horizontal()
        Columns available: 'code', 'initial_idx', 'patient_id', 'time'
        >>> stage_cfg = DictConfig({"global_code_end": "foo"})
        >>> cfg = DictConfig({"stage_cfg": stage_cfg})
        >>> match_revise_fn = match_revise_fntr(cfg, stage_cfg, compute_fn)
        Traceback (most recent call last):
            ...
        ValueError: Invalid match and revise configuration...
    """
    try:
        validate_match_revise(stage_cfg)
    except ValueError as e:
        raise ValueError("Invalid match and revise configuration") from e

    stage_cfg = hydra.utils.instantiate(stage_cfg)

    matchers_and_fns = []
    for match_revise_cfg in stage_cfg.pop(MATCH_REVISE_KEY):
        matcher, cols = matcher_to_expr(match_revise_cfg.pop(MATCHER_KEY))
        local_stage_cfg = DictConfig({**stage_cfg, **match_revise_cfg})
        local_compute_fn = bind_compute_fn(cfg, local_stage_cfg, compute_fn)

        matchers_and_fns.append((matcher, cols, local_compute_fn))

    @wraps(compute_fn)
    def match_revise_fn(df: DF_T) -> DF_T:
        unmatched_df = df
        cols = set(df.collect_schema().names())

        revision_parts = []
        for i, (matcher_expr, need_cols, local_compute_fn) in enumerate(matchers_and_fns):
            if not need_cols.issubset(cols):
                cols_str = "', '".join(x for x in sorted(cols))
                raise ValueError(
                    f"Missing needed columns {need_cols - cols} for local matcher {i}: "
                    f"{matcher_expr}\nColumns available: '{cols_str}'"
                )
            matched_df = unmatched_df.filter(matcher_expr)
            unmatched_df = unmatched_df.filter(~matcher_expr)

            revision_parts.append(local_compute_fn(matched_df))

        revision_parts.append(unmatched_df)
        return pl.concat(revision_parts, how="vertical").sort(["patient_id", "time"], maintain_order=True)

    return match_revise_fn


def bind_compute_fn(cfg: DictConfig, stage_cfg: DictConfig, compute_fn: ANY_COMPUTE_FN_T) -> COMPUTE_FN_T:
    """Bind the compute function to the appropriate parameters based on the type of the compute function.

    Args:
        cfg: The DictConfig configuration object.
        stage_cfg: The DictConfig stage configuration object. This is separated from the ``cfg`` argument
            because in some cases, such as under the match & revise paradigm, the stage config may be modified
            dynamically under different matcher conditions to yield different compute functions.
        compute_fn: The compute function to bind.

    Returns:
        The compute function bound to the appropriate parameters.

    Raises:
        ValueError: If the compute function is not a valid compute function.

    Examples:
        >>> compute_fn = bind_compute_fn(DictConfig({}), DictConfig({}), None)
        >>> compute_fn("foobar")
        'foobar'
        >>> def compute_fntr(df: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame:
        ...     return df.with_columns(pl.lit(cfg.val).alias("val"))
        >>> compute_fn = bind_compute_fn(DictConfig({"val": "foo"}), None, compute_fntr)
        >>> compute_fn(pl.DataFrame({"a": [1, 2, 3]}))
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ val │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 1   ┆ foo │
        │ 2   ┆ foo │
        │ 3   ┆ foo │
        └─────┴─────┘
        >>> def compute_fntr(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
        ...     return lambda df: df.with_columns(pl.lit(cfg.val).alias("val"))
        >>> compute_fn = bind_compute_fn(DictConfig({"val": "foo"}), None, compute_fntr)
        >>> compute_fn(pl.DataFrame({"a": [1, 2, 3]}))
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ val │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 1   ┆ foo │
        │ 2   ┆ foo │
        │ 3   ┆ foo │
        └─────┴─────┘
        >>> def compute_fntr(stage_cfg, cfg) -> Callable[[pl.DataFrame], pl.DataFrame]:
        ...     return lambda df: df.with_columns(
        ...         pl.lit(stage_cfg.val).alias("stage_val"), pl.lit(cfg.val).alias("cfg_val")
        ...     )
        >>> compute_fn = bind_compute_fn(DictConfig({"val": "quo"}), DictConfig({"val": "bar"}), compute_fntr)
        >>> compute_fn(pl.DataFrame({"a": [1, 2, 3]}))
        shape: (3, 3)
        ┌─────┬───────────┬─────────┐
        │ a   ┆ stage_val ┆ cfg_val │
        │ --- ┆ ---       ┆ ---     │
        │ i64 ┆ str       ┆ str     │
        ╞═════╪═══════════╪═════════╡
        │ 1   ┆ bar       ┆ quo     │
        │ 2   ┆ bar       ┆ quo     │
        │ 3   ┆ bar       ┆ quo     │
        └─────┴───────────┴─────────┘
        >>> def compute_fntr(df, code_metadata):
        ...     return df.join(code_metadata, on="a")
        >>> code_metadata_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmpdir:
        ...     code_metadata_fp = Path(tmpdir) / "codes.parquet"
        ...     code_metadata_df.write_parquet(code_metadata_fp)
        ...     stage_cfg = DictConfig({"metadata_input_dir": tmpdir})
        ...     compute_fn = bind_compute_fn(DictConfig({}), stage_cfg, compute_fntr)
        ...     compute_fn(pl.DataFrame({"a": [1, 2, 3]}))
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        └─────┴─────┘
        >>> def compute_fntr(df: pl.DataFrame, cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
        ...     return lambda df: df
        >>> bind_compute_fn(DictConfig({}), DictConfig({}), compute_fntr)
        Traceback (most recent call last):
            ...
        ValueError: Invalid compute function
    """

    def fntr_params(compute_fn: ANY_COMPUTE_FN_T) -> ComputeFnArgs:
        compute_fn_params = inspect.signature(compute_fn).parameters
        kwargs = ComputeFnArgs()

        if "cfg" in compute_fn_params:
            kwargs["cfg"] = cfg
        if "stage_cfg" in compute_fn_params:
            kwargs["stage_cfg"] = stage_cfg
        if "code_modifiers" in compute_fn_params:
            code_modifiers = cfg.get("code_modifiers", None)
            kwargs["code_modifiers"] = code_modifiers
        if "code_metadata" in compute_fn_params:
            kwargs["code_metadata"] = pl.read_parquet(
                Path(stage_cfg.metadata_input_dir) / "codes.parquet", use_pyarrow=True
            )
        return kwargs

    if compute_fn is None:
        return identity_fn
    match compute_fn_type(compute_fn):
        case ComputeFnType.DIRECT:
            pass
        case ComputeFnType.UNBOUND:
            compute_fn = partial(compute_fn, **fntr_params(compute_fn))
        case ComputeFnType.FUNCTOR:
            compute_fn = compute_fn(**fntr_params(compute_fn))
        case _:
            raise ValueError("Invalid compute function")

    return compute_fn


def map_over(
    cfg: DictConfig,
    compute_fn: COMPUTE_FN_T | None = None,
    read_fn: Callable[[Path], DF_T] = partial(pl.scan_parquet, glob=False),
    write_fn: Callable[[DF_T, Path], None] = write_lazyframe,
    shard_iterator_fntr: SHARD_ITR_FNTR_T = shard_iterator,
) -> list[Path]:
    stage_init(cfg)

    start = datetime.now()

    process_split = cfg.stage_cfg.get("process_split", None)
    split_fp = Path(cfg.input_dir) / "metadata" / "patient_split.parquet"
    shards_map_fp = Path(cfg.shards_map_fp) if "shards_map_fp" in cfg else None
    if process_split and split_fp.exists():
        split_patients = (
            pl.scan_parquet(split_fp)
            .filter(pl.col("split") == process_split)
            .select(pl.col("patient_id"))
            .collect()
            .to_list()
        )
        read_fn = read_and_filter_fntr(pl.col("patient_id").isin(split_patients), read_fn)
    elif process_split and shards_map_fp and shards_map_fp.exists():
        logger.warning(
            f"Split {process_split} requested, but no patient split file found at {str(split_fp)}. "
            f"Assuming this is handled through shard filtering."
        )
    elif process_split:
        raise ValueError(
            f"Split {process_split} requested, but no patient split file found at {str(split_fp)}."
        )

    if is_match_revise(cfg.stage_cfg):
        compute_fn = match_revise_fntr(cfg, cfg.stage_cfg, compute_fn)
    else:
        compute_fn = bind_compute_fn(cfg, cfg.stage_cfg, compute_fn)

    all_out_fps = []
    for in_fp, out_fp in shard_iterator_fntr(cfg):
        logger.info(f"Processing {str(in_fp.resolve())} into {str(out_fp.resolve())}")
        rwlock_wrap(
            in_fp,
            out_fp,
            read_fn,
            write_fn,
            compute_fn,
            do_return=False,
            do_overwrite=cfg.do_overwrite,
        )
        all_out_fps.append(out_fp)

    logger.info(f"Finished mapping in {datetime.now() - start}")
    return all_out_fps
