"""Basic utilities for parallelizable map operations on sharded MEDS datasets with caching and locking."""

import inspect
from collections.abc import Callable, Generator
from datetime import datetime
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import Any, NotRequired, TypedDict, TypeVar

import polars as pl
from loguru import logger
from omegaconf import DictConfig

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
    return df


def read_and_filter_fntr(patients: list[int], read_fn: Callable[[Path], DF_T]) -> Callable[[Path], DF_T]:
    def read_and_filter(in_fp: Path) -> DF_T:
        df = read_fn(in_fp)
        return df.filter(pl.col("patient_id").isin(patients))

    return read_and_filter


def map_over(
    cfg: DictConfig,
    compute_fn: COMPUTE_FN_T | None = None,
    read_fn: Callable[[Path], DF_T] = partial(pl.scan_parquet, glob=False),
    write_fn: Callable[[DF_T, Path], None] = write_lazyframe,
    shard_iterator_fntr: SHARD_ITR_FNTR_T = shard_iterator,
) -> list[Path]:
    stage_init(cfg)

    start = datetime.now()

    train_only = cfg.stage_cfg.get("train_only", False)
    split_fp = Path(cfg.input_dir) / "metadata" / "patient_split.parquet"

    shards, includes_only_train = shard_iterator_fntr(cfg)

    if train_only:
        if includes_only_train:
            logger.info(
                f"Processing train split only via shard prefix. Not filtering with {str(split_fp.resolve())}."
            )
        elif split_fp.exists():
            logger.info(f"Processing train split only by filtering read dfs via {str(split_fp.resolve())}")
            train_patients = (
                pl.scan_parquet(split_fp)
                .filter(pl.col("split") == "train")
                .select(pl.col("patient_id"))
                .collect()
                .to_list()
            )
            read_fn = read_and_filter_fntr(train_patients, read_fn)
        else:
            raise FileNotFoundError(
                f"Train split requested, but shard prefixes can't be used and "
                f"patient split file not found at {str(split_fp.resolve())}."
            )
    elif includes_only_train:
        raise ValueError("All splits should be used, but shard iterator is returning only train splits?!?")

    def fntr_params(compute_fn: ANY_COMPUTE_FN_T) -> ComputeFnArgs:
        compute_fn_params = inspect.signature(compute_fn).parameters
        kwargs = ComputeFnArgs()

        if "cfg" in compute_fn_params:
            kwargs["cfg"] = cfg
        if "stage_cfg" in compute_fn_params:
            kwargs["stage_cfg"] = cfg.stage_cfg
        if "code_modifiers" in compute_fn_params:
            code_modifiers = cfg.get("code_modifiers", None)
            kwargs["code_modifiers"] = code_modifiers
        if "code_metadata" in compute_fn_params:
            kwargs["code_metadata"] = pl.read_parquet(
                Path(cfg.stage_cfg.metadata_input_dir) / "codes.parquet", use_pyarrow=True
            )
        return kwargs

    if compute_fn is None:
        compute_fn = identity_fn
    match compute_fn_type(compute_fn):
        case ComputeFnType.DIRECT:
            pass
        case ComputeFnType.UNBOUND:
            compute_fn = partial(compute_fn, **fntr_params(compute_fn))
        case ComputeFnType.FUNCTOR:
            compute_fn = compute_fn(**fntr_params(compute_fn))
        case _:
            raise ValueError("Invalid compute function")

    all_out_fps = []
    for in_fp, out_fp in shards:
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
