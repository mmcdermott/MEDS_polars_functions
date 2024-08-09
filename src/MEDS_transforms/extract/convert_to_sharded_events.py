#!/usr/bin/env python
"""Utilities for converting input data structures into MEDS events."""

import copy
import json
import random
from collections.abc import Sequence
from functools import partial, reduce
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig

from MEDS_transforms.extract import CONFIG_YAML
from MEDS_transforms.extract.shard_events import META_KEYS
from MEDS_transforms.mapreduce.mapper import rwlock_wrap
from MEDS_transforms.utils import (
    is_col_field,
    parse_col_field,
    stage_init,
    write_lazyframe,
)


def in_format(fmt: str, ts_name: str) -> pl.Expr:
    """Returns an expression formatting the column ``ts_name`` in time format ``fmt``."""
    return pl.col(ts_name).str.strptime(pl.Datetime, fmt, strict=False)


def get_code_expr(code_field: str | list | ListConfig) -> tuple[pl.Expr, pl.Expr | None, set[str]]:
    """Converts the code field in an event config file to a polars expression, null filter, and column set.

    Args:
        code_field: The string or list representation of the code field in the event configuration file.

    Returns:
        pl.Expr: The polars expression representing the code field.
        pl.Expr | None: The null filter expression for the code field.
        set[str]: The set of columns needed to construct the code field.

    Raises:
        ValueError: If the code field is not a valid type.

    Examples:
        >>> print(*get_code_expr("A"))
        String(A).strict_cast(String) None set()
        >>> print(*get_code_expr("col(B)")) # doctest: +NORMALIZE_WHITESPACE
        col("B").strict_cast(String).fill_null([String(UNK)]) col("B").is_not_null() {'B'}
        >>> print(*get_code_expr(["col(A)", "B"])) # doctest: +NORMALIZE_WHITESPACE
        [([(col("A").strict_cast(String).fill_null([String(UNK)])) + (String(//))]) +
           (String(B).strict_cast(String))]
        col("A").is_not_null()
        {'A'}
        >>> get_code_expr(34)
        Traceback (most recent call last):
            ...
        ValueError: Invalid code field: 34
        >>> get_code_expr(["a", 34, "b"])
        Traceback (most recent call last):
            ...
        ValueError: Invalid code literal: 34

    Note that it only takes the first column field for the null filter, not all of them.
        >>> expr, null_filter, cols = get_code_expr(["col(A)", "col(c)"])
        >>> print(expr) # doctest: +NORMALIZE_WHITESPACE
        [([(col("A").strict_cast(String).fill_null([String(UNK)])) + (String(//))]) +
           (col("c").strict_cast(String).fill_null([String(UNK)]))]
        >>> print(null_filter)
        col("A").is_not_null()
        >>> print(sorted(cols))
        ['A', 'c']
    """
    if isinstance(code_field, str):
        code_field = [code_field]
    elif not isinstance(code_field, (list, ListConfig)):
        raise ValueError(f"Invalid code field: {code_field}")

    code_exprs = []
    code_null_filter_expr = None
    needed_cols = set()
    for i, code in enumerate(code_field):
        match code:
            case str() if is_col_field(code):
                code_col = parse_col_field(code)
                needed_cols.add(code_col)
                code_exprs.append(pl.col(code_col).cast(pl.Utf8).fill_null("UNK"))
                if i == 0:
                    code_null_filter_expr = pl.col(code_col).is_not_null()
            case str():
                code_exprs.append(pl.lit(code, dtype=pl.Utf8))
            case _:
                raise ValueError(f"Invalid code literal: {code}")
    code_expr = reduce(lambda a, b: a + pl.lit("//") + b, code_exprs)

    return code_expr, code_null_filter_expr, needed_cols


def extract_event(df: pl.LazyFrame, event_cfg: dict[str, str | None]) -> pl.LazyFrame:
    """Extracts a single event dataframe from the raw data.

    Args:
        df: The raw data DataFrame. This must have a `"patient_id"` column containing the patient ID. The
            other columns it must have are determined by the `event_cfg` configuration dictionary.
        event_cfg: A dictionary containing the configuration for the event. This must contain two critical
            keys (`"code"` and `"time"`) and may contain additional keys for other columns to include
            in the event DataFrame.
            The `"code"` key must contain either (1) a string literal representing the code for the event or
            (2) the name of a column in the raw data from which the code should be extracted. In the latter
            case, the column name should be enclosed in `col()` function call syntax--e.g.,
            `col(my_code_column)`. Note there are no quotes used inside the `col()` function syntax.
            The `"time"` key must contain either (1) the value `None` if the event has no time
            (e.g., a static event) or (2) the name of a column in the raw data from which the time should
            be extracted. In the latter case, the column name should be enclosed in `col()` function call
            syntax--e.g., `col(my_time_column)`. Note there are no quotes used inside the `col()`
            function syntax.
            If there is a "time_format" key in the `event_cfg` dictionary, the value of this key should
            be a string representing the format of the time column in the raw data. This format should
            conform to the `strftime` format codes. If this key is not present, the time column will be
            parsed as a datetime64 column.
            Any additional key/value pairs in the `event_cfg` dictionary will be interpreted as additional
            columns to extract for the output MEDS data, where the key corresponds to the MEDS column name and
            the value corresponds to the raw name (without any `col()` syntax) of the column in the raw data
            from which the MEDS column should be extracted. These columns must be either numeric or
            categorical (represented as either a `str` or a `Categorical` column in the raw data). Where
            possible, these additional columns should conform to the conventions of the MEDS data schema ---
            e.g., primary numeric values associated with the event should be named `"numeric_value"` in
            the output MEDS data (and thus have the key `"numeric_value"` in the `event_cfg` dictionary).

    Returns:
        A DataFrame containing the event data extracted from the raw data, containing only unique rows across
        all columns. If the raw data has no duplicates when considering the event column space, the output
        dataframe will have the same number of rows as the raw data and be in the same order. The output
        dataframe will contain at least three columns: `"patient_id"`, `"code"`, and `"time"`. If the
        event has additional columns, they will be included in the output dataframe as well. **_Events that
        would be extracted with a null code or a time that should be specified via a column with or
        without a formatting option but in practice is null will be dropped._** Note that this dropping logic
        for code list fields applies to columns in list order -- not to all columns. So, if you have a code
        that starts with a string literal or with a column that is not null, it will always appear, even if a
        subsequent list element is null, just with the missing code columns filled with "UNK".

    Raises:
        KeyError: If the event configuration dictionary is missing the `"code"` or `"time"` keys or if
            columns referenced by the event configuration dictionary are not found in the raw data.

    Examples:
        >>> _ = pl.Config.set_tbl_width_chars(600)
        >>> _ = pl.Config.set_tbl_rows(20)
        >>> _ = pl.Config.set_tbl_cols(20)
        >>> raw_data = pl.DataFrame({
        ...     "patient_id": [1, 1, 2, 2],
        ...     "code": ["A", "B", "C", "D"],
        ...     "code_modifier": ["1", "2", "3", "4"],
        ...     "time": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
        ...     "numeric_value": [1, 2, 3, 4],
        ... })
        >>> event_cfg = {
        ...     "code": ["FOO", "col(code)", "col(code_modifier)"],
        ...     "time": "col(time)",
        ...     "time_format": "%Y-%m-%d",
        ...     "numeric_value": "numeric_value",
        ... }
        >>> extract_event(raw_data, event_cfg)
        shape: (4, 4)
        ┌────────────┬───────────┬─────────────────────┬───────────────┐
        │ patient_id ┆ code      ┆ time                ┆ numeric_value │
        │ ---        ┆ ---       ┆ ---                 ┆ ---           │
        │ i64        ┆ str       ┆ datetime[μs]        ┆ i64           │
        ╞════════════╪═══════════╪═════════════════════╪═══════════════╡
        │ 1          ┆ FOO//A//1 ┆ 2021-01-01 00:00:00 ┆ 1             │
        │ 1          ┆ FOO//B//2 ┆ 2021-01-02 00:00:00 ┆ 2             │
        │ 2          ┆ FOO//C//3 ┆ 2021-01-03 00:00:00 ┆ 3             │
        │ 2          ┆ FOO//D//4 ┆ 2021-01-04 00:00:00 ┆ 4             │
        └────────────┴───────────┴─────────────────────┴───────────────┘
        >>> data_with_nulls = pl.DataFrame({
        ...     "patient_id": [1, 1, 2, 2],
        ...     "code": ["A", None, "C", "D"],
        ...     "code_modifier": ["1", "2", "3", None],
        ...     "time": [None, "2021-01-02", "2021-01-03", "2021-01-04"],
        ...     "numeric_value": [1, 2, 3, 4],
        ... })
        >>> event_cfg = {
        ...     "code": ["col(code)", "col(code_modifier)"],
        ...     "time": "col(time)",
        ...     "time_format": "%Y-%m-%d",
        ...     "numeric_value": "numeric_value",
        ... }
        >>> extract_event(data_with_nulls, event_cfg)
        shape: (2, 4)
        ┌────────────┬────────┬─────────────────────┬───────────────┐
        │ patient_id ┆ code   ┆ time                ┆ numeric_value │
        │ ---        ┆ ---    ┆ ---                 ┆ ---           │
        │ i64        ┆ str    ┆ datetime[μs]        ┆ i64           │
        ╞════════════╪════════╪═════════════════════╪═══════════════╡
        │ 2          ┆ C//3   ┆ 2021-01-03 00:00:00 ┆ 3             │
        │ 2          ┆ D//UNK ┆ 2021-01-04 00:00:00 ┆ 4             │
        └────────────┴────────┴─────────────────────┴───────────────┘
        >>> from datetime import datetime
        >>> complex_raw_data = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 2, 2, 2, 3],
        ...         "admission_time": [
        ...             "2021-01-01 00:00:00",
        ...             "2021-01-02 00:00:00",
        ...             "2021-01-03 00:00:00",
        ...             "2021-01-04 00:00:00",
        ...             "2021-01-05 00:00:00",
        ...             "2021-01-06 00:00:00",
        ...         ],
        ...         "discharge_time": [
        ...             datetime(2021, 1, 1, 11, 23, 45),
        ...             datetime(2021, 1, 2, 12, 34, 56),
        ...             datetime(2021, 1, 3, 13, 45, 56),
        ...             datetime(2021, 1, 4, 14, 56, 45),
        ...             datetime(2021, 1, 5, 15, 23, 45),
        ...             datetime(2021, 1, 6, 16, 34, 56),
        ...         ],
        ...         "admission_type": ["A", "B", "C", "D", "E", "F"],
        ...         "discharge_location": ["Home", "SNF", "Home", "SNF", "Home", "SNF"],
        ...         "discharge_status": ["AOx4", "AO", "AAO", "AOx3", "AOx4", "AOx4"],
        ...         "severity_score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ...         "death_time": [
        ...             "2023/01/01",
        ...             "2023/01/01",
        ...             "2023/01/04",
        ...             "2023/01/04",
        ...             "2023/01/04",
        ...             "2023/01/07"
        ...         ],
        ...         "eye_color": ["blue", "blue", "green", "green", "green", "brown"],
        ...     },
        ...     schema={
        ...         "patient_id": pl.UInt8,
        ...         "admission_time": pl.Utf8,
        ...         "discharge_time": pl.Datetime,
        ...         "admission_type": pl.Utf8,
        ...         "discharge_location": pl.Categorical,
        ...         "discharge_status": pl.Utf8,
        ...         "severity_score": pl.Float64,
        ...         "death_time": pl.Utf8,
        ...         "eye_color": pl.Categorical,
        ...     },
        ... )
        >>> valid_admission_event_cfg = {
        ...     "code": ["ADMISSION", "col(admission_type)"],
        ...     "time": "col(admission_time)",
        ...     "time_format": "%Y-%m-%d %H:%M:%S",
        ...     "numeric_value": "severity_score",
        ... }
        >>> valid_discharge_event_cfg = {
        ...     "code": ["DISCHARGE", "col(discharge_location)"],
        ...     "time": "col(discharge_time)",
        ...     "categorical_value": "discharge_status", # Note the raw dtype of this col is str
        ...     "text_value": "discharge_location", # Note the raw dtype of this col is categorical
        ... }
        >>> valid_death_event_cfg = {
        ...     "code": "DEATH",
        ...     "time": "col(death_time)",
        ...     "time_format": "%Y/%m/%d",
        ... }
        >>> valid_static_event_cfg = {
        ...     "code": ["EYE_COLOR", "col(eye_color)"],
        ...     "time": None,
        ... }
        >>> # We'll print the raw data so you can see what it looks like
        >>> complex_raw_data
        shape: (6, 9)
        ┌────────────┬─────────────────────┬─────────────────────┬────────────────┬────────────────────┬──────────────────┬────────────────┬────────────┬───────────┐
        │ patient_id ┆ admission_time      ┆ discharge_time      ┆ admission_type ┆ discharge_location ┆ discharge_status ┆ severity_score ┆ death_time ┆ eye_color │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---            ┆ ---                ┆ ---              ┆ ---            ┆ ---        ┆ ---       │
        │ u8         ┆ str                 ┆ datetime[μs]        ┆ str            ┆ cat                ┆ str              ┆ f64            ┆ str        ┆ cat       │
        ╞════════════╪═════════════════════╪═════════════════════╪════════════════╪════════════════════╪══════════════════╪════════════════╪════════════╪═══════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 2021-01-01 11:23:45 ┆ A              ┆ Home               ┆ AOx4             ┆ 1.0            ┆ 2023/01/01 ┆ blue      │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 2021-01-02 12:34:56 ┆ B              ┆ SNF                ┆ AO               ┆ 2.0            ┆ 2023/01/01 ┆ blue      │
        │ 2          ┆ 2021-01-03 00:00:00 ┆ 2021-01-03 13:45:56 ┆ C              ┆ Home               ┆ AAO              ┆ 3.0            ┆ 2023/01/04 ┆ green     │
        │ 2          ┆ 2021-01-04 00:00:00 ┆ 2021-01-04 14:56:45 ┆ D              ┆ SNF                ┆ AOx3             ┆ 4.0            ┆ 2023/01/04 ┆ green     │
        │ 2          ┆ 2021-01-05 00:00:00 ┆ 2021-01-05 15:23:45 ┆ E              ┆ Home               ┆ AOx4             ┆ 5.0            ┆ 2023/01/04 ┆ green     │
        │ 3          ┆ 2021-01-06 00:00:00 ┆ 2021-01-06 16:34:56 ┆ F              ┆ SNF                ┆ AOx4             ┆ 6.0            ┆ 2023/01/07 ┆ brown     │
        └────────────┴─────────────────────┴─────────────────────┴────────────────┴────────────────────┴──────────────────┴────────────────┴────────────┴───────────┘
        >>> extract_event(complex_raw_data, valid_admission_event_cfg)
        shape: (6, 4)
        ┌────────────┬──────────────┬─────────────────────┬───────────────┐
        │ patient_id ┆ code         ┆ time                ┆ numeric_value │
        │ ---        ┆ ---          ┆ ---                 ┆ ---           │
        │ u8         ┆ str          ┆ datetime[μs]        ┆ f64           │
        ╞════════════╪══════════════╪═════════════════════╪═══════════════╡
        │ 1          ┆ ADMISSION//A ┆ 2021-01-01 00:00:00 ┆ 1.0           │
        │ 1          ┆ ADMISSION//B ┆ 2021-01-02 00:00:00 ┆ 2.0           │
        │ 2          ┆ ADMISSION//C ┆ 2021-01-03 00:00:00 ┆ 3.0           │
        │ 2          ┆ ADMISSION//D ┆ 2021-01-04 00:00:00 ┆ 4.0           │
        │ 2          ┆ ADMISSION//E ┆ 2021-01-05 00:00:00 ┆ 5.0           │
        │ 3          ┆ ADMISSION//F ┆ 2021-01-06 00:00:00 ┆ 6.0           │
        └────────────┴──────────────┴─────────────────────┴───────────────┘
        >>> extract_event(
        ...     complex_raw_data.with_columns(pl.col("severity_score").cast(pl.Utf8)),
        ...     valid_admission_event_cfg
        ... )
        shape: (6, 4)
        ┌────────────┬──────────────┬─────────────────────┬───────────────┐
        │ patient_id ┆ code         ┆ time                ┆ numeric_value │
        │ ---        ┆ ---          ┆ ---                 ┆ ---           │
        │ u8         ┆ str          ┆ datetime[μs]        ┆ f64           │
        ╞════════════╪══════════════╪═════════════════════╪═══════════════╡
        │ 1          ┆ ADMISSION//A ┆ 2021-01-01 00:00:00 ┆ 1.0           │
        │ 1          ┆ ADMISSION//B ┆ 2021-01-02 00:00:00 ┆ 2.0           │
        │ 2          ┆ ADMISSION//C ┆ 2021-01-03 00:00:00 ┆ 3.0           │
        │ 2          ┆ ADMISSION//D ┆ 2021-01-04 00:00:00 ┆ 4.0           │
        │ 2          ┆ ADMISSION//E ┆ 2021-01-05 00:00:00 ┆ 5.0           │
        │ 3          ┆ ADMISSION//F ┆ 2021-01-06 00:00:00 ┆ 6.0           │
        └────────────┴──────────────┴─────────────────────┴───────────────┘
        >>> extract_event(
        ...     complex_raw_data.with_columns(pl.col("severity_score").cast(pl.Utf8).cast(pl.Categorical)),
        ...     valid_admission_event_cfg
        ... )
        shape: (6, 4)
        ┌────────────┬──────────────┬─────────────────────┬───────────────┐
        │ patient_id ┆ code         ┆ time                ┆ numeric_value │
        │ ---        ┆ ---          ┆ ---                 ┆ ---           │
        │ u8         ┆ str          ┆ datetime[μs]        ┆ f64           │
        ╞════════════╪══════════════╪═════════════════════╪═══════════════╡
        │ 1          ┆ ADMISSION//A ┆ 2021-01-01 00:00:00 ┆ 1.0           │
        │ 1          ┆ ADMISSION//B ┆ 2021-01-02 00:00:00 ┆ 2.0           │
        │ 2          ┆ ADMISSION//C ┆ 2021-01-03 00:00:00 ┆ 3.0           │
        │ 2          ┆ ADMISSION//D ┆ 2021-01-04 00:00:00 ┆ 4.0           │
        │ 2          ┆ ADMISSION//E ┆ 2021-01-05 00:00:00 ┆ 5.0           │
        │ 3          ┆ ADMISSION//F ┆ 2021-01-06 00:00:00 ┆ 6.0           │
        └────────────┴──────────────┴─────────────────────┴───────────────┘
        >>> extract_event(complex_raw_data, valid_discharge_event_cfg)
        shape: (6, 5)
        ┌────────────┬─────────────────┬─────────────────────┬───────────────────┬────────────┐
        │ patient_id ┆ code            ┆ time                ┆ categorical_value ┆ text_value │
        │ ---        ┆ ---             ┆ ---                 ┆ ---               ┆ ---        │
        │ u8         ┆ str             ┆ datetime[μs]        ┆ str               ┆ str        │
        ╞════════════╪═════════════════╪═════════════════════╪═══════════════════╪════════════╡
        │ 1          ┆ DISCHARGE//Home ┆ 2021-01-01 11:23:45 ┆ AOx4              ┆ Home       │
        │ 1          ┆ DISCHARGE//SNF  ┆ 2021-01-02 12:34:56 ┆ AO                ┆ SNF        │
        │ 2          ┆ DISCHARGE//Home ┆ 2021-01-03 13:45:56 ┆ AAO               ┆ Home       │
        │ 2          ┆ DISCHARGE//SNF  ┆ 2021-01-04 14:56:45 ┆ AOx3              ┆ SNF        │
        │ 2          ┆ DISCHARGE//Home ┆ 2021-01-05 15:23:45 ┆ AOx4              ┆ Home       │
        │ 3          ┆ DISCHARGE//SNF  ┆ 2021-01-06 16:34:56 ┆ AOx4              ┆ SNF        │
        └────────────┴─────────────────┴─────────────────────┴───────────────────┴────────────┘
        >>> extract_event(complex_raw_data, valid_death_event_cfg)
        shape: (3, 3)
        ┌────────────┬───────┬─────────────────────┐
        │ patient_id ┆ code  ┆ time                │
        │ ---        ┆ ---   ┆ ---                 │
        │ u8         ┆ str   ┆ datetime[μs]        │
        ╞════════════╪═══════╪═════════════════════╡
        │ 1          ┆ DEATH ┆ 2023-01-01 00:00:00 │
        │ 2          ┆ DEATH ┆ 2023-01-04 00:00:00 │
        │ 3          ┆ DEATH ┆ 2023-01-07 00:00:00 │
        └────────────┴───────┴─────────────────────┘
        >>> # Note that the eye color is a static event, so the time is null
        >>> extract_event(complex_raw_data, valid_static_event_cfg)
        shape: (3, 3)
        ┌────────────┬──────────────────┬──────────────┐
        │ patient_id ┆ code             ┆ time         │
        │ ---        ┆ ---              ┆ ---          │
        │ u8         ┆ str              ┆ datetime[μs] │
        ╞════════════╪══════════════════╪══════════════╡
        │ 1          ┆ EYE_COLOR//blue  ┆ null         │
        │ 2          ┆ EYE_COLOR//green ┆ null         │
        │ 3          ┆ EYE_COLOR//brown ┆ null         │
        └────────────┴──────────────────┴──────────────┘
        >>> extract_event(complex_raw_data, {"time": "col(admission_time)"})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain 'code' key. Got: [time]."
        >>> extract_event(complex_raw_data, {"code": "test", "value": "severity_score"})
        Traceback (most recent call last):
            ..".
        KeyError: "Event configuration dictionary must contain 'time' key. Got: [code, value]."
        >>> extract_event(complex_raw_data, {"code": "test", "time": "12-01-23"})
        Traceback (most recent call last):
            ...
        ValueError: Invalid time literal: 12-01-23
        >>> extract_event(complex_raw_data, {"code": "test", "time": None, "patient_id": 3})
        Traceback (most recent call last):
            ...
        KeyError: "Event column name 'patient_id' cannot be overridden."
        >>> extract_event(complex_raw_data, {"code": "test", "time": None, "foobar": "fuzz"})
        Traceback (most recent call last):
            ...
        KeyError: "Source column 'fuzz' for event column foobar not found in DataFrame schema."
        >>> extract_event(complex_raw_data, {"code": "test", "time": None, "foobar": 32})
        Traceback (most recent call last):
            ...
        ValueError: For event column foobar, source column 32 must be a string column name. Got <class 'int'>.
        >>> extract_event(complex_raw_data, {"code": "test", "time": None, "foobar": "discharge_time"})
        Traceback (most recent call last):
            ...
        ValueError: Source column 'discharge_time' for event column foobar is not numeric, string, or categorical! Cannot be used as an event col.
    """  # noqa: E501
    event_cfg = copy.deepcopy(event_cfg)
    event_exprs = {"patient_id": pl.col("patient_id")}

    if "code" not in event_cfg:
        raise KeyError(
            "Event configuration dictionary must contain 'code' key. "
            f"Got: [{', '.join(event_cfg.keys())}]."
        )
    if "time" not in event_cfg:
        raise KeyError(
            "Event configuration dictionary must contain 'time' key. "
            f"Got: [{', '.join(event_cfg.keys())}]."
        )
    if "patient_id" in event_cfg:
        raise KeyError("Event column name 'patient_id' cannot be overridden.")

    code_expr, code_null_filter_expr, needed_cols = get_code_expr(event_cfg.pop("code"))

    for col in needed_cols:
        if col not in df.schema:
            raise KeyError(f"Source column '{col}' for event column code not found in DataFrame schema.")
        logger.info(f"Extracting column {col}")

    event_exprs["code"] = code_expr

    ts = event_cfg.pop("time")
    ts_format = event_cfg.pop("time_format", None)
    if isinstance(ts_format, str):
        ts_format = [ts_format]

    ts_filter_expr = None
    match ts:
        case str() if is_col_field(ts):
            ts_name = parse_col_field(ts)
            if isinstance(ts_format, (ListConfig, list)):
                logger.info(f"Adding time column {ts_name} in possible formats {', '.join(ts_format)}")
                assert len(ts_format) > 0, "Time format list is empty"
                event_exprs["time"] = pl.coalesce(*(in_format(fmt, ts_name) for fmt in ts_format))
            else:
                logger.info(f"{ts_name} should already be in Date/time format")
                assert ts_format is None
                event_exprs["time"] = pl.col(ts_name).cast(pl.Datetime)
            ts_filter_expr = event_exprs["time"].is_not_null()
        case None:
            logger.info("Adding null literate for time")
            event_exprs["time"] = pl.lit(None, dtype=pl.Datetime)
        case _:
            raise ValueError(f"Invalid time literal: {ts}")

    for k, v in event_cfg.items():
        if k in META_KEYS:
            continue

        if not isinstance(v, str):
            raise ValueError(
                f"For event column {k}, source column {v} must be a string column name. Got {type(v)}."
            )
        elif is_col_field(v):
            logger.warning(
                f"Source column '{v}' for event column {k} is always interpreted as a column name. "
                f"Removing col() function call and setting source column to {parse_col_field(v)}."
            )
            v = parse_col_field(v)

        if v not in df.schema:
            raise KeyError(f"Source column '{v}' for event column {k} not found in DataFrame schema.")

        col = pl.col(v)
        is_numeric = df.schema[v].is_numeric()
        is_str = df.schema[v] == pl.Utf8
        is_cat = isinstance(df.schema[v], pl.Categorical)
        match k:
            case "numeric_value" if is_numeric:
                pass
            case "numeric_value" if is_str:
                logger.warning(f"Converting numeric_value to float from string for {code_expr}")
                col = col.cast(pl.Float64, strict=False)
            case "numeric_value" if is_cat:
                logger.warning(f"Converting numeric_value to float from categorical for {code_expr}")
                col = col.cast(pl.Utf8).cast(pl.Float64, strict=False)
            case "text_value" if not df.schema[v] == pl.Utf8:
                logger.warning(f"Converting text_value to string for {code_expr}")
                col = col.cast(pl.Utf8, strict=False)
            case "categorical_value" if not is_str:
                logger.warning(f"Converting categorical_value to string for {code_expr}")
                col = col.cast(pl.Utf8)
            case _ if is_str:
                pass
            case _ if not (is_numeric or is_str or is_cat):
                raise ValueError(
                    f"Source column '{v}' for event column {k} is not numeric, string, or categorical! "
                    "Cannot be used as an event col."
                )

        event_exprs[k] = col

    if code_null_filter_expr is not None:
        logger.info(f"Filtering out rows with null codes via {code_null_filter_expr}")
        df = df.filter(code_null_filter_expr)
    if ts_filter_expr is not None:
        logger.info(f"Filtering out rows with null times via {ts_filter_expr}")
        df = df.filter(ts_filter_expr)

    df = df.select(**event_exprs).unique(maintain_order=True)

    return df


def convert_to_events(
    df: pl.LazyFrame, event_cfgs: dict[str, dict[str, str | None | Sequence[str]]]
) -> pl.LazyFrame:
    """Converts a DataFrame of raw data into a DataFrame of events.

    Args:
        df: The raw data DataFrame. This must have a `"patient_id"` column containing the patient ID. The
            other columns it must have are determined by the `event_cfgs` configuration dictionary.
            For the precise mechanism of column determination, see the `extract_event` function.
        event_cfgs: A dictionary containing the configurations for the events to extract. The keys of this
            dictionary should be the names of the events to extract (these are only used for logging, and will
            not automatically appear in any manner in the output data), and the values should be dictionaries
            containing the configuration for each event. Each event configuration dictionary should have the
            same structure as the `event_cfg` dictionary in the `extract_event` function. Please see that
            function for further details on how these configuration dictionaries should be structured and are
            used to extract events.

    Returns:
        A DataFrame containing the events extracted from the raw data. This DataFrame will contain all the
        events extracted from the raw data, with the rows from each event DataFrame concatenated together.
        After concatenation, this dataframe will not be deduplicated, so if the raw data results in duplicates
        across events of different name, these will be preserved in the output DataFrame.
        The output DataFrame will contain at least three columns: `"patient_id"`, `"code"`, and `"time"`.
        If any events have additional columns, these will be included in the output DataFrame as well. All
        columns across all event configurations will be included in the output DataFrame, with `null` values
        filled in for events that do not have a particular column.

    Raises:
        ValueError: If no event configurations are provided or if an error occurs during event extraction.

    Examples:
        >>> _ = pl.Config.set_tbl_width_chars(600)
        >>> _ = pl.Config.set_tbl_rows(20)
        >>> _ = pl.Config.set_tbl_cols(20)
        >>> from datetime import datetime
        >>> complex_raw_data = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 2, 2, 2, 3],
        ...         "admission_time": [
        ...             "2021-01-01 00:00:00",
        ...             "2021-01-02 00:00:00",
        ...             "2021-01-03 00:00:00",
        ...             "2021-01-04 00:00:00",
        ...             "2021-01-05 00:00:00",
        ...             "2021-01-06 00:00:00",
        ...         ],
        ...         "discharge_time": [
        ...             datetime(2021, 1, 1, 11, 23, 45),
        ...             datetime(2021, 1, 2, 12, 34, 56),
        ...             datetime(2021, 1, 3, 13, 45, 56),
        ...             datetime(2021, 1, 4, 14, 56, 45),
        ...             datetime(2021, 1, 5, 15, 23, 45),
        ...             datetime(2021, 1, 6, 16, 34, 56),
        ...         ],
        ...         "admission_type": ["A", "B", "C", "D", "E", "F"],
        ...         "discharge_location": ["Home", "SNF", "Home", "SNF", "Home", "SNF"],
        ...         "severity_score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ...         "death_time": [
        ...             "2023/01/01",
        ...             "2023/01/01",
        ...             "2023/01/04",
        ...             "2023/01/04",
        ...             "2023/01/04",
        ...             "2023/01/07"
        ...         ],
        ...         "eye_color": ["blue", "blue", "green", "green", "green", "brown"],
        ...     },
        ...     schema={
        ...         "patient_id": pl.UInt8,
        ...         "admission_time": pl.Utf8,
        ...         "discharge_time": pl.Datetime,
        ...         "admission_type": pl.Utf8,
        ...         "discharge_location": pl.Categorical,
        ...         "severity_score": pl.Float64,
        ...         "death_time": pl.Utf8,
        ...         "eye_color": pl.Categorical,
        ...     },
        ... )
        >>> event_cfgs = {
        ...     "admission": {
        ...         "code": "ADMISSION",
        ...         "time": "col(admission_time)",
        ...         "time_format": "%Y-%m-%d %H:%M:%S",
        ...         "admission_type": "admission_type",
        ...         "severity_on_admission": "severity_score",
        ...     },
        ...     "discharge": {
        ...         "code": "DISCHARGE",
        ...         "time": "col(discharge_time)",
        ...         "discharge_location": "discharge_location",
        ...     },
        ...     "death": {
        ...         "code": "DEATH",
        ...         "time": "col(death_time)",
        ...         "time_format": "%Y/%m/%d",
        ...     },
        ...     "eye_color": {
        ...         "code": "EYE_COLOR",
        ...         "time": None,
        ...         "eye_color": "eye_color",
        ...     },
        ... }
        >>> # We'll print the raw data so you can see what it looks like
        >>> complex_raw_data
        shape: (6, 8)
        ┌────────────┬─────────────────────┬─────────────────────┬────────────────┬────────────────────┬────────────────┬────────────┬───────────┐
        │ patient_id ┆ admission_time      ┆ discharge_time      ┆ admission_type ┆ discharge_location ┆ severity_score ┆ death_time ┆ eye_color │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---            ┆ ---                ┆ ---            ┆ ---        ┆ ---       │
        │ u8         ┆ str                 ┆ datetime[μs]        ┆ str            ┆ cat                ┆ f64            ┆ str        ┆ cat       │
        ╞════════════╪═════════════════════╪═════════════════════╪════════════════╪════════════════════╪════════════════╪════════════╪═══════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 2021-01-01 11:23:45 ┆ A              ┆ Home               ┆ 1.0            ┆ 2023/01/01 ┆ blue      │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 2021-01-02 12:34:56 ┆ B              ┆ SNF                ┆ 2.0            ┆ 2023/01/01 ┆ blue      │
        │ 2          ┆ 2021-01-03 00:00:00 ┆ 2021-01-03 13:45:56 ┆ C              ┆ Home               ┆ 3.0            ┆ 2023/01/04 ┆ green     │
        │ 2          ┆ 2021-01-04 00:00:00 ┆ 2021-01-04 14:56:45 ┆ D              ┆ SNF                ┆ 4.0            ┆ 2023/01/04 ┆ green     │
        │ 2          ┆ 2021-01-05 00:00:00 ┆ 2021-01-05 15:23:45 ┆ E              ┆ Home               ┆ 5.0            ┆ 2023/01/04 ┆ green     │
        │ 3          ┆ 2021-01-06 00:00:00 ┆ 2021-01-06 16:34:56 ┆ F              ┆ SNF                ┆ 6.0            ┆ 2023/01/07 ┆ brown     │
        └────────────┴─────────────────────┴─────────────────────┴────────────────┴────────────────────┴────────────────┴────────────┴───────────┘
        >>> convert_to_events(complex_raw_data, event_cfgs)
        shape: (18, 7)
        ┌────────────┬───────────┬─────────────────────┬────────────────┬───────────────────────┬────────────────────┬───────────┐
        │ patient_id ┆ code      ┆ time                ┆ admission_type ┆ severity_on_admission ┆ discharge_location ┆ eye_color │
        │ ---        ┆ ---       ┆ ---                 ┆ ---            ┆ ---                   ┆ ---                ┆ ---       │
        │ u8         ┆ str       ┆ datetime[μs]        ┆ str            ┆ f64                   ┆ cat                ┆ cat       │
        ╞════════════╪═══════════╪═════════════════════╪════════════════╪═══════════════════════╪════════════════════╪═══════════╡
        │ 1          ┆ ADMISSION ┆ 2021-01-01 00:00:00 ┆ A              ┆ 1.0                   ┆ null               ┆ null      │
        │ 1          ┆ ADMISSION ┆ 2021-01-02 00:00:00 ┆ B              ┆ 2.0                   ┆ null               ┆ null      │
        │ 2          ┆ ADMISSION ┆ 2021-01-03 00:00:00 ┆ C              ┆ 3.0                   ┆ null               ┆ null      │
        │ 2          ┆ ADMISSION ┆ 2021-01-04 00:00:00 ┆ D              ┆ 4.0                   ┆ null               ┆ null      │
        │ 2          ┆ ADMISSION ┆ 2021-01-05 00:00:00 ┆ E              ┆ 5.0                   ┆ null               ┆ null      │
        │ 3          ┆ ADMISSION ┆ 2021-01-06 00:00:00 ┆ F              ┆ 6.0                   ┆ null               ┆ null      │
        │ 1          ┆ DISCHARGE ┆ 2021-01-01 11:23:45 ┆ null           ┆ null                  ┆ Home               ┆ null      │
        │ 1          ┆ DISCHARGE ┆ 2021-01-02 12:34:56 ┆ null           ┆ null                  ┆ SNF                ┆ null      │
        │ 2          ┆ DISCHARGE ┆ 2021-01-03 13:45:56 ┆ null           ┆ null                  ┆ Home               ┆ null      │
        │ 2          ┆ DISCHARGE ┆ 2021-01-04 14:56:45 ┆ null           ┆ null                  ┆ SNF                ┆ null      │
        │ 2          ┆ DISCHARGE ┆ 2021-01-05 15:23:45 ┆ null           ┆ null                  ┆ Home               ┆ null      │
        │ 3          ┆ DISCHARGE ┆ 2021-01-06 16:34:56 ┆ null           ┆ null                  ┆ SNF                ┆ null      │
        │ 1          ┆ DEATH     ┆ 2023-01-01 00:00:00 ┆ null           ┆ null                  ┆ null               ┆ null      │
        │ 2          ┆ DEATH     ┆ 2023-01-04 00:00:00 ┆ null           ┆ null                  ┆ null               ┆ null      │
        │ 3          ┆ DEATH     ┆ 2023-01-07 00:00:00 ┆ null           ┆ null                  ┆ null               ┆ null      │
        │ 1          ┆ EYE_COLOR ┆ null                ┆ null           ┆ null                  ┆ null               ┆ blue      │
        │ 2          ┆ EYE_COLOR ┆ null                ┆ null           ┆ null                  ┆ null               ┆ green     │
        │ 3          ┆ EYE_COLOR ┆ null                ┆ null           ┆ null                  ┆ null               ┆ brown     │
        └────────────┴───────────┴─────────────────────┴────────────────┴───────────────────────┴────────────────────┴───────────┘
        >>> convert_to_events(complex_raw_data, {})
        Traceback (most recent call last):
            ...
        ValueError: No event configurations provided.
        >>> convert_to_events(complex_raw_data, {"admission": {}})
        Traceback (most recent call last):
            ...
        ValueError: Error extracting event admission: ...
    """  # noqa: E501

    if not event_cfgs:
        raise ValueError("No event configurations provided.")

    event_dfs = []
    for event_name, event_cfg in event_cfgs.items():
        try:
            logger.info(f"Building computational graph for extracting {event_name}")
            event_dfs.append(extract_event(df, event_cfg))
        except Exception as e:
            raise ValueError(f"Error extracting event {event_name}: {e}") from e

    df = pl.concat(event_dfs, how="diagonal_relaxed")
    return df


@hydra.main(version_base=None, config_path=str(CONFIG_YAML.parent), config_name=CONFIG_YAML.stem)
def main(cfg: DictConfig):
    """Converts the event-sharded raw data into MEDS events and storing them in patient subsharded flat files.

    All arguments are specified through the command line into the `cfg` object through Hydra.

    The `cfg.stage_cfg` object is a special key that is imputed by OmegaConf to contain the stage-specific
    configuration arguments based on the global, pipeline-level configuration file. It cannot be overwritten
    directly on the command line, but can be overwritten implicitly by overwriting components of the
    `stage_configs.convert_to_sharded_events` key.

    This stage has no stage-specific configuration arguments. It does, naturally, require the global,
    `event_conversion_config_fp` configuration argument to be set to the path of the event conversion yaml
    file.
    """

    input_dir, patient_subsharded_dir, metadata_input_dir = stage_init(cfg)

    shards = json.loads(Path(cfg.shards_map_fp).read_text())

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")

    logger.info("Starting event conversion.")

    logger.info(f"Reading event conversion config from {event_conversion_cfg_fp}")
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    logger.info(f"Event conversion config:\n{OmegaConf.to_yaml(event_conversion_cfg)}")

    default_patient_id_col = event_conversion_cfg.pop("patient_id_col", "patient_id")

    patient_subsharded_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(event_conversion_cfg, patient_subsharded_dir / "event_conversion_config.yaml")

    patient_splits = list(shards.items())
    random.shuffle(patient_splits)

    event_configs = list(event_conversion_cfg.items())
    random.shuffle(event_configs)

    # Here, we'll be reading files directly, so we'll turn off globbing
    read_fn = partial(pl.scan_parquet, glob=False)

    for sp, patients in patient_splits:
        for input_prefix, event_cfgs in event_configs:
            event_cfgs = copy.deepcopy(event_cfgs)
            input_patient_id_column = event_cfgs.pop("patient_id_col", default_patient_id_col)

            event_shards = list((input_dir / input_prefix).glob("*.parquet"))
            random.shuffle(event_shards)

            for shard_fp in event_shards:
                out_fp = patient_subsharded_dir / sp / input_prefix / shard_fp.name
                logger.info(f"Converting {shard_fp} to events and saving to {out_fp}")

                def compute_fn(df: pl.LazyFrame) -> pl.LazyFrame:
                    typed_patients = pl.Series(patients, dtype=df.schema[input_patient_id_column])

                    if input_patient_id_column != "patient_id":
                        df = df.rename({input_patient_id_column: "patient_id"})

                    try:
                        logger.info(f"Extracting events for {input_prefix}/{shard_fp.name}")
                        return convert_to_events(
                            df.filter(pl.col("patient_id").is_in(typed_patients)),
                            event_cfgs=copy.deepcopy(event_cfgs),
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Error converting {str(shard_fp.resolve())} for {sp}/{input_prefix}: {e}"
                        ) from e

                rwlock_wrap(
                    shard_fp, out_fp, read_fn, write_lazyframe, compute_fn, do_overwrite=cfg.do_overwrite
                )

    logger.info("Subsharded into converted events.")


if __name__ == "__main__":
    main()
