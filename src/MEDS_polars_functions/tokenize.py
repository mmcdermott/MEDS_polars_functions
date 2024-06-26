"""Functions for tokenizing MEDS datasets.

Here, _tokenization_ refers specifically to the process of converting a longitudinal, irregularly sampled,
continuous time sequence into a temporal sequence at the level that will be consumed by deep-learning models.

All these functions take in _normalized_ data -- meaning data where there are _no longer_ any code modifiers,
as those have been normalized alongside codes into integer indices (in the output code column). The only
columns of concern here thus are `patient_id`, `timestamp`, `code`, `numerical_value`.
"""

import polars as pl

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60.0
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24.0


def fill_to_nans(col: str | pl.Expr) -> pl.Expr:
    """This function fills infinite and null values with NaN.

    This enables the downstream functions to naturally tensorize data into numpy or Torch tensors.

    Args:
        col: The input column.

    Returns:
        A `pl.Expr` object that fills infinite and null values with NaN.

    Examples:
        >>> print(fill_to_nans("value")) # doctest: +NORMALIZE_WHITESPACE
        .when([(col("value").is_infinite()) |
               (col("value").is_null())]).then(dyn float: NaN).otherwise(col("value"))
        >>> print(fill_to_nans(pl.col("time_delta"))) # doctest: +NORMALIZE_WHITESPACE
        .when([(col("time_delta").is_infinite()) |
               (col("time_delta").is_null())]).then(dyn float: NaN).otherwise(col("time_delta"))
        >>> df = pl.DataFrame({"value": [1.0, float("inf"), None, -float("inf"), 2.0]})
        >>> df.select(fill_to_nans("value").alias("value"))["value"].to_list()
        [1.0, nan, nan, nan, 2.0]
    """

    if isinstance(col, str):
        col = pl.col(col)

    return pl.when(col.is_infinite() | col.is_null()).then(float("nan")).otherwise(col)


def split_static_and_dynamic(df: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """This function splits the input data into static and dynamic data.

    Static data is data that has a null timestamp, and dynamic data is everything else.

    Args:
        df: The input data.

    Returns:
        A tuple of two `pl.LazyFrame` objects, the first being the static data and the second being the
        dynamic data.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 2, 2],
        ...     "timestamp": [None, datetime(2021, 1, 1), None, datetime(2021, 1, 2)],
        ...     "code": [100, 101, 200, 201],
        ...     "numerical_value": [1.0, 2.0, 3.0, 4.0]
        ... }).lazy()
        >>> static, dynamic = split_static_and_dynamic(df)
        >>> static.collect()
        shape: (2, 3)
        ┌────────────┬──────┬─────────────────┐
        │ patient_id ┆ code ┆ numerical_value │
        │ ---        ┆ ---  ┆ ---             │
        │ i64        ┆ i64  ┆ f64             │
        ╞════════════╪══════╪═════════════════╡
        │ 1          ┆ 100  ┆ 1.0             │
        │ 2          ┆ 200  ┆ 3.0             │
        └────────────┴──────┴─────────────────┘
        >>> dynamic.collect()
        shape: (2, 4)
        ┌────────────┬─────────────────────┬──────┬─────────────────┐
        │ patient_id ┆ timestamp           ┆ code ┆ numerical_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---             │
        │ i64        ┆ datetime[μs]        ┆ i64  ┆ f64             │
        ╞════════════╪═════════════════════╪══════╪═════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 101  ┆ 2.0             │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 201  ┆ 4.0             │
        └────────────┴─────────────────────┴──────┴─────────────────┘
    """

    static = df.filter(pl.col("timestamp").is_null()).drop("timestamp")
    dynamic = df.filter(pl.col("timestamp").is_not_null())
    return static, dynamic


def extract_statics_and_schema(df: pl.LazyFrame) -> pl.LazyFrame:
    """This function extracts static data and schema information (sequence of patient unique timestamps).

    Args:
        df: The input data.

    Returns:
        A `pl.LazyFrame` object containing the static data and the unique timestamps of the patient, grouped
        by patient as lists, in the same order as the patient IDs occurred in the original file.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 1, 2, 2, 2],
        ...     "timestamp": [
        ...         None, datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 13),
        ...         None, datetime(2021, 1, 2), datetime(2021, 1, 2)],
        ...     "code": [100, 101, 102, 103, 200, 201, 202],
        ...     "numerical_value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ... }).lazy()
        >>> df = extract_statics_and_schema(df).collect()
        >>> df.drop("timestamp")
        shape: (2, 4)
        ┌────────────┬───────────┬─────────────────┬─────────────────────┐
        │ patient_id ┆ code      ┆ numerical_value ┆ start_time          │
        │ ---        ┆ ---       ┆ ---             ┆ ---                 │
        │ i64        ┆ list[i64] ┆ list[f64]       ┆ datetime[μs]        │
        ╞════════════╪═══════════╪═════════════════╪═════════════════════╡
        │ 1          ┆ [100]     ┆ [1.0]           ┆ 2021-01-01 00:00:00 │
        │ 2          ┆ [200]     ┆ [5.0]           ┆ 2021-01-02 00:00:00 │
        └────────────┴───────────┴─────────────────┴─────────────────────┘
        >>> df.select("patient_id", "timestamp").explode("timestamp")
        shape: (3, 2)
        ┌────────────┬─────────────────────┐
        │ patient_id ┆ timestamp           │
        │ ---        ┆ ---                 │
        │ i64        ┆ datetime[μs]        │
        ╞════════════╪═════════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 │
        │ 1          ┆ 2021-01-13 00:00:00 │
        │ 2          ┆ 2021-01-02 00:00:00 │
        └────────────┴─────────────────────┘
    """

    static, dynamic = split_static_and_dynamic(df)

    # This collects static data by patient ID and stores only (as a list) the codes and numerical values.
    static_by_patient = static.group_by("patient_id", maintain_order=True).agg("code", "numerical_value")

    # This collects the unique timestamps for each patient.
    schema_by_patient = dynamic.group_by("patient_id", maintain_order=True).agg(
        pl.col("timestamp").min().alias("start_time"), pl.col("timestamp").unique(maintain_order=True)
    )

    # TODO(mmd): Consider tracking patient offset explicitly here.

    return static_by_patient.join(schema_by_patient, on="patient_id", how="inner")


def extract_seq_of_patient_events(df: pl.LazyFrame) -> pl.LazyFrame:
    """This function extracts sequences of patient events, which are sequences of measurements.

    The result of this can be naturally tensorized into a `JointNestedRaggedTensorDict` object.

    Args:
        df: The input data.

    Returns:
        A `pl.LazyFrame` object containing the sequences of patient events, with the following columns:
            - `patient_id`: The patient ID.
            - `time_delta_days`: The time delta in days, as a list of floats (ragged).
            - `code`: The code, as a list of lists of ints (ragged in both levels).
            - `numerical_value`: The numerical value as a list of lists of floats (ragged in both levels).

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 1, 2, 2, 2],
        ...     "timestamp": [
        ...         None, datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 13),
        ...         None, datetime(2021, 1, 2), datetime(2021, 1, 2)],
        ...     "code": [100, 101, 102, 103, 200, 201, 202],
        ...     "numerical_value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ... }).lazy()
        >>> extract_seq_of_patient_events(df).collect()
        shape: (2, 4)
        ┌────────────┬─────────────────┬───────────────────────────┬─────────────────────┐
        │ patient_id ┆ time_delta_days ┆ code                      ┆ numerical_value     │
        │ ---        ┆ ---             ┆ ---                       ┆ ---                 │
        │ i64        ┆ list[f64]       ┆ list[list[f64]]           ┆ list[list[f64]]     │
        ╞════════════╪═════════════════╪═══════════════════════════╪═════════════════════╡
        │ 1          ┆ [NaN, 12.0]     ┆ [[101.0, 102.0], [103.0]] ┆ [[2.0, 3.0], [4.0]] │
        │ 2          ┆ [NaN]           ┆ [[201.0, 202.0]]          ┆ [[6.0, 7.0]]        │
        └────────────┴─────────────────┴───────────────────────────┴─────────────────────┘
    """

    _, dynamic = split_static_and_dynamic(df)

    time_delta_days_expr = (pl.col("timestamp").diff().dt.total_seconds() / SECONDS_PER_DAY).cast(pl.Float64)

    return (
        dynamic.group_by("patient_id", "timestamp", maintain_order=True)
        .agg(fill_to_nans("code").keep_name(), fill_to_nans("numerical_value").keep_name())
        .group_by("patient_id", maintain_order=True)
        .agg(
            fill_to_nans(time_delta_days_expr).alias("time_delta_days"),
            "code",
            "numerical_value",
        )
    )
