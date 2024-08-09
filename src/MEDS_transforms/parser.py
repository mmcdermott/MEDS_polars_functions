"""Module for code that provides a structured DSL to specify columns of dataframes or operations on said.

This module contains two key concepts: column expressions and matchers.

Matchers are used to specify conditionals over dataframes. They are expressed simply as dictionaries mapping
column names to values. Exact equality is used to match the column values.

Column expressions currently support the following types:
  - COL (`'col'`): A column expression that extracts a specified column.
  - STR (`'str'`): A column expression that is a string, with interpolation allowed to other column names via
    python's f-string syntax.
  - LITERAL (`'literal'`): A column expression that is a literal value regardless of type. No interpolation is
    allowed here.

Column expressions can be expressed either dictionary or via a shorthand string. If a structured dictionary,
the dictionary has length 1 and the key is one of the column expression types and the value is the expression
target (e.g., the column to load for `COL`, the string to interpolate with `{...}` escaped interpolation
targets for `STR`, or the literal value for `LITERAL`). If a string, the string is interpreted as a `COL` if
it has no interpolation targets, and as a `STR` otherwise.

These types can be combined or filtered via two modes:
  - Coalescing: Multiple column expressions can be combined into a single expression where the first non-null
    is used by specifying them in an ordered list.
  - Conditional: A column expression can be conditionally applied to a dataframe based on a matcher, by
    specifying the column expression and matcher in a dictionary, in one of two possible forms:
    - A single key-value pair, where the key is a string realization of either a `COL` or `STR` type expressed
      as a string and the value is the matcher dictionary.
    - Two key-value pairs, where the first key is `"output"` and the value is the column expression and the
      second key is `"matcher"` and the value is the matcher dictionary.
"""
from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

import polars as pl
from omegaconf import DictConfig, ListConfig, OmegaConf


def is_matcher(matcher_cfg: dict[str, Any]) -> bool:
    """Checks if a dictionary is a valid matcher configuration.

    Args:
        matcher_cfg: A dictionary of key-value pairs to match against.

    Returns:
        bool: True if the input is a valid matcher configuration, False otherwise.

    Examples:
        >>> is_matcher({"foo": "bar"})
        True
        >>> is_matcher(DictConfig({"foo": "bar"}))
        True
        >>> is_matcher({"foo": "bar", 32: "baz"})
        False
        >>> is_matcher(["foo", "bar"])
        False
        >>> is_matcher({})
        True
    """
    return isinstance(matcher_cfg, (dict, DictConfig)) and all(isinstance(k, str) for k in matcher_cfg.keys())


def matcher_to_expr(matcher_cfg: DictConfig | dict) -> tuple[pl.Expr, set[str]]:
    """Returns an expression and the necessary columns to match a collection of key-value pairs.

    Currently, this only supports checking for equality between column names and values.
    TODO: Expand (as needed only) to support other types of matchers.

    Args:
        matcher_cfg: A dictionary of key-value pairs to match against.

    Raises:
        ValueError: If the matcher configuration is not a dictionary.

    Returns:
        pl.Expr: A Polars expression that matches the key-value pairs in the input dictionary.
        set[str]: The set of input columns needed to form the returned expression.

    Examples:
        >>> expr, cols = matcher_to_expr({"foo": "bar", "buzz": "baz"})
        >>> print(expr)
        [(col("foo")) == (String(bar))].all_horizontal([[(col("buzz")) == (String(baz))]])
        >>> sorted(cols)
        ['buzz', 'foo']
        >>> expr, cols = matcher_to_expr(DictConfig({"foo": "bar", "buzz": "baz"}))
        >>> print(expr)
        [(col("foo")) == (String(bar))].all_horizontal([[(col("buzz")) == (String(baz))]])
        >>> sorted(cols)
        ['buzz', 'foo']
        >>> matcher_to_expr(["foo", "bar"])
        Traceback (most recent call last):
            ...
        ValueError: Matcher configuration must be a dictionary with string keys. Got: ['foo', 'bar']
    """
    if not is_matcher(matcher_cfg):
        raise ValueError(f"Matcher configuration must be a dictionary with string keys. Got: {matcher_cfg}")

    return pl.all_horizontal((pl.col(k) == v) for k, v in matcher_cfg.items()), set(matcher_cfg.keys())


STR_INTERPOLATION_REGEX = r"\{([^}]+)\}"


class ColExprType(StrEnum):
    """Enumeration of the different types of column expressions that can be parsed.

    Members:
        COL: A column expression that extracts a specified column.
        STR: A column expression that is a string, with interpolation allowed to other column names
            via python's f-string syntax.
        LITERAL: A column expression that is a literal value regardless of type. No interpolation is allowed
            here.
    """

    COL = "col"
    STR = "str"
    LITERAL = "literal"

    @classmethod
    def is_valid(cls, expr_dict: dict[ColExprType, Any]) -> tuple[bool, str | None]:
        """Checks if a dictionary of expression key to value is a valid column expression.

        Args:
            expr_dict: A dictionary of column expression type to value.

        Returns:
            bool: True if the input is a valid column expression, False otherwise.
            str | None: The reason the input is invalid, if it is invalid.

        Examples:
            >>> ColExprType.is_valid({"col": "foo"})
            (True, None)
            >>> ColExprType.is_valid({"col": 32})
            (False, 'Column expressions must have a string value. Got 32')
            >>> ColExprType.is_valid({ColExprType.STR: "bar//{foo}"})
            (True, None)
            >>> ColExprType.is_valid({ColExprType.STR: ["bar//{foo}"]})
            (False, "String interpolation expressions must have a string value. Got ['bar//{foo}']")
            >>> ColExprType.is_valid({"literal": ["baz", 32]})
            (True, None)
            >>> ColExprType.is_valid({"col": "foo", "str": "bar"}) # doctest: +NORMALIZE_WHITESPACE
            (False, "Column expressions can only contain a single key-value pair.
                    Got {'col': 'foo', 'str': 'bar'}")
            >>> ColExprType.is_valid({"foo": "bar"})
            (False, "Column expressions must have a key in ColExprType: ['col', 'str', 'literal']. Got foo")
            >>> ColExprType.is_valid([("col", "foo")])
            (False, "Column expressions must be a dictionary. Got [('col', 'foo')]")
        """

        if not isinstance(expr_dict, dict):
            return False, f"Column expressions must be a dictionary. Got {expr_dict}"
        if len(expr_dict) != 1:
            return False, f"Column expressions can only contain a single key-value pair. Got {expr_dict}"

        expr_type, expr_val = next(iter(expr_dict.items()))
        match expr_type:
            case cls.COL if isinstance(expr_val, str):
                return True, None
            case cls.COL:
                return False, f"Column expressions must have a string value. Got {expr_val}"
            case cls.STR if isinstance(expr_val, str):
                return True, None
            case cls.STR:
                return False, f"String interpolation expressions must have a string value. Got {expr_val}"
            case cls.LITERAL:
                return True, None
            case _:
                return (
                    False,
                    f"Column expressions must have a key in ColExprType: {[x.value for x in cls]}. Got "
                    f"{expr_type}",
                )

    @classmethod
    def to_pl_expr(cls, expr_type: ColExprType, expr_val: Any) -> tuple[pl.Expr, set[str]]:
        """Converts a column expression type and value to a Polars expression.

        Args:
            expr_type: The type of column expression.
            expr_val: The value of the column expression.

        Returns:
            pl.Expr: A Polars expression that extracts the column from the metadata DataFrame.
            set[str]: The set of input columns needed to form the returned expression.

        Raises:
            ValueError: If the column expression type is invalid.

        Examples:
            >>> print(*ColExprType.to_pl_expr(ColExprType.COL, "foo"))
            col("foo") {'foo'}
            >>> expr, cols = ColExprType.to_pl_expr(ColExprType.STR, "bar//{foo}//{baz}")
            >>> print(expr)
            String(bar//).str.concat_horizontal([col("foo"), String(//), col("baz")])
            >>> sorted(cols)
            ['baz', 'foo']
            >>> expr, cols = ColExprType.to_pl_expr(ColExprType.LITERAL, ListConfig(["foo", "bar"]))
            >>> print(expr)
            Series[literal]
            >>> pl.select(expr).item().to_list()
            ['foo', 'bar']
            >>> cols
            set()
            >>> ColExprType.to_pl_expr(ColExprType.COL, 32)
            Traceback (most recent call last):
                ...
            ValueError: ...
        """
        is_valid, err_msg = cls.is_valid({expr_type: expr_val})
        if not is_valid:
            raise ValueError(err_msg)

        match expr_type:
            case cls.COL:
                return pl.col(expr_val), {expr_val}
            case cls.STR:
                cols = list(re.findall(STR_INTERPOLATION_REGEX, expr_val))
                expr_val = re.sub(STR_INTERPOLATION_REGEX, "{}", expr_val)
                return pl.format(expr_val, *cols), set(cols)
            case cls.LITERAL:
                if isinstance(expr_val, ListConfig):
                    expr_val = OmegaConf.to_object(expr_val)
                return pl.lit(expr_val), set()


def parse_col_expr(cfg: str | list | dict[str, str] | ListConfig | DictConfig) -> dict:
    """Parses a column expression configuration object into a dictionary expressing the desired expression.

    Args:
        col_expr: A configuration object that specifies how to extract a column from the metadata. See the
            module docstring for formatting details.

    Returns:
        A dictionary specifying, in a structured form, the desired column expression.

    Examples:
        >>> parse_col_expr("foo")
        {'col': 'foo'}
        >>> parse_col_expr("bar//{foo}")
        {'str': 'bar//{foo}'}
        >>> parse_col_expr({'col': 'bar//{foo}'})
        {'col': 'bar//{foo}'}
        >>> parse_col_expr({"literal": ["foo", "bar"]})
        {'literal': ['foo', 'bar']}
        >>> parse_col_expr({"output": "foo", "matcher": {"bar": "baz"}})
        {'output': {'col': 'foo'}, 'matcher': {'bar': 'baz'}}
        >>> parse_col_expr({"output": "foo", "matcher": {32: "baz"}}) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: A pre-specified output/matcher configuration must have a valid matcher dictionary,
                    which is a dictionary with string-type keys. Got cfg['matcher']={32: 'baz'}
        >>> parse_col_expr({"foo": {"bar": "baz"}})
        {'output': {'col': 'foo'}, 'matcher': {'bar': 'baz'}}
        >>> parse_col_expr({"foo": {32: "baz"}}) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: A simple-form conditional expression is expressed with a single key-value pair dict,
                    where the key is not a column expression type and the value is a valid matcher dict,
                    which is a dictionary with string-type keys. This config has a single key-value pair
                    with key foo but an invalid matcher: {32: 'baz'}
        >>> parse_col_expr(["bar//{foo}", {"str": "bar//UNK"}])
        [{'str': 'bar//{foo}'}, {'str': 'bar//UNK'}]
        >>> parse_col_expr({"foo": "bar", "buzz": "baz", "fuzz": "fizz"}) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: Dictionary column expression must either be explicit output/matcher configs, with two
                    keys, 'output' and 'matcher' with a valid matcher dictionary, or a simple column
                    expression with a single key-value pair where the key is a column expression type, or a
                    simple-form conditional expression with a single key-value pair where the key is the
                    conditional value and the value is a valid matcher dict. Got a dictionary with 3 elements:
                    {'foo': 'bar', 'buzz': 'baz', 'fuzz': 'fizz'}
        >>> parse_col_expr(('foo', 'bar')) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: A simple column expression must be a string, list, or dictionary.
                    Got <class 'tuple'>: ('foo', 'bar')
        >>> parse_col_expr({"col": "foo", "str": "bar"}) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: Dictionary column expression must either be explicit output/matcher configs, with two
                    keys, 'output' and 'matcher' with a valid matcher dictionary, or a simple column
                    expression with a single key-value pair where the key is a column expression type, or a
                    simple-form conditional expression with a single key-value pair where the key is the
                    conditional value and the value is a valid matcher dict. Got a dictionary with 2 elements:
                    {'col': 'foo', 'str': 'bar'}
        >>> parse_col_expr(["foo", 32]) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: If a list (which coalesces columns), all elements must be strings or dictionaries.
                    Got: ['foo', 32]
    """
    match cfg:
        case str() if re.search(STR_INTERPOLATION_REGEX, cfg):
            return {"str": cfg}
        case str():
            return {"col": cfg}
        case list() | ListConfig() if all(isinstance(x, (str, dict, DictConfig)) for x in cfg):
            return [parse_col_expr(x) for x in cfg]
        case list() | ListConfig():
            raise ValueError(
                "If a list (which coalesces columns), all elements must be strings or dictionaries. "
                f"Got: {cfg}"
            )
        case dict() | DictConfig() if set(cfg.keys()) == {"output", "matcher"} and is_matcher(cfg["matcher"]):
            return {"output": parse_col_expr(cfg["output"]), "matcher": cfg["matcher"]}
        case dict() | DictConfig() if set(cfg.keys()) == {"output", "matcher"}:
            raise ValueError(
                "A pre-specified output/matcher configuration must have a valid matcher dictionary, which is "
                f"a dictionary with string-type keys. Got cfg['matcher']={cfg['matcher']}"
            )
        case dict() | DictConfig() if len(cfg) == 1 and ColExprType.is_valid(cfg)[0]:
            return cfg
        case dict() | DictConfig() if len(cfg) == 1:
            out_cfg, matcher_cfg = next(iter(cfg.items()))
            if is_matcher(matcher_cfg):
                return {"output": parse_col_expr(out_cfg), "matcher": matcher_cfg}
            else:
                raise ValueError(
                    "A simple-form conditional expression is expressed with a single key-value pair dict, "
                    "where the key is not a column expression type and the value is a valid matcher dict, "
                    "which is a dictionary with string-type keys. This config has a single key-value pair "
                    f"with key {out_cfg} but an invalid matcher: {matcher_cfg}"
                )
        case dict() | DictConfig():
            raise ValueError(
                "Dictionary column expression must either be explicit output/matcher configs, with two keys, "
                "'output' and 'matcher' with a valid matcher dictionary, or a simple column expression with "
                "a single key-value pair where the key is a column expression type, or a simple-form "
                "conditional expression with a single key-value pair where the key is the conditional value "
                f"and the value is a valid matcher dict. Got a dictionary with {len(cfg)} elements: {cfg}"
            )
        case _:
            raise ValueError(
                f"A simple column expression must be a string, list, or dictionary. Got {type(cfg)}: {cfg}"
            )


def structured_expr_to_pl(cfg: dict | list[dict] | ListConfig | DictConfig) -> tuple[pl.Expr, set[str]]:
    """Converts a structured column expression configuration object to a Polars expression.

    Args:
        structured_expr: A structured column expression configuration object. See the module docstring for DSL
            details.

    Returns:
        pl.Expr: A Polars expression that extracts the column from the metadata DataFrame.
        set[str]: The set of input columns needed to form the returned expression.

    Raises:
        ValueError: If the configuration object is invalid.

    Examples:
        >>> expr, cols = structured_expr_to_pl([{"col": "foo"}, {"str": "bar//{baz}"}, {"literal": "fizz"}])
        >>> print(expr)
        col("foo").coalesce([String(bar//).str.concat_horizontal([col("baz")]), String(fizz)])
        >>> sorted(cols)
        ['baz', 'foo']
        >>> expr, cols = structured_expr_to_pl({"output": {"literal": "foo"}, "matcher": {"bar": "baz"}})
        >>> print(expr)
        .when([(col("bar")) == (String(baz))].all_horizontal()).then(String(foo)).otherwise(null)
        >>> sorted(cols)
        ['bar']
        >>> expr, cols = structured_expr_to_pl({"col": "bar"})
        >>> print(expr)
        col("bar")
        >>> sorted(cols)
        ['bar']
        >>> structured_expr_to_pl(["foo", 32])
        Traceback (most recent call last):
            ...
        ValueError: Error processing list config on field 1 for ['foo', 32]
        >>> structured_expr_to_pl({"output": 32, "matcher": {"bar": "baz"}}) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: Error processing output/matcher config output expression for
                    {'output': 32, 'matcher': {'bar': 'baz'}}
        >>> structured_expr_to_pl({"output": "foo", "matcher": {32: "baz"}}) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: A pre-specified output/matcher configuration must have a valid matcher dictionary, which
                    is a dictionary with string-type keys. Got cfg['matcher']={32: 'baz'}
        >>> structured_expr_to_pl({"col": 32})
        Traceback (most recent call last):
            ...
        ValueError: Column expressions must have a string value. Got 32
        >>> structured_expr_to_pl("foo")
        Traceback (most recent call last):
            ...
        ValueError: A structured column expression must be a list or dictionary. Got <class 'str'>: foo
    """

    match cfg:
        case list() | ListConfig() as cfg_fields:
            component_exprs = []
            needed_cols = set()
            for i, field in enumerate(cfg_fields):
                try:
                    expr, cols = cfg_to_expr(field)
                except ValueError as e:
                    raise ValueError(f"Error processing list config on field {i} for {cfg}") from e
                component_exprs.append(expr)
                needed_cols.update(cols)
            return pl.coalesce(*component_exprs), needed_cols
        case dict() | DictConfig() if set(cfg.keys()) == {"output", "matcher"} and is_matcher(cfg["matcher"]):
            matcher_expr, matcher_cols = matcher_to_expr(cfg["matcher"])
            try:
                out_expr, out_cols = cfg_to_expr(cfg["output"])
            except ValueError as e:
                raise ValueError(f"Error processing output/matcher config output expression for {cfg}") from e
            return pl.when(matcher_expr).then(out_expr), out_cols | matcher_cols
        case dict() | DictConfig() if set(cfg.keys()) == {"output", "matcher"}:
            # TODO(mmd): DRY out this and other error messages.
            raise ValueError(
                "A pre-specified output/matcher configuration must have a valid matcher dictionary, which is "
                f"a dictionary with string-type keys. Got cfg['matcher']={cfg['matcher']}"
            )
        case dict() | DictConfig() if ColExprType.is_valid(cfg)[0]:
            expr_type, expr_val = next(iter(cfg.items()))
            return ColExprType.to_pl_expr(expr_type, expr_val)
        case dict() | DictConfig():
            _, err_msg = ColExprType.is_valid(cfg)
            raise ValueError(err_msg)
        case _:
            raise ValueError(
                f"A structured column expression must be a list or dictionary. Got {type(cfg)}: {cfg}"
            )


def cfg_to_expr(cfg: str | ListConfig | DictConfig) -> tuple[pl.Expr, set[str]]:
    """Converts a metadata column configuration object to a Polars expression.

    Args:
        cfg: A configuration object that specifies how to extract a column from the metadata. See the module
            docstring for formatting details.

    Returns:
        pl.Expr: A Polars expression that extracts the column from the metadata DataFrame.
        set[str]: The set of input columns needed to form the returned expression.

    Examples:
        >>> data = pl.DataFrame({
        ...     "foo": ["a", "b", "c"],
        ...     "bar": ["d", "e", "f"],
        ...     "baz": [1,   2,   3]
        ... })
        >>> expr, cols = cfg_to_expr("foo")
        >>> data.select(expr.alias("out"))["out"].to_list()
        ['a', 'b', 'c']
        >>> sorted(cols)
        ['foo']
        >>> expr, cols = cfg_to_expr("bar//{foo}//{baz}")
        >>> data.select(expr.alias("out"))["out"].to_list()
        ['bar//a//1', 'bar//b//2', 'bar//c//3']
        >>> sorted(cols)
        ['baz', 'foo']
        >>> expr, cols = cfg_to_expr({"literal": 34.2})
        >>> data.select(expr.alias("out"))["out"].to_list()
        [34.2]
        >>> cols
        set()
        >>> expr, cols = cfg_to_expr({"{baz}//{bar}": {"foo": "a"}})
        >>> data.select(expr.alias("out"))["out"].to_list()
        ['1//d', None, None]
        >>> sorted(cols)
        ['bar', 'baz', 'foo']
        >>> cfg = [
        ...    {"matcher": {"baz": 2}, "output": {"str": "bar//{baz}"}},
        ...    {"literal": "34.2"},
        ... ]
        >>> expr, cols = cfg_to_expr(cfg)
        >>> data.select(expr.alias("out"))["out"].to_list()
        ['34.2', 'bar//2', '34.2']
        >>> sorted(cols)
        ['baz']

    Note that sometimes coalescing can lead to unexpected results. For example, if the first expression is of
    a different type than the second, the second expression may have its type coerced to match the first,
    potentially in an unexpected manner. This is also related to some polars, bugs, such as
    https://github.com/pola-rs/polars/issues/17773
        >>> cfg = [
        ...    {"matcher": {"baz": 2}, "output": {"str": "bar//{baz}"}},
        ...    {"literal": 34.8218},
        ... ]
        >>> expr, cols = cfg_to_expr(cfg)
        >>> data.select(expr.alias("out"))["out"].to_list()
        ['34', 'bar//2', '34']
    """
    structured_expr = parse_col_expr(cfg)
    return structured_expr_to_pl(structured_expr)
