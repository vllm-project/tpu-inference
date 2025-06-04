# Copyright 2024 The CLU Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper function for creating and logging JAX variable overviews."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging

_ParamsContainer = dict[str, np.ndarray] | Mapping[str, Mapping[str, Any]]


@dataclasses.dataclass
class _ParamRow:
    name: str
    shape: tuple[int, ...]
    dtype: str
    size: int


@dataclasses.dataclass
class _ParamRowWithSharding(_ParamRow):
    sharding: tuple[int | None, ...] | str


@dataclasses.dataclass
class _ParamRowWithStats(_ParamRow):
    mean: float
    std: float


@dataclasses.dataclass
class _ParamRowWithStatsAndSharding(_ParamRowWithStats):
    sharding: tuple[int | None, ...] | str


@jax.jit
def _mean_std_jit(x):
    return jax.tree_util.tree_map(jnp.mean,
                                  x), jax.tree_util.tree_map(jnp.std, x)


def _mean_std(x):
    mean = jax.tree_util.tree_map(lambda x: x.mean(), x)
    std = jax.tree_util.tree_map(lambda x: x.std(), x)
    return mean, std


def flatten_dict(input_dict: dict[str, Any],
                 *,
                 prefix: str = "",
                 delimiter: str = ".") -> dict[str, Any]:
    """Flattens the keys of a nested dictionary."""
    output_dict = {}
    for key, value in input_dict.items():
        nested_key = f"{prefix}{delimiter}{key}" if prefix else key
        if isinstance(value, (dict, flax.core.FrozenDict)):
            output_dict.update(
                flatten_dict(value, prefix=nested_key, delimiter=delimiter))
        else:
            output_dict[nested_key] = value
    return output_dict


def _count_parameters(params: _ParamsContainer) -> int:
    """Returns the count of variables for the module or parameter dictionary."""
    params = flatten_dict(params)
    return sum(np.prod(v.shape) for v in params.values())


def _parameters_size(params: _ParamsContainer) -> int:
    """Returns total size (bytes) for the module or parameter dictionary."""
    params = flatten_dict(params)
    return sum(np.prod(v.shape) * v.dtype.itemsize for v in params.values())


def count_parameters(params: _ParamsContainer) -> int:
    """Returns the count of variables for the module or parameter dictionary."""

    return _count_parameters(params)


def _make_row(name, value) -> _ParamRow:
    return _ParamRow(
        name=name,
        shape=value.shape,
        dtype=str(value.dtype),
        size=int(np.prod(value.shape)),
    )


def _make_row_with_sharding(name, value) -> _ParamRowWithSharding:
    row = _make_row(name, value)
    if hasattr(value, "sharding"):
        if hasattr(value.sharding, "spec"):
            sharding = tuple(value.sharding.spec)
        else:
            sharding = str(value.sharding)
    else:
        sharding = ()
    return _ParamRowWithSharding(**dataclasses.asdict(row), sharding=sharding)


def _make_row_with_stats(name, value, mean, std) -> _ParamRowWithStats:
    row = _make_row(name, value)
    return _ParamRowWithStats(
        **dataclasses.asdict(row),
        mean=float(jax.device_get(mean)),
        std=float(jax.device_get(std)),
    )


def _make_row_with_stats_and_sharding(name, value, mean,
                                      std) -> _ParamRowWithStatsAndSharding:
    row = _make_row_with_sharding(name, value)
    return _ParamRowWithStatsAndSharding(
        **dataclasses.asdict(row),
        mean=float(jax.device_get(mean)),
        std=float(jax.device_get(std)),
    )


def _get_parameter_rows(
    params: _ParamsContainer,
    *,
    include_stats: bool | str = False,
) -> list[_ParamRow]:
    """Returns information about parameters as a list of dictionaries.

    Args:
      params: Dictionary with parameters as NumPy arrays. The dictionary can be
        nested. Alternatively a `tf.Module` can be provided, in which case the
        `trainable_variables` of the module will be used.
      include_stats: If True, add columns with mean and std for each variable.
        If the string "sharding", add column a column with the sharding of the
        variable.
        If the string "global", params are sharded global arrays and this
        function assumes it is called on every host, i.e. can use collectives.
        The sharding of the variables is also added as a column.

    Returns:
      A list of `ParamRow`, or `ParamRowWithStats`, depending on the passed value
      of `include_stats`.
    """
    if not isinstance(params, (dict, flax.core.FrozenDict)):
        raise ValueError(
            f"Expected `params` to be a dictionary but got {type(params)}")

    if params:
        params = flatten_dict(params)
        names, values = map(list, tuple(zip(*sorted(params.items()))))
    else:
        names, values = [], []

    match include_stats:
        case False:
            return jax.tree_util.tree_map(_make_row, names, values)

        case True:
            mean_and_std = _mean_std(values)
            return jax.tree_util.tree_map(_make_row_with_stats, names, values,
                                          *mean_and_std)

        case "global":
            mean_and_std = _mean_std_jit(values)
            return jax.tree_util.tree_map(_make_row_with_stats_and_sharding,
                                          names, values, *mean_and_std)

        case "sharding":
            return jax.tree_util.tree_map(_make_row_with_sharding, names,
                                          values)

        case _:
            raise ValueError(f"Unknown `include_stats`: {include_stats}")


def _default_table_value_formatter(value):
    """Formats ints with "," between thousands and floats to 3 digits."""
    if isinstance(value, bool):
        return str(value)
    elif isinstance(value, int):
        return "{:,}".format(value)
    elif isinstance(value, float):
        return "{:.3}".format(value)
    else:
        return str(value)


def make_table(
    rows: list[Any],
    *,
    column_names: Sequence[str] | None = None,
    value_formatter: Callable[[Any], str] = _default_table_value_formatter,
    max_lines: int | None = None,
) -> str:
    """Renders a list of rows to a table.

    Args:
      rows: List of dataclass instances of a single type (e.g. `ParamRow`).
      column_names: List of columns that that should be included in the output. If
        not provided, then the columns are taken from keys of the first row.
      value_formatter: Callable used to format cell values.
      max_lines: Don't render a table longer than this.

    Returns:
      A string representation of the table in the form:

      +---------+---------+
      | Col1    | Col2    |
      +---------+---------+
      | value11 | value12 |
      | value21 | value22 |
      +---------+---------+
    """

    if any(not dataclasses.is_dataclass(row) for row in rows):
        raise ValueError("Expected `rows` to be list of dataclasses")
    if len(set(map(type, rows))) > 1:
        raise ValueError("Expected elements of `rows` be of same type.")

    class Column:

        def __init__(self, name, values):
            self.name = name.capitalize()
            self.values = values
            self.width = max(len(v) for v in values + [name])

    if column_names is None:
        if not rows:
            return "(empty table)"
        column_names = [field.name for field in dataclasses.fields(rows[0])]

    columns = [
        Column(name, [value_formatter(getattr(row, name)) for row in rows])
        for name in column_names
    ]

    var_line_format = "|" + "".join(f" {{: <{c.width}s}} |" for c in columns)
    sep_line_format = var_line_format.replace(" ", "-").replace("|", "+")
    header = var_line_format.replace(">",
                                     "<").format(*[c.name for c in columns])
    separator = sep_line_format.format(*["" for c in columns])

    lines = [separator, header, separator]
    for i in range(len(rows)):
        if max_lines and len(lines) >= max_lines - 3:
            lines.append("[...]")
            break
        lines.append(var_line_format.format(*[c.values[i] for c in columns]))
    lines.append(separator)

    return "\n".join(lines)


def _get_parameter_overview(
    params: _ParamsContainer,
    *,
    include_stats: bool | str = True,
    max_lines: int | None = None,
) -> str:
    """See get_parameter_overview()."""
    if include_stats is True and isinstance(params,
                                            (dict, flax.core.FrozenDict)):  # pylint: disable=g-bool-id-comparison
        params = jax.device_get(params)  # A no-op if already numpy array.
    rows = _get_parameter_rows(params, include_stats=include_stats)
    RowType = {  # pylint: disable=invalid-name
        False: _ParamRow,
        True: _ParamRowWithStats,
        "global": _ParamRowWithStatsAndSharding,
        "sharding": _ParamRowWithSharding,
    }[include_stats]
    # Pass in `column_names` to enable rendering empty tables.
    column_names = [field.name for field in dataclasses.fields(RowType)]
    table = make_table(rows, max_lines=max_lines, column_names=column_names)
    total_weights = _count_parameters(params)
    total_size = _parameters_size(params)
    return table + f"\nTotal: {total_weights:,} -- {total_size:,} bytes"


def get_parameter_overview(
    params: _ParamsContainer,
    *,
    include_stats: bool | str = True,
    max_lines: int | None = None,
) -> str:
    """Returns a string with variables names, their shapes, count.

    Args:
      params: Dictionary with parameters as NumPy arrays. The dictionary can be
        nested.
      include_stats: If True, add columns with mean and std for each variable.
        If the string "sharding", add column a column with the sharding of the
        variable.
        If the string "global", params are sharded global arrays and this
        function assumes it is called on every host, i.e. can use collectives.
        The sharding of the variables is also added as a column.
      max_lines: If not `None`, the maximum number of variables to include.

    Returns:
      A string with a table like in the example.

    +----------------+---------------+------------+
    | Name           | Shape         | Size       |
    +----------------+---------------+------------+
    | FC_1/weights:0 | (63612, 1024) | 65,138,688 |
    | FC_1/biases:0  |       (1024,) |      1,024 |
    | FC_2/weights:0 |    (1024, 32) |     32,768 |
    | FC_2/biases:0  |         (32,) |         32 |
    +----------------+---------------+------------+
    Total: 65,172,512
    """

    return _get_parameter_overview(params,
                                   include_stats=include_stats,
                                   max_lines=max_lines)


def _log_parameter_overview(
    params: _ParamsContainer,
    *,
    include_stats: bool | str = True,
    max_lines: int | None = None,
    msg: str | None = None,
    jax_logging_process: int | None = None,
):
    """See log_parameter_overview()."""

    table = _get_parameter_overview(params,
                                    include_stats=include_stats,
                                    max_lines=max_lines)
    if jax_logging_process is None or jax_logging_process == jax.process_index(
    ):
        lines = [msg] if msg else []
        lines += table.split("\n")
        # The table can be too large to fit into one log entry.
        for i in range(0, len(lines), 80):
            logging.info("\n%s", "\n".join(lines[i:i + 80]))


def log_parameter_overview(
    params: _ParamsContainer,
    *,
    include_stats: bool | str = True,
    max_lines: int | None = None,
    msg: str | None = None,
    jax_logging_process: int | None = None,
):
    """Writes a table with variables name and shapes to INFO log.

    See get_parameter_overview for details.

    Args:
      params: Dictionary with parameters as NumPy arrays. The dictionary can be
        nested.
      include_stats: If True, add columns with mean and std for each variable.
        If the string "global", params are sharded global arrays and this
        function assumes it is called on every host, i.e. can use collectives.
      max_lines: If not `None`, the maximum number of variables to include.
      msg: Message to be logged before the overview.
      jax_logging_process: Which JAX process ID should do the logging. None = all.
        Use this to avoid logspam when include_stats="global".
    """

    _log_parameter_overview(
        params,
        include_stats=include_stats,
        max_lines=max_lines,
        msg=msg,
        jax_logging_process=jax_logging_process,
    )
