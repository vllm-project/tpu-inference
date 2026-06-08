# Copyright 2026 Google LLC
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

from dataclasses import asdict, is_dataclass

from rich.console import Console, Group
from rich.table import Table


def print_dataclasses_as_table(*instances):
    console = Console()

    # We will group multiple tables together
    render_groups = []

    for inst in instances:
        if not is_dataclass(inst):
            continue

        data = asdict(inst)
        class_name = type(inst).__name__

        # 1. Create the Header Table (The "Spanning" Row)
        header_table = Table(show_header=False, show_edge=False, expand=True)
        header_table.add_column(justify="center")
        header_table.add_row(
            f"[bold white on blue] {class_name.upper()} [/bold white on blue]")

        # 2. Create the Data Table
        data_table = Table(show_header=True,
                           header_style="bold magenta",
                           expand=True)

        # Add columns
        for key in data.keys():
            data_table.add_column(key)

        # Add values
        data_table.add_row(*[str(v) for v in data.values()])

        # 3. Add both to our render group
        render_groups.append(header_table)
        render_groups.append(data_table)
        render_groups.append("\n")  # Add spacing between dataclasses

    # Print the group
    console.print(Group(*render_groups))


# --- Example Usage ---
# print_dataclasses_as_table(tuning_key, tunable_params)
