# from dataclasses import asdict, is_dataclass
# from rich.console import Console
# from rich.table import Table

# def print_dataclasses_as_table(*instances):
#     """
#     Prints multiple dataclass instances in a table with two-level headers.
#     """
#     console = Console()
    
#     # 1. Flatten all dataclasses into a structure: {Class_Name: {field: value, ...}}
#     data = {}
#     for inst in instances:
#         if is_dataclass(inst):
#             name = type(inst).__name__
#             data[name] = asdict(inst)
#         else:
#             console.print(f"[red]Warning:[/red] {type(inst).__name__} is not a dataclass.")

#     # 2. Initialize Table
#     table = Table(show_header=True, header_style="bold magenta", border_style="dim")

#     # 3. Create Columns
#     # We iterate through the dataclasses to create the "Span" and "Field" columns
#     for class_name, fields in data.items():
#         for field_name in fields.keys():
#             # The 'header' is the field name, we will simulate the span row later
#             table.add_column(field_name, justify="right")

#     # 4. Construct the first row (The Span/Category Row)
#     # This acts as your custom "Span" row
#     span_row = []
#     for class_name, fields in data.items():
#         # Add the class name to the first column, fill the rest with empty strings
#         for i, _ in enumerate(fields.keys()):
#             span_row.append(class_name if i == 0 else "")
    
#     table.add_row(*span_row, end_section=True)

#     # 5. Construct the data row
#     values = []
#     for fields in data.values():
#         for val in fields.values():
#             values.append(str(val))
    
#     table.add_row(*values)

#     console.print(table)

# # --- Example Usage ---
# # Assuming TuningKey and TunableParams are your defined classes
# print_dataclasses_as_table(tuning_key_instance, tunable_params_instance)



from dataclasses import asdict, is_dataclass
from rich.console import Console
from rich.table import Table

def print_dataclasses_as_table(*instances):
    console = Console()
    
    # 1. Structure the data: { 'ClassName': {'field1': 'val1', ...} }
    data_map = {}
    for inst in instances:
        if is_dataclass(inst):
            data_map[type(inst).__name__] = asdict(inst)
        else:
            console.print(f"[red]Warning:[/red] {type(inst).__name__} is not a dataclass.")

    # 2. Flatten all field names for column creation
    all_fields = []
    for cls_name, fields in data_map.items():
        all_fields.extend(fields.keys())

    # 3. Initialize Table with lines for a grid effect
    table = Table(show_header=False, show_lines=True, header_style="bold magenta")

    # Add columns (one for every single field)
    for field in all_fields:
        table.add_column(field)

    # 4. Construct the "Span" Row (Row 1)
    # We place the ClassName in the first column of its respective group, 
    # then leave the other columns in that group empty.
    span_row = []
    for cls_name, fields in data_map.items():
        span_row.append(f"[bold cyan]{cls_name}[/bold cyan]")  # The header name
        span_row.extend([""] * (len(fields) - 1))              # Padding for the span
    table.add_row(*span_row)

    # 5. Construct the "Field" Row (Row 2)
    field_names = [f for fields in data_map.values() for f in fields.keys()]
    table.add_row(*[f"[yellow]{f}[/yellow]" for f in field_names])

    # 6. Construct the "Data" Row (Row 3)
    values = [str(v) for fields in data_map.values() for v in fields.values()]
    table.add_row(*values)

    console.print(table)

# --- Example Usage ---
# print_dataclasses_as_table(tuning_key, tunable_params)

from dataclasses import asdict, is_dataclass
from rich.console import Console, Group
from rich.table import Table
from rich.align import Align

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
        header_table.add_row(f"[bold white on blue] {class_name.upper()} [/bold white on blue]")

        # 2. Create the Data Table
        data_table = Table(show_header=True, header_style="bold magenta", expand=True)
        
        # Add columns
        for key in data.keys():
            data_table.add_column(key)
        
        # Add values
        data_table.add_row(*[str(v) for v in data.values()])
        
        # 3. Add both to our render group
        render_groups.append(header_table)
        render_groups.append(data_table)
        render_groups.append("\n") # Add spacing between dataclasses

    # Print the group
    console.print(Group(*render_groups))

# --- Example Usage ---
# print_dataclasses_as_table(tuning_key, tunable_params)