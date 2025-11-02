import ast
import os
import sys

# Dictionary to map file suffixes to device names
DEVICE_NAME_MAP = {
    'gf': 'TPU v7',
    'gl': 'TPU v6e',
    'vl': 'TPU v5e',
}


def parse_data_to_multilevel_table(filepath):
  """Parses a text file with data in a specific format and converts it into

  a multi-level dictionary (table). The device name is determined by the
  file's suffix.

  Args:
      filepath (str): The path to the input text file.

  Returns:
      dict: A multi-level dictionary containing the parsed data.
  """
  data_table = {}

  try:
    # Extract the device name from the file's suffix
    filename = os.path.basename(filepath)
    suffix = filename.rsplit('.', 1)[0].rsplit('_', 1)[-1]
    device_name = DEVICE_NAME_MAP.get(suffix, 'unknown_device')

    with open(filepath, 'r') as file:
      for line in file:
        line = line.strip()
        if not line:
          continue  # Skip empty lines

        try:
          # Split the line into the key and value parts
          key_str, value_str = line.split(':', 1)

          # Safely parse the string representations of tuples
          key_tuple = ast.literal_eval(key_str.strip())
          value_tuple = ast.literal_eval(value_str.strip())[0]

          # Unpack the key tuple
          (
              page_size,
              q_dtype,
              kv_dtype,
              num_q_heads,
              num_kv_heads,
              head_dim,
              max_model_len,
          ) = key_tuple

          page_size = int(page_size)
          # Construct the hierarchical keys
          q_kv_dtype_key = f'q_{q_dtype}_kv_{kv_dtype}'
          head_key = (
              f'q_head-{num_q_heads}_kv_head-{num_kv_heads}_head-{head_dim}'
          )

          # Build the nested dictionary structure using device_name
          if device_name not in data_table:
            data_table[device_name] = {}

          if page_size not in data_table[device_name]:
            data_table[device_name][page_size] = {}

          if q_kv_dtype_key not in data_table[device_name][page_size]:
            data_table[device_name][page_size][q_kv_dtype_key] = {}

          if head_key not in data_table[device_name][page_size][q_kv_dtype_key]:
            data_table[device_name][page_size][q_kv_dtype_key][head_key] = {}

          # Assign the final value
          data_table[device_name][page_size][q_kv_dtype_key][head_key][
              max_model_len
          ] = value_tuple

        except (ValueError, IndexError, SyntaxError) as e:
          print(f"Skipping malformed line: '{line}' - Error: {e}")
          continue

  except FileNotFoundError:
    print(f"Error: The file '{filepath}' was not found.")
    return None

  return data_table


if __name__ == '__main__':
  # Allow the user to pass the file path as a command-line argument.
  # If no argument is provided, default to 'data.txt'.
  if len(sys.argv) > 1:
    file_to_parse = sys.argv[1]
  else:
    raise ValueError('No file path provided.')

  parsed_data = parse_data_to_multilevel_table(file_to_parse)

  if parsed_data:
    print(parsed_data)
  else:
    raise ValueError('Failed to parse the data.')
