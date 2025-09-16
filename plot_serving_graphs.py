import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd


def parse_log_file(file_path):
    """
    Parses a log file to extract both model_fn and throughput data.

    Args:
        file_path (str): The path to the log file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a log entry.
    """
    data = []

    # Regex for the detailed model_fn performance line
    model_fn_pattern = re.compile(
        r"model_fn time: (?P<model_fn_time>[\d.]+) for batch "
        r"{'total_num_scheduled_tokens': (?P<total_num_scheduled_tokens>\d+), "
        r"'num_prefill_tokens': (?P<num_prefill_tokens>\d+), "
        r"'num_decode_tokens': (?P<num_decode_tokens>\d+), "
        r"'padded_total_num_scheduled_tokens': (?P<padded_total_num_scheduled_tokens>\d+), "
        r"'num_reqs': (?P<num_reqs>\d+)}.* "
        r"with (?P<preempted_requests>\d+) preempted requests")

    # Regex for the summary throughput and usage line
    throughput_pattern = re.compile(
        r"Avg prompt throughput: (?P<avg_prompt_throughput>[\d.]+) tokens/s, "
        r"Avg generation throughput: (?P<avg_generation_throughput>[\d.]+) tokens/s, "
        r"Running: (?P<running_reqs>\d+) reqs, "
        r"Waiting: (?P<waiting_reqs>\d+) reqs, "
        r"GPU KV cache usage: (?P<gpu_kv_cache_usage>[\d.]+)%, "
        r"Prefix cache hit rate: (?P<prefix_cache_hit_rate>[\d.]+)%")

    with open(file_path, 'r') as f:
        for line in f:
            # First, check for the detailed model_fn line
            entry_data = None
            if "model_fn time:" in line:
                match = model_fn_pattern.search(line)
                if match:
                    # Create a new dictionary for this entry
                    entry_data = match.groupdict()

                    if "Avg prompt throughput:" in line:
                        match = throughput_pattern.search(line)
                        entry_data.update(match.groupdict())
                    data.append(entry_data)

    return data


def plot_data(df):
    """
    Generates and displays plots from the parsed log data.

    Args:
        df (pd.DataFrame): A DataFrame containing the log data.
    """
    if df.empty:
        print("No data to plot.")
        return

    # Convert data types for plotting
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])

    # Create subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
    fig.suptitle('Log File Analysis', fontsize=16)

    # Plot 1: Model Function Time Over Entries
    axes[0, 0].plot(df.index, df['model_fn_time'], marker='o', linestyle='-')
    axes[0, 0].set_title('Model Function Time')
    axes[0, 0].set_xlabel('Log Entry Index')
    axes[0, 0].set_ylabel('Time (s)')

    # Plot 2: Number of Requests Over Entries
    axes[0, 1].plot(df.index,
                    df['num_reqs'],
                    marker='o',
                    linestyle='-',
                    color='green')
    axes[0, 1].set_title('Number of Requests')
    axes[0, 1].set_xlabel('Log Entry Index')
    axes[0, 1].set_ylabel('Count')

    # Plot 3: Scheduled Tokens Over Entries
    axes[1, 0].plot(df.index,
                    df['total_num_scheduled_tokens'],
                    marker='o',
                    linestyle='-',
                    color='red',
                    label='Total')
    # axes[1, 0].plot(df.index, df['padded_total_num_scheduled_tokens'], marker='o', linestyle='--', color='purple', label='Padded')
    axes[1, 0].set_title('Scheduled Tokens')
    axes[1, 0].set_xlabel('Log Entry Index')
    axes[1, 0].set_ylabel('Token Count')
    axes[1, 0].legend()

    # Plot 4: Prefill vs Decode Tokens
    axes[1, 1].scatter(df['num_prefill_tokens'], df['num_decode_tokens'])
    axes[1, 1].set_title('Prefill vs. Decode Tokens')
    axes[1, 1].set_xlabel('Prefill Tokens')
    axes[1, 1].set_ylabel('Decode Tokens')

    # Plot 5: Preempted Requests Over Entries
    axes[2, 0].plot(df.index,
                    df['preempted_requests'],
                    marker='o',
                    linestyle='-',
                    color='orange')
    axes[2, 0].set_title('Preempted Requests')
    axes[2, 0].set_xlabel('Log Entry Index')
    axes[2, 0].set_ylabel('Count')

    # Plot 6: Relationship between model_fn_time and total_num_scheduled_tokens
    axes[2, 1].scatter(df['total_num_scheduled_tokens'], df['model_fn_time'])
    axes[2, 1].set_title('Model Time vs. Scheduled Tokens')
    axes[2, 1].set_xlabel('Total Scheduled Tokens')
    axes[2, 1].set_ylabel('Model Function Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # save to png
    plt.savefig('graph_vllm_logs.png')


def plot_data_2(df):
    """
    Generates and displays plots from the parsed log data, including model_fn_time
    as a function of the prefill/scheduled token ratio.

    Args:
        df (pd.DataFrame): A DataFrame containing the log data.
    """
    if df.empty:
        print(
            "No data to plot. Check if the log file contains relevant entries."
        )
        return

    # Convert all possible columns to numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    fig.suptitle('Log File Analysis', fontsize=20)

    # Plot 1: Throughputs
    throughput_df = df.dropna(
        subset=['avg_prompt_throughput', 'avg_generation_throughput'])
    axes[0, 0].plot(throughput_df.index,
                    throughput_df['avg_generation_throughput'],
                    marker='o',
                    linestyle='-',
                    label='Generation')
    axes[0, 0].plot(throughput_df.index,
                    throughput_df['avg_prompt_throughput'],
                    marker='s',
                    linestyle='--',
                    label='Prompt')
    axes[0, 0].set_title('Average Throughput')
    axes[0, 0].set_xlabel('Log Entry Index')
    axes[0, 0].set_ylabel('Tokens/s')
    axes[0, 0].legend()

    # Plot 2: Cache Usage and Hit Rate
    cache_df = df.dropna(
        subset=['gpu_kv_cache_usage', 'prefix_cache_hit_rate'])
    ax2_left = axes[0, 1]
    ax2_right = ax2_left.twinx()  # Create a second y-axis
    ax2_left.plot(cache_df.index,
                  cache_df['gpu_kv_cache_usage'],
                  marker='o',
                  color='tab:blue',
                  label='GPU KV Cache')
    ax2_right.plot(cache_df.index,
                   cache_df['prefix_cache_hit_rate'],
                   marker='s',
                   color='tab:red',
                   label='Prefix Cache Hit')
    ax2_left.set_ylabel('Usage (%)', color='tab:blue')
    ax2_right.set_ylabel('Hit Rate (%)', color='tab:red')
    ax2_left.set_title('Cache Performance')
    ax2_left.set_xlabel('Log Entry Index')
    lines, labels = ax2_left.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2_right.legend(lines + lines2, labels + labels2, loc='upper right')

    # Plot 3: Running vs Waiting Requests
    req_status_df = df.dropna(subset=['running_reqs', 'waiting_reqs'])
    axes[1, 0].plot(req_status_df.index,
                    req_status_df['running_reqs'],
                    marker='o',
                    linestyle='-',
                    label='Running')
    axes[1, 0].plot(req_status_df.index,
                    req_status_df['waiting_reqs'],
                    marker='s',
                    linestyle='--',
                    label='Waiting')
    axes[1, 0].set_title('Request Queue Status')
    axes[1, 0].set_xlabel('Log Entry Index')
    axes[1, 0].set_ylabel('Number of Requests')
    axes[1, 0].legend()

    # Plot 4: NEW - Stacked Bar Chart of Prefill/Decode Tokens
    token_df = df.dropna(
        subset=['num_prefill_tokens', 'num_decode_tokens']).copy()

    # Plotting the stacked bar chart
    axes[1, 1].bar(token_df.index,
                   token_df['num_prefill_tokens'],
                   label='Prefill Tokens')
    axes[1, 1].bar(token_df.index,
                   token_df['num_decode_tokens'],
                   bottom=token_df['num_prefill_tokens'],
                   label='Decode Tokens')

    axes[1, 1].set_title('Prefill and Decode Tokens per Log Entry')
    axes[1, 1].set_xlabel('Log Entry Index')
    axes[1, 1].set_ylabel('Token Count')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    plt.savefig('graph_vllm_logs_2.png')


if __name__ == "__main__":
    # Setup the argparser and parse the command line arguments for logfile
    parser = argparse.ArgumentParser(description='Plot log file data.')
    parser.add_argument('log_file', type=str, help='Path to the log file')
    args = parser.parse_args()
    log_file_path = args.log_file
    parsed_data = parse_log_file(log_file_path)

    # total number of preempted requests
    print("Total number of preempted requests:",
          sum(int(d['preempted_requests']) for d in parsed_data))

    # exclude any entry with model_fn > 0.7
    parsed_data = [d for d in parsed_data if float(d['model_fn_time']) < 0.7]

    # Create a pandas DataFrame
    df = pd.DataFrame(parsed_data)

    # Plot the data
    plot_data(df)
    plot_data_2(df)
