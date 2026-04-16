#!/usr/bin/env python3
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
"""
Utility script to parse and visualize TPU inference batch statistics from JSONL logs.
"""
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from tpu_inference.runner.utils import (
    InferencePhase, determine_phase_from_batch_composition_stats)


def plot_tokens_vs_requests(df: pd.DataFrame, output_dir: str):
    """
    Plots prefill/decode token counts and active requests over time (by batch number).
    
    Args:
        df: DataFrame containing parsed batch statistics.
        output_dir: Local directory path to save the generated plot.
        
    Returns:
        Path to the generated image file, or None if skipped due to missing data.
    """
    required_cols = ['batch_no', 'num_prefill_tokens', 'num_decode_tokens']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Warning: Skipping 'Tokens vs Requests' plot. Missing columns: {missing_cols}"
        )
        return None

    plot_df = df.dropna(subset=required_cols).copy()
    if plot_df.empty:
        print(
            "Warning: Insufficient valid data for 'Tokens vs Requests' plot.")
        return None

    plain_formatter = mticker.ScalarFormatter()
    plain_formatter.set_scientific(False)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.fill_between(plot_df['batch_no'],
                     plot_df['num_prefill_tokens'],
                     color='orange',
                     alpha=0.3)
    ax1.plot(plot_df['batch_no'],
             plot_df['num_prefill_tokens'],
             label='Prefill Tokens',
             color='darkorange',
             linewidth=2)

    ax1.fill_between(plot_df['batch_no'],
                     plot_df['num_decode_tokens'],
                     color='green',
                     alpha=0.3)
    ax1.plot(plot_df['batch_no'],
             plot_df['num_decode_tokens'],
             label='Decode Tokens',
             color='darkgreen',
             linewidth=2)

    ax1.set_yscale('symlog', linthresh=1)
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel('Token Count')
    ax1.set_xlabel('Batch Number')
    ax1.set_title('Tokens vs Requests')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(plain_formatter)

    if 'num_reqs' in plot_df.columns:
        ax1_req = ax1.twinx()
        ax1_req.plot(plot_df['batch_no'],
                     plot_df['num_reqs'],
                     label='Active Requests',
                     color='blue',
                     linestyle=':',
                     linewidth=2)
        ax1_req.set_ylabel('Active Requests', color='blue')
        ax1_req.tick_params(axis='y', labelcolor='blue')
        ax1_req.set_ylim(bottom=0)
        ax1_req.legend(loc='upper right')
        ax1_req.yaxis.set_major_formatter(plain_formatter)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "tokens_vs_requests.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_bucket_utilization(df: pd.DataFrame, output_dir: str):
    """
    Plots the hardware bucket utilization (actual tokens vs wasted padding) per batch.
    
    Args:
        df: DataFrame containing parsed batch statistics.
        output_dir: Local directory path to save the generated plot.
        
    Returns:
        Path to the generated image file, or None if skipped due to missing data.
    """
    required_cols = [
        'batch_no', 'padded_total_num_scheduled_tokens',
        'total_num_scheduled_tokens'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Warning: Skipping 'Bucket Utilization' plot. Missing columns: {missing_cols}"
        )
        return None

    plot_df = df.dropna(subset=required_cols).copy()
    if plot_df.empty:
        print(
            "Warning: Insufficient valid data for 'Bucket Utilization' plot.")
        return None

    fig, ax2 = plt.subplots(figsize=(14, 5))
    padded_safe = plot_df['padded_total_num_scheduled_tokens'].replace(
        0, 1)  # Prevent division by zero
    actual_pct = (plot_df['total_num_scheduled_tokens'] / padded_safe) * 100
    waste_pct = ((padded_safe - plot_df['total_num_scheduled_tokens']) /
                 padded_safe) * 100

    ax2.stackplot(plot_df['batch_no'],
                  actual_pct,
                  waste_pct,
                  labels=['Actual Tokens Processed (%)', 'Wasted Padding (%)'],
                  colors=['blue', 'red'],
                  alpha=0.5)
    ax2.set_title('Bucket Utilization')
    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('Percentage of Bucket (%)')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "bucket_utilization.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_token_volume_by_phase(df: pd.DataFrame, output_dir: str):
    """
    Plots the total token volume distributed across different inference phases.
    
    Args:
        df: DataFrame containing parsed batch statistics.
        output_dir: Local directory path to save the generated plot.
        
    Returns:
        Path to the generated image file, or None if skipped due to missing data.
    """
    required_cols = [
        'num_prefill_tokens', 'num_decode_tokens', 'total_num_scheduled_tokens'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Warning: Skipping 'Token Volume by Phase' plot. Missing columns: {missing_cols}"
        )
        return None

    plot_df = df.dropna(subset=required_cols).copy()
    # Calculate the phase for each row using the utility function and avoid division by zero
    plot_df = plot_df[plot_df['total_num_scheduled_tokens'] > 0].copy()
    if plot_df.empty:
        print(
            "Warning: Insufficient valid data for 'Token Volume by Phase' plot."
        )
        return None

    try:
        plot_df['phase'] = plot_df.apply(
            lambda row: determine_phase_from_batch_composition_stats(
                row.to_dict()).name,
            axis=1)
    except Exception as e:
        print(
            f"Warning: Failed to determine phases for 'Token Volume by Phase' plot: {e}"
        )
        return None

    plain_formatter = mticker.ScalarFormatter()
    plain_formatter.set_scientific(False)

    fig, ax3 = plt.subplots(figsize=(14, 5))
    phase_stats = plot_df.groupby('phase')[[
        'num_prefill_tokens', 'num_decode_tokens'
    ]].sum()

    # Enforce a logical progression of phases on the x-axis
    expected_phases = [
        InferencePhase.PREFILL_ONLY.name, InferencePhase.PREFILL_HEAVY.name,
        InferencePhase.BALANCED.name, InferencePhase.DECODE_HEAVY.name,
        InferencePhase.DECODE_ONLY.name, InferencePhase.AMBIGUOUS.name
    ]
    all_phases = expected_phases + [
        p for p in phase_stats.index if p not in expected_phases
    ]
    phase_stats = phase_stats.reindex(all_phases).fillna(0)

    total_phase_tokens = phase_stats['num_prefill_tokens'] + phase_stats[
        'num_decode_tokens']

    phase_stats.plot(kind='bar',
                     stacked=True,
                     ax=ax3,
                     color=['darkorange', 'darkgreen'],
                     alpha=0.8)
    ax3.set_title('Total Token Volume by Phase')
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Total Tokens')
    ax3.legend(['Prefill Tokens', 'Decode Tokens'])
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    ax3.yaxis.set_major_formatter(plain_formatter)
    ax3.tick_params(axis='x', rotation=0)
    ax3.margins(y=0.15)

    # Annotate Graph 3 with the actual token counts
    for i, phase in enumerate(phase_stats.index):
        total_tokens = total_phase_tokens[phase]
        prefill_tokens = phase_stats.loc[phase, 'num_prefill_tokens']
        decode_tokens = phase_stats.loc[phase, 'num_decode_tokens']

        if total_tokens > 0:
            prefill_pct = (prefill_tokens / total_tokens) * 100
            decode_pct = (decode_tokens / total_tokens) * 100
        else:
            prefill_pct = decode_pct = 0.0

        annotation_text = f"Phase Total: {int(total_tokens):,}\nPrefill: {int(prefill_tokens):,} ({prefill_pct:.1f}%)\nDecode: {int(decode_tokens):,} ({decode_pct:.1f}%)"

        ax3.text(i,
                 total_tokens,
                 annotation_text,
                 ha='center',
                 va='bottom',
                 fontsize=10,
                 bbox=dict(facecolor='white',
                           alpha=0.8,
                           edgecolor='gray',
                           boxstyle='round,pad=0.3'))

    plt.tight_layout()
    out_path = os.path.join(output_dir, "token_volume_by_phase.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_batch_volume_by_phase(df: pd.DataFrame, output_dir: str):
    """
    Plots the number of batches that fall into each inference phase.
    
    Args:
        df: DataFrame containing parsed batch statistics.
        output_dir: Local directory path to save the generated plot.
        
    Returns:
        Path to the generated image file, or None if skipped due to missing data.
    """
    required_cols = [
        'num_prefill_tokens', 'num_decode_tokens', 'total_num_scheduled_tokens'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Warning: Skipping 'Batch Volume by Phase' plot. Missing columns: {missing_cols}"
        )
        return None

    plot_df = df.dropna(subset=required_cols).copy()
    # Calculate the phase for each row using the utility function and avoid division by zero
    plot_df = plot_df[plot_df['total_num_scheduled_tokens'] > 0].copy()
    if plot_df.empty:
        print(
            "Warning: Insufficient valid data for 'Batch Volume by Phase' plot."
        )
        return None

    try:
        plot_df['phase'] = plot_df.apply(
            lambda row: determine_phase_from_batch_composition_stats(
                row.to_dict()).name,
            axis=1)
    except Exception as e:
        print(
            f"Warning: Failed to determine phases for 'Batch Volume by Phase' plot: {e}"
        )
        return None

    plain_formatter = mticker.ScalarFormatter()
    plain_formatter.set_scientific(False)

    fig, ax4 = plt.subplots(figsize=(14, 5))
    phase_counts = plot_df['phase'].value_counts()

    # Enforce a logical progression of phases on the x-axis
    expected_phases = [
        InferencePhase.PREFILL_ONLY.name, InferencePhase.PREFILL_HEAVY.name,
        InferencePhase.BALANCED.name, InferencePhase.DECODE_HEAVY.name,
        InferencePhase.DECODE_ONLY.name, InferencePhase.AMBIGUOUS.name
    ]
    all_phases = expected_phases + [
        p for p in phase_counts.index if p not in expected_phases
    ]
    phase_counts = phase_counts.reindex(all_phases).fillna(0)

    total_batches = phase_counts.sum()

    phase_counts.plot(kind='bar', ax=ax4, color='purple', alpha=0.8)
    ax4.set_title('Total Batch Volume by Phase')
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Number of Batches')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    ax4.yaxis.set_major_formatter(plain_formatter)
    ax4.tick_params(axis='x', rotation=0)
    ax4.margins(y=0.15)

    # Annotate Graph 4 with the actual batch counts
    for i, phase in enumerate(phase_counts.index):
        count = phase_counts[phase]

        if total_batches > 0:
            pct = (count / total_batches) * 100
        else:
            pct = 0.0

        annotation_text = f"Batches: {int(count):,}\n({pct:.1f}%)"

        ax4.text(i,
                 count,
                 annotation_text,
                 ha='center',
                 va='bottom',
                 fontsize=10,
                 bbox=dict(facecolor='white',
                           alpha=0.8,
                           edgecolor='gray',
                           boxstyle='round,pad=0.3'))

    plt.tight_layout()
    out_path = os.path.join(output_dir, "batch_volume_by_phase.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_stats(input_file: str, output_dir: str, graphs: list) -> list:
    """
    Reads the input JSONL file and triggers the requested plotting functions.
    
    Args:
        input_file: Path to the raw JSONL batch stats log file.
        output_dir: Local directory path to save the output graphs.
        graphs: List of specific graph types to generate (or ["all"]).
        
    Returns:
        List of file paths to the successfully generated image files.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return []

    try:
        df = pd.read_json(input_file, lines=True)
    except ValueError as e:
        print(f"Error reading JSONL file {input_file}: {e}")
        return []

    if df.empty:
        print(f"Warning: Input file {input_file} contains no data.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    if 'all' in graphs or 'tokens_vs_requests' in graphs:
        if file1 := plot_tokens_vs_requests(df, output_dir):
            generated_files.append(file1)

    if 'all' in graphs or 'bucket_utilization' in graphs:
        if file2 := plot_bucket_utilization(df, output_dir):
            generated_files.append(file2)

    if 'all' in graphs or 'token_volume_by_phase' in graphs:
        if file3 := plot_token_volume_by_phase(df, output_dir):
            generated_files.append(file3)

    if 'all' in graphs or 'batch_volume_by_phase' in graphs:
        if file4 := plot_batch_volume_by_phase(df, output_dir):
            generated_files.append(file4)

    return generated_files


def main():

    parser = argparse.ArgumentParser(description="Plot TPU batch statistics.")
    parser.add_argument("--input",
                        "-i",
                        type=str,
                        required=True,
                        help="Path to the input JSONL stats file")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="/tmp/batch_stats",
        help="Local directory to save the generated graph images")
    parser.add_argument("--graphs",
                        nargs="+",
                        default=["all"],
                        choices=[
                            "tokens_vs_requests", "bucket_utilization",
                            "token_volume_by_phase", "batch_volume_by_phase",
                            "all"
                        ],
                        help="Specify which graphs to generate (default: all)")

    args = parser.parse_args()

    generated_files = plot_stats(args.input, args.output_dir, args.graphs)
    if generated_files:
        print(f"Graphs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
