# Update Logic and Layout of Parallelism Techniques Table

This plan outlines how we'll update the `README.md` Parallelism Techniques table based on your request.

## Branch Confirmation
First, to answer your question: Yes, **you are on your own new branch** called `parallelism-techniques-update`. We are **not** modifying Rob's branch, so your changes are safe.

## Data Source Analysis
You asked where the data for this table comes from, and if there's data for single/multi-host. 
- Currently, the `scripts/update_readme.py` pulls parallelism data from a single *static* file (`support_matrices/parallelism_support_matrix.csv`), which does not have any nightly hardware distinctions or single/multi-host data.
- **Single-host vs multi-host data:** The nightly `parallelism_support_matrix.csv` files only contain raw techniques (CP, DP, EP, PP, SP, TP). However, I did find rows for `"multi-host"` and `"Single-Host-P-D-disaggregation"` in your **nightly feature matrices** (`feature_support_matrix.csv`).

## Proposed Changes

We will modify `scripts/update_readme.py` with the following new logic:

### 1. Update the Data Map (`CSV_MAP`)
We'll update the `CSV_MAP` for `parallelism` to map to the new nightly test paths (the same way we did for core features, including both `v6e` and `v7x` for `vllm` and `flax_nnx`). We'll also cross-reference the `feature_support_matrix.csv` to dynamically pull the `"multi-host"` and `"Single-Host-P-D-disaggregation"` rows.

### 2. New Table Generator (`generate_html_parallelism_table`)
We will create a custom HTML table generator designed specifically for this layout:
- **Columns:** `Features`, `torchax`, and `flax`.
- **Sub-columns:** Under `torchax` and `flax`, we will have `CorrectnessTest` and `PerformanceTest` explicitly surfaced.
- **Hardware merging:** We'll continue the pattern of stacking `v6e` and `v7x` emoji indicators if the results differ, exactly like the "Core Features" table.

Table layout will look like:
| Features | torchax | flax |
| :-- | :-- | :-- |
| | **Correctness** / **Performance** | **Correctness** / **Performance** |
| DP | ✅ (v6) / ❌ (v7) | etc. |

### 3. Add Single/Multi-Host Rows
We will inject `"multi-host"` and `"Single-Host-P-D-disaggregation"` directly into the new Parallelism table since they tightly relate to these techniques.

---

> [!IMPORTANT]
> **Open Question For You:**
> Does pulling `"multi-host"` and `"Single-Host-P-D-disaggregation"` from the feature matrix and adding them into the Parallelism table match your expectations for "uplevel multihost and single host data"? 

If this plan looks good to you, I will begin editing `scripts/update_readme.py` and run the script to see the new layout!
