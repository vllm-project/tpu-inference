# Reduce-Scatter Matmul Kernel Pipelining

This document visualizes the pipelining structure of the reduce-scatter matmul kernel.

## Overview

The kernel uses a **bidirectional ring topology** for communication and overlaps computation with data movement through careful pipelining.

**Grid dimensions:** `(num_devices + 2, grid_n, grid_k)`
- `outer_step`: Device iteration (0 to num_devices + 1)
- `bn_i`: Block iteration over n dimension
- `bk_i`: Block iteration over k dimension

## Memory Hierarchy

```mermaid
graph TB
    subgraph "Memory Spaces"
        HBM_X["x_hbm_ref<br/>[m, k]"]
        HBM_Y["y_hbm_ref<br/>[k, n_per_device]"]
        HBM_O["o_hbm_ref<br/>[m_per_device, n_per_device]"]
        HBM_SCRATCH["o_hbm_scratch_ref<br/>[num_devices-1, m_per_device, n_per_device]"]

        VMEM_X["x_vmem_scratch_ref<br/>[m, bk]"]
        VMEM_Y["y_vmem_scratch_ref<br/>[bk, n_per_device]"]
        VMEM_O["o_vmem_scratch_ref<br/>[2, m_per_device, n_per_device]<br/>(double-buffered)"]
        VMEM_ACC["acc_vmem_scratch_ref<br/>[m, n_per_device]<br/>(float32 accumulator)"]
    end

    HBM_X -->|"Local DMA"| VMEM_X
    HBM_Y -->|"Local DMA"| VMEM_Y
    VMEM_X -->|"MXU"| VMEM_ACC
    VMEM_Y -->|"MXU"| VMEM_ACC
    VMEM_ACC -->|"Cast"| VMEM_O
    VMEM_O -->|"Remote DMA<br/>(bidirectional ring)"| HBM_SCRATCH
    VMEM_O -->|"Local DMA"| HBM_O
    HBM_SCRATCH -->|"Local DMA"| VMEM_O

    style HBM_X fill:#e1f5fe
    style HBM_Y fill:#e1f5fe
    style HBM_O fill:#e1f5fe
    style HBM_SCRATCH fill:#e1f5fe
    style VMEM_X fill:#fff3e0
    style VMEM_Y fill:#fff3e0
    style VMEM_O fill:#fff3e0
    style VMEM_ACC fill:#fff3e0
```

## Bidirectional Ring Topology

```mermaid
graph LR
    subgraph "4-Device Ring Example"
        D0["Device 0"]
        D1["Device 1"]
        D2["Device 2"]
        D3["Device 3"]
    end

    D0 -->|"Right →<br/>upper half"| D1
    D1 -->|"Right →<br/>upper half"| D2
    D2 -->|"Right →<br/>upper half"| D3
    D3 -->|"Right →<br/>upper half"| D0

    D0 -->|"← Left<br/>lower half"| D3
    D3 -->|"← Left<br/>lower half"| D2
    D2 -->|"← Left<br/>lower half"| D1
    D1 -->|"← Left<br/>lower half"| D0

    style D0 fill:#c8e6c9
    style D1 fill:#c8e6c9
    style D2 fill:#c8e6c9
    style D3 fill:#c8e6c9
```

## Pipeline Timeline (4 Devices, grid_n=1, grid_k=1)

This shows the execution timeline across `global_step_id`:

```mermaid
gantt
    title Reduce-Scatter Matmul Pipeline Timeline
    dateFormat X
    axisFormat %s

    section Prologue
    Init & Barrier           :done, init, 0, 1
    Start X Local Copy       :done, x_start, 0, 1
    Start Y Local Copy       :done, y_start, 0, 1
    Wait X Local Copy        :done, x_wait, 0, 1
    Wait Y Local Copy        :done, y_wait, 0, 1

    section MXU Compute
    MXU (device 0 portion)   :active, mxu0, 1, 2
    MXU (device 1 portion)   :active, mxu1, 2, 3
    MXU (device 2 portion)   :active, mxu2, 3, 4
    MXU (device 3 portion)   :active, mxu3, 4, 5

    section Remote Copy
    Start 1st Remote Copy    :crit, rc1_start, 1, 2
    Wait 1st Remote Copy     :crit, rc1_wait, 2, 3
    Start 2nd Remote Copy    :crit, rc2_start, 2, 3
    Wait 2nd Remote Copy     :crit, rc2_wait, 3, 4
    Start 3rd Remote Copy    :crit, rc3_start, 3, 4
    Wait 3rd Remote Copy     :crit, rc3_wait, 4, 5

    section Epilogue
    Output Local Copy        :done, o_copy, 5, 6
```

## Detailed Pipeline State Machine

```mermaid
stateDiagram-v2
    [*] --> Prologue: global_step_id == 0

    Prologue --> MXU_Compute: global_step_id >= 1

    state Prologue {
        [*] --> Init
        Init --> StartXCopy: outer_step == 0
        StartXCopy --> StartYCopy
        StartYCopy --> WaitXCopy
        WaitXCopy --> WaitYCopy
        WaitYCopy --> [*]
    }

    state MXU_Compute {
        [*] --> ComputeBlock
        ComputeBlock --> Accumulate: grid_k > 1
        Accumulate --> ComputeBlock: bk_i < grid_k - 1
        Accumulate --> WriteToVMEM: bk_i == grid_k - 1
        ComputeBlock --> WriteToVMEM: grid_k == 1
        WriteToVMEM --> [*]
    }

    MXU_Compute --> StartFirstRemote: global_step_id == gn_by_gk
    StartFirstRemote --> MXU_Compute: more compute

    MXU_Compute --> WaitFirstRemote: global_step_id == 2*gn_by_gk - 1
    WaitFirstRemote --> MXU_Compute: more compute

    MXU_Compute --> StartSubsequentRemote: outer_step > 1 && step % gn_by_gk == 0
    StartSubsequentRemote --> MXU_Compute

    MXU_Compute --> WaitSubsequentRemote: outer_step > 1 && step % gn_by_gk == gn_by_gk - 1
    WaitSubsequentRemote --> MXU_Compute

    MXU_Compute --> Epilogue: global_step_id >= mxu_total_steps + 1

    state Epilogue {
        [*] --> OutputCopy
        OutputCopy --> [*]
    }

    Epilogue --> [*]
```

## Double-Buffering Scheme

The kernel uses double-buffering for the output VMEM scratch to overlap computation with communication:

```mermaid
sequenceDiagram
    participant MXU as MXU Compute
    participant VMEM0 as VMEM Slot 0
    participant VMEM1 as VMEM Slot 1
    participant DMA as Remote DMA
    participant HBM as HBM Scratch

    Note over MXU,HBM: outer_step = 0 (Prologue)
    MXU->>VMEM0: Compute device 0 portion

    Note over MXU,HBM: outer_step = 1
    MXU->>VMEM1: Compute device 1 portion
    VMEM0->>DMA: Start remote copy (slot 0)
    DMA->>HBM: Send to neighbors

    Note over MXU,HBM: outer_step = 2
    MXU->>VMEM0: Compute device 2 portion
    HBM->>VMEM1: Receive from neighbors
    VMEM1->>DMA: Start remote copy (slot 1)

    Note over MXU,HBM: outer_step = 3
    MXU->>VMEM1: Compute device 3 portion
    HBM->>VMEM0: Receive & accumulate
    VMEM0->>DMA: Continue ring

    Note over MXU,HBM: Epilogue
    VMEM0->>HBM: Write final output
```

## Slot Management

| Variable | Formula | Purpose |
|----------|---------|---------|
| `o_hbm_receiving_slot` | `outer_step` | HBM slot to receive remote data |
| `o_hbm_working_slot` | `outer_step - 1` | HBM slot with data to process/forward |
| `o_vmem_receiving_slot` | `outer_step % 2` | VMEM slot for new computation |
| `o_vmem_working_slot` | `(global_step_id - 1) // gn_by_gk % 2` | VMEM slot with completed computation |

## Data Flow Per Device

```mermaid
flowchart TD
    subgraph "Input"
        X["x [m, k]<br/>(replicated)"]
        Y["y [k, n_per_device]<br/>(sharded by n)"]
    end

    subgraph "Compute Phase"
        M1["Matmul Block 0<br/>x[0:m_per_device] @ y"]
        M2["Matmul Block 1<br/>x[m_per_device:2*m_per_device] @ y"]
        M3["Matmul Block 2<br/>..."]
        M4["Matmul Block N-1<br/>x[(N-1)*m_per_device:] @ y"]
    end

    subgraph "Communication Phase"
        RS["Reduce-Scatter<br/>(bidirectional ring)"]
    end

    subgraph "Output"
        O["output [m_per_device, n_per_device]<br/>(sharded by m)"]
    end

    X --> M1 & M2 & M3 & M4
    Y --> M1 & M2 & M3 & M4

    M1 --> RS
    M2 --> RS
    M3 --> RS
    M4 --> RS

    RS --> O

    style X fill:#e3f2fd
    style Y fill:#e3f2fd
    style O fill:#c8e6c9
    style RS fill:#ffecb3
```

## Key Pipeline Conditions

| Condition | When Triggered | Action |
|-----------|----------------|--------|
| `global_step_id == 0` | Start | Initialize accumulators, barrier sync |
| `outer_step == 0` | Prologue | Load x, y from HBM to VMEM |
| `global_step_id ∈ [1, mxu_total_steps]` | Main loop | MXU computation |
| `global_step_id == gn_by_gk` | After 1st device compute | Start first remote copy |
| `global_step_id == 2*gn_by_gk - 1` | Before 2nd device compute ends | Wait for first remote copy |
| `outer_step > 1 && step % gn_by_gk == 0` | Each device boundary | Start subsequent remote copy |
| `outer_step > 1 && step % gn_by_gk == gn_by_gk - 1` | Each device boundary end | Wait for subsequent remote copy |
| `global_step_id >= mxu_total_steps + 1` | Epilogue | Write output to HBM |

## Performance Optimization

The pipelining achieves high efficiency by:

1. **Overlapping compute & communication**: While MXU computes for device i+1, DMA transfers results for device i
2. **Double-buffering**: Two VMEM slots allow concurrent read/write operations
3. **Bidirectional ring**: Sends upper half left, lower half right simultaneously, halving communication time
4. **Block-based computation**: Processes in (bn, bk) blocks to fit in VMEM while maximizing MXU utilization
