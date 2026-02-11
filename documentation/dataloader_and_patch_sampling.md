# Dataloader Design in v0

The *dataloader* design follows a two-stage pipeline: **Data Loading** → **Batch Processing**

## Implementation Details

### 1. `Dataset.__getitem__`

- **Responsibility**: Raw data reading only
- **Scope**: Minimal I/O operations, no preprocessing

### 2. `Collate_fn`

- **Responsibility**: Per-case processing and batch assembly
- **Operations**:
  - Patch sampling
  - Data augmentation
- **Output Format**: `B, C, (Z,) Y, X` Tensor for training

> **Note**: `(Z,)` indicates an optional depth dimension for 3D data

# **Design Evolution: v1**

In **v1**, we relocated **patch sampling** and **data augmentation** from `Collate_fn` to `Dataset.__getitem__`.

In `Collate_fn`, tensors are manually stacked along dimension `B`, while the remaining data are packed as lists.

This change was motivated by the fact that operations in `Collate_fn` do not benefit from **multi-threading** acceleration by default, whereas `Dataset.__getitem__` does.

**Trade-off**: In v0, we enforced a precise number of oversampled cases (patches centered on foreground class pixels) per batch. This is no longer efficiently achievable in v1.

**v1 Approach**: Each case is determined to be oversampled based on probability. While this cannot guarantee an exact count of oversampled cases per batch, it is statistically equivalent to the v0 approach, except when using **BatchNorm** with a very small batch size. But when the batch size is small, why not use **InstanceNorm**?

> **Note**: Of course, precise control as in v0 can be achieved in v1: simply sample both a regular patch and an oversampled patch for each case in the `Dataset.__getitem__`, then statistically select one patch in `Collate_fn`.
>
> However, as noted above, since the probabilistic approach is statistically equivalent, we opt for the simpler implementation.