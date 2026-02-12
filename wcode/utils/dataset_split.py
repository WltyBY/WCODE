import random
from typing import List, Dict


def dataset_split(
    case_lst: List[str],
    num_fold: int = None,
    split_rate: List[int] = [7, 1, 2],
    random_seed: int = 319,
) -> Dict:
    """
    Dataset Splitter

    Args:
        case_lst: List of filenames (e.g., ['patient_0001', ...])
        num_fold: Number of folds for cross-validation. If None, do train/val/test split
        split_rate: Ratio list [train, val, test] for single split
        random_seed: Random seed for reproducibility

    Returns:
        split: Dict
            If num_fold is not None:
                Dictionary of K-Fold splits (each fold are different)
            Else:
                Dictionary of train/val/test split (all folds are the same)
        all_fold_is_the_same: Bool
            Whether all folds are the same (True for train/val/test split)
    """
    if num_fold is not None:
        all_fold_is_the_same, split = create_kfold_splits(
            case_lst, num_fold=num_fold, random_seed=random_seed
        )
    else:
        all_fold_is_the_same, split = create_train_val_test_split(
            case_lst, split_rate=split_rate, random_seed=random_seed
        )

    return split, all_fold_is_the_same


def create_kfold_splits(
    case_lst: List[str], num_fold: int = 5, random_seed: int = 319
) -> Dict[str, Dict[str, List[str]]]:
    """
    K-Fold Cross Validation - For casename lists

    Args:
        case_lst: List of filenames (e.g., ['patient_0001', ...])
        num_fold: Number of folds, default 5
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary format: {
            'fold_0': {'train': ['patient_0001', ...], 'val': ['patient_0005', ...]},
            'fold_1': {'train': ['patient_0001', ...], 'val': ['patient_0003', ...]},
            'fold_2': {'train': ['patient_0003', ...], 'val': ['patient_0001', ...]},
            ...
        }
    """
    random.seed(random_seed)

    cases = case_lst.copy()
    random.shuffle(cases)

    num_cases = len(cases)
    base_fold_size = num_cases // num_fold
    remainder = num_cases % num_fold

    folds = {}
    start_idx = 0
    for i in range(num_fold):
        current_fold_size = base_fold_size + (1 if i < remainder else 0)

        val_start = start_idx
        val_end = start_idx + current_fold_size

        val_cases = cases[val_start:val_end]
        train_cases = cases[:val_start] + cases[val_end:]

        folds[str(i)] = {
            "train": sorted(train_cases),
            "val": sorted(val_cases),
        }

        start_idx = val_end

    all_fold_is_the_same = False
    return all_fold_is_the_same, folds


def create_train_val_test_split(
    case_lst: List[str], split_rate: List[int] = [7, 1, 2], random_seed: int = 319
) -> Dict[str, List[str]]:
    """
    Train/Val/Test Split - For filename lists

    Args:
        case_lst: List of filenames
        split_rate: Ratio list [train, val, test], e.g., [7, 1, 2] means 70%/10%/20%
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary format: {
            'train': [...],
            'val': [...],
            'test': [...]
        }
    """
    random.seed(random_seed)

    # Copy and shuffle
    cases = case_lst.copy()
    random.shuffle(cases)

    num_cases = len(cases)
    total_rate = sum(split_rate)

    # Calculate split points
    train_cut = int(num_cases * split_rate[0] / total_rate)
    val_cut = train_cut + int(num_cases * split_rate[1] / total_rate)

    folds = {}
    # Split dataset (sorted for readability)
    one_fold_split = {
        "train": sorted(cases[:train_cut]),  # Training set
        "val": sorted(cases[train_cut:val_cut]),  # Validation set
    }
    for i in range(5):
        folds[str(i)] = one_fold_split
    folds["test"] = sorted(cases[val_cut:]) # Test set
    all_fold_is_the_same = True
    return all_fold_is_the_same, folds


# Verification function
def verify_kfold_splits(folds: Dict, total_cases: List[str]):
    """Verify that all cases are used exactly once across validation sets"""
    all_val_cases = []
    for fold_name, data in folds.items():
        # Check no overlap with train
        overlap = set(data["train"]) & set(data["val"])
        assert len(overlap) == 0, f"Data leakage in {fold_name}!"
        all_val_cases.extend(data["val"])

    # Check all cases are used exactly once
    assert sorted(all_val_cases) == sorted(
        total_cases
    ), "Some cases are missing or duplicated!"
    print(
        f"✅ Verification passed: {len(total_cases)} cases used exactly once across folds"
    )


if __name__ == "__main__":
    # Test with non-divisible number
    cases = [f"case_{i:03d}.nii" for i in range(23)]

    _, folds = create_kfold_splits(cases, num_fold=5)

    # Print fold sizes
    for i, data in folds.items():
        print(f"{i}: val={len(data['val'])}, train={len(data['train'])}")

    # Verify correctness
    verify_kfold_splits(folds, cases)

    # Split with 70% / 10% / 20% ratio
    _, splits = create_train_val_test_split(cases, split_rate=[7, 1, 2])

    print(f"Train: {len(splits['0']['train'])} samples")
    print(f"Val: {len(splits['0']['val'])} samples")  
    print(f"Test: {len(splits['test'])} samples")

    # Verify mutual exclusivity
    assert len(set(splits['0']['train']) & set(splits['0']['val']) & set(splits['test'])) == 0
