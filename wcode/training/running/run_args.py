import argparse


def parse_device(v):
    """Custom type: convert string to list[int] or None."""
    if v.lower() == "all":
        return None  # Do not set CUDA_VISIBLE_DEVICES, use all
    if v == "-1":
        return []  # Empty list -> CPU
    try:
        return [int(x) for x in v.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid format: {v}")


def build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dataset", "-d", type=str, help="Name of dataset", required=True)
    p.add_argument(
        "--setting",
        "-s",
        type=str,
        help="File Name of Setting yaml, or you can just give the absolute path of the file.",
        required=True,
    )
    p.add_argument(
        "--method_name",
        "-m",
        type=str,
        help="Method name for saving logs and models.",
        required=True,
    )
    p.add_argument(
        "--fold",
        "-f",
        type=str,
        help="Fold of dataset. Can be 'all' for using both train and val split for training.",
        required=True,
    )
    p.add_argument(
        "--num_epoch",
        type=int,
        help="Number of epochs for training",
        required=True,
    )
    p.add_argument(
        "--warmup_epoch",
        type=int,
        default=0,
        help="Number of epochs for warmup",
    )
    p.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        help="Batch size for training",
        required=True,
    )
    p.add_argument(
        "--gpu",
        type=parse_device,
        default="0",
        help="CPU=-1; GPU ID=0; Multi-GPU=0,1,2...; All GPUs=all",
    )
    p.add_argument(
        "--continue_train",
        "--c",
        action="store_true",
        help="If call this param, continue training based on the given Params: --dataset/--d, --fold/-f, --method_name/-m, etc.",
    )
    p.add_argument(
        "--pretrained_weight",
        type=str,
        help="Continue training from a specific checkpoint folder.",
    )
    p.add_argument("--num_workers", type=int, default=12, help="Number of workers")
    p.add_argument("--seed", type=int, default=319, help="Random seed")
    p.add_argument(
        "--no_deterministic",
        action="store_false",
        help="Enable deterministic behavior. Call this param to disable it.",
    )
    return p
