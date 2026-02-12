import argparse

from wcode.training.running.run_args import build_train_parser
from wcode.training.Trainers.Fully.PatchBased2DSliceTrainer.PatchBased2DSliceTrainer import (
    PatchBased2DSliceTrainer,
)

parser = argparse.ArgumentParser(
    parents=[build_train_parser()], description="Training-specific args"
)
parser.add_argument(
    "--w_ce", type=float, default=1.0, help="weight of CrossEntropy Loss"
)
parser.add_argument("--w_dice", type=float, default=1.0, help="weight of Dice Loss")
parser.add_argument(
    "--w_class",
    nargs="+",
    type=float,
    default=None,
    help="weight of class in CrossEntropyLoss",
)
args = parser.parse_args()


if __name__ == "__main__":
    Trainer = PatchBased2DSliceTrainer(training_args=args)
    Trainer.run_training()