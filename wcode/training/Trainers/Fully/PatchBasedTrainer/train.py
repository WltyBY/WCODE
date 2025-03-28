import os
import argparse

from wcode.training.Trainers.Fully.PatchBasedTrainer.PatchBasedTrainer import (
    PatchBasedTrainer,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting",
    type=str,
    default=None,
    help="File Name of Setting yaml, or you can just give the absolute path of the file.",
)
parser.add_argument("-f", type=str, default=None, help="fold")
parser.add_argument(
    "--w_ce", type=float, default=1.0, help="weight of CrossEntropyLoss"
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
    settings_path = os.path.join("./Configs", args.name_setting)
    Trainer = PatchBasedTrainer(
        settings_path, args.f, args.w_ce, args.w_dice, args.w_class
    )
    Trainer.run_training()
