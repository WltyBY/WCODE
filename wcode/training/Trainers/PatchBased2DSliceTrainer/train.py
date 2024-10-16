import os
import argparse

from wcode.training.Trainers.PatchBased2DSliceTrainer.PatchBased2DSliceTrainer import PatchBased2DSliceTrainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting", type=str, default=None, help="File Name of Setting yaml"
)
parser.add_argument("-f", type=int, default=None, help="fold")
args = parser.parse_args()


if __name__ == "__main__":
    settings_path = os.path.join("./Configs", args.name_setting)
    Trainer = PatchBased2DSliceTrainer(settings_path, args.f)
    Trainer.run_training()
