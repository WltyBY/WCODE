import os
import argparse

from wcode.training.Trainers.HintonDistillTrainer.HintonDistillTrainer import HintonDistillTrainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name_setting", type=str, default=None, help="File Name of Setting yaml"
)
parser.add_argument("-f", type=int, default=None, help="fold")
parser.add_argument("-t", type=float, default=None, help="temperature index")
parser.add_argument("-a", type=float, default=None, help="weight of distillation loss")
args = parser.parse_args()


if __name__ == "__main__":
    settings_path = os.path.join("./Configs", args.name_setting)
    Trainer = HintonDistillTrainer(settings_path, args.f, args.t, args.a)
    Trainer.run_training()