import argparse

from wcode.preprocessing.preprocessor import Preprocessor
from wcode.preprocessing.dataset_analysis import DatasetFingerprintExtractor


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SegRap2023", help="Name of dataset")
parser.add_argument(
    "--five_fold", type=bool, default=True, help="whether do 5-fold val"
)
parser.add_argument(
    "--data_analysis_flag",
    type=bool,
    default=True,
    help="Whether need to analyze dataset",
)
parser.add_argument(
    "--preprocess_flag",
    type=bool,
    default=True,
    help="Whether need to preprocess dataset",
)
parser.add_argument("--num_processes", type=int, default=8, help="Number of workers")
args = parser.parse_args()


def data_analysis_preprocess(
    dataset_name: str,
    five_fold: bool,
    data_analysis_flag: bool,
    preprocess_flag: bool,
    num_processes: int,
):
    if data_analysis_flag:
        print("Analyzing...")
        extractor = DatasetFingerprintExtractor(dataset_name, five_fold)
        extractor.run(num_processes=num_processes)
    if preprocess_flag:
        print("Preprocessing...")
        pp = Preprocessor(dataset_name)
        pp.run(num_processes=num_processes)


if __name__ == "__main__":
    data_analysis_preprocess(
        args.dataset,
        args.five_fold,
        args.data_analysis_flag,
        args.preprocess_flag,
        args.num_processes,
    )
