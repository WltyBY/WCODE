import argparse

from wcode.preprocessing.preprocessor import Preprocessor
from wcode.preprocessing.dataset_analysis import DatasetFingerprintExtractor


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="LNQ2023", help="Name of dataset"
)
parser.add_argument("--preprocess_config", type=str, default="2d", help="2d or 3d")
parser.add_argument(
    "--kfold", type=int, default=5, help="whether do 5-fold cross-validation"
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
parser.add_argument(
    "--seed",
    type=int,
    default=319,
    help="Whether need to preprocess dataset",
)
parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
args = parser.parse_args()


def data_analysis_preprocess(
    dataset_name: str,
    preprocess_config: str,
    kfold: int,
    data_analysis_flag: bool,
    preprocess_flag: bool,
    num_workers: int,
    random_seed: int,
):
    if data_analysis_flag:
        print("Analyzing...")
        extractor = DatasetFingerprintExtractor(
            dataset_name=dataset_name,
            kfold=kfold,
            split_rate=[7, 1, 2],
            random_seed=random_seed,
        )
        extractor.run(num_workers=num_workers)
    if preprocess_flag:
        print("Preprocessing...")
        pp = Preprocessor(dataset_name, random_seed=random_seed)
        pp.run(preprocess_config, num_workers=num_workers)


if __name__ == "__main__":
    data_analysis_preprocess(
        dataset_name=args.dataset,
        preprocess_config=args.preprocess_config,
        kfold=args.kfold,
        data_analysis_flag=args.data_analysis_flag,
        preprocess_flag=args.preprocess_flag,
        num_workers=args.num_workers,
        random_seed=args.seed,
    )
