import random
import numpy as np


def dataset_split(case_lst, split_rate=[7, 1, 2], random_seed=319):
    # split_rate in [train, val, test]
    random.seed(random_seed)

    num_cases = len(case_lst)
    fold_cases_dict = {}

    test_cases = random.sample(
        case_lst, round(num_cases * (split_rate[2] / np.sum(split_rate)))
    )

    train_and_val_cases = np.setdiff1d(case_lst, test_cases)
    for i in range(5):
        fold_cases_dict["fold" + str(i)] = {}
        train_cases = random.sample(
            list(train_and_val_cases),
            round(num_cases * (split_rate[0] / np.sum(split_rate))),
        )
        val_cases = list(np.setdiff1d(train_and_val_cases, train_cases))
        fold_cases_dict["fold" + str(i)]["train"] = sorted(train_cases)
        fold_cases_dict["fold" + str(i)]["val"] = sorted(val_cases)
    fold_cases_dict["test"] = sorted(test_cases)

    return train_and_val_cases, fold_cases_dict
