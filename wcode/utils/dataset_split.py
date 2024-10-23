import random
import numpy as np


def dataset_split(
    case_lst, five_fold: bool, split_rate=[7, 1, 2], random_seed=319, num_fold=5
):
    # split_rate in [train, val, test]
    random.seed(random_seed)

    num_cases = len(case_lst)
    fold_cases_dict = {}

    if five_fold:
        # only need train and val sets, and cases in each fold are different
        each_fold_val_length = num_cases // num_fold
        remaining_cases = case_lst.copy()
        for i in range(num_fold - 1):
            fold_cases_dict["fold" + str(i)] = {}

            val_cases = random.sample(list(remaining_cases), each_fold_val_length)
            train_cases = list(np.setdiff1d(case_lst, val_cases))

            fold_cases_dict["fold" + str(i)]["train"] = sorted(train_cases)
            fold_cases_dict["fold" + str(i)]["val"] = sorted(val_cases)

            remaining_cases = np.setdiff1d(remaining_cases, val_cases)
        fold_cases_dict["fold" + str(num_fold - 1)] = {}
        train_cases = list(np.setdiff1d(case_lst, remaining_cases))
        fold_cases_dict["fold" + str(num_fold - 1)]["train"] = sorted(list(train_cases))
        fold_cases_dict["fold" + str(num_fold - 1)]["val"] = sorted(list(remaining_cases))

        train_and_val_cases = case_lst
        all_fold_is_the_same = False
    else:
        # train, val and test sets are all needed, and cases in each fold are same
        test_cases = random.sample(
            case_lst, round(num_cases * (split_rate[2] / np.sum(split_rate)))
        )

        train_and_val_cases = list(np.setdiff1d(case_lst, test_cases))
        val_cases = random.sample(
            train_and_val_cases, round(num_cases * (split_rate[1] / np.sum(split_rate)))
        )
        train_cases = list(np.setdiff1d(train_and_val_cases, val_cases))

        for i in range(num_fold):
            fold_cases_dict["fold" + str(i)] = {}
            fold_cases_dict["fold" + str(i)]["train"] = sorted(train_cases)
            fold_cases_dict["fold" + str(i)]["val"] = sorted(val_cases)
        fold_cases_dict["test"] = sorted(test_cases)
        all_fold_is_the_same = True

    return train_and_val_cases, fold_cases_dict, all_fold_is_the_same
