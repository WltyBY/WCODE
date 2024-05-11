import os

import numpy as np

from wcode.utils.file_operations import open_json


def get_fold_stastic(dataset_name, folder_name, task_name, num_fold=5):
    result_lst = []
    for i in range(num_fold):
        folder_path = os.path.join("./Predictions", dataset_name, folder_name,
                                   task_name + "_fold" + str(i))
        summary_dict = open_json(os.path.join(folder_path, "summary.json"))
        case_lst = [
            case for case in summary_dict.keys() if case != "statistics"
        ]
        for case in case_lst:
            result_lst.append(summary_dict[case]["DSC"])
    print("MEAN:", np.mean(result_lst, axis=0))
    print("STD:", np.std(result_lst, axis=0))


if __name__ == "__main__":
    dataset_name = "SegRap2023"
    folder_name = "CE_And_Tversky"
    task_name = "output_1"
    get_fold_stastic(dataset_name, folder_name, task_name)
