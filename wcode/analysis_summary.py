import numpy as np
import matplotlib.pyplot as plt
from wcode.utils.file_operations import open_json

if __name__ == "__main__":
    file_path_lst = ["./summary_0.json", "./summary_1.json", "./summary_whole.json"]

    DSC_lst_1_all = []
    DSC_lst_2_all = []
    for file_path in file_path_lst:
        summary = open_json(file_path)
        # a list
        cases_results = summary["metric_per_case"]
        DSC_lst_1 = []
        DSC_lst_2 = []
        case_name_lst = []
        for case in cases_results:
            # case is a dict
            DSC_lst_1.append(case["metrics"]["1"]["Dice"])
            DSC_lst_2.append(case["metrics"]["2"]["Dice"])
            case_name_lst.append(
                case["reference_file"].split("/")[-1].split(".")[0].split("_")[1]
            )
        DSC_lst_1_all.append(DSC_lst_1)
        DSC_lst_2_all.append(DSC_lst_2)

        print(np.std(DSC_lst_1))
        print(np.std(DSC_lst_2))

    x = np.arange(len(case_name_lst))  # the label locations
    width = 0.3  # the width of the bars
    fig, ax = plt.subplots(figsize=(30, 10))
    rects1 = ax.bar(x - width, DSC_lst_1_all[0], width, label="CT")
    rects2 = ax.bar(x, DSC_lst_1_all[1], width, label="Enhanced_CT")
    rects3 = ax.bar(x + width, DSC_lst_1_all[2], width, label="CT+Enhanced_CT")
    ax.set_ylabel("DSC")
    ax.set_title("GTVnd")
    ax.set_xticks(x)
    ax.set_xticklabels(case_name_lst)
    ax.legend()

    plt.savefig("./nnUNet_GTVnd.svg", dpi=600, format="svg", pad_inches=0.0)
    plt.close()

    fig, ax = plt.subplots(figsize=(30, 10))
    rects1 = ax.bar(x - width, DSC_lst_2_all[0], width, label="CT")
    rects2 = ax.bar(x, DSC_lst_2_all[1], width, label="Enhanced_CT")
    rects3 = ax.bar(x + width, DSC_lst_2_all[2], width, label="CT+Enhanced_CT")
    ax.set_ylabel("DSC")
    ax.set_title("GTVp")
    ax.set_xticks(x)
    ax.set_xticklabels(case_name_lst)
    ax.legend()

    plt.savefig("./nnUNet_GTVp.svg", dpi=600, format="svg", pad_inches=0.0)
