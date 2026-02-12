import numpy as np

from wcode.utils.file_operations import open_json, save_json
from wcode.utils.json_export import recursive_fix_for_json_export

if __name__ == "__main__":
    # file_path_lst = ["./Logs/LNQ2023/only_P1/0_tversky_alpha_0.3_awce_beta_1.0_consis_weight_0.1_rampup_epoch_100_update_way_least_select_way_merge_num_prototype_2_memory_rate_0.999/fold_0/test_best/summary.json"]

    base = "./Logs/CTLymphNodes02/BS8/tversky_alpha_0.4_awce_beta_1.0_consis_weight_1.0_rampup_epoch_100_update_way_least_select_way_merge_num_prototype_3_memory_rate_0.999"
    file_path_lst = [
        base + "/fold_{}/validation/summary.json".format(i) for i in range(5)
    ]

    results = {}
    having_results = {}
    Hallucinations = {}
    metric_per_case = {}
    for file_path in file_path_lst:
        summary = open_json(file_path)

        cases_results = summary["metric_per_case"]
        metric_per_case.update(cases_results)
        for case in cases_results:
            # case is a dict
            for key in cases_results[case].keys():
                key = int(key)
                if not results.__contains__(key):
                    results[key] = dict()
                    having_results[key] = dict()
                    Hallucinations[key] = {
                        "num_of_case": 0,
                        "num_of_case_have_hallucinations": 0,
                    }
                    for metric in cases_results[case][str(key)].keys():
                        results[key][metric] = []
                        having_results[key][metric] = []

                for metric in results[key].keys():
                    results[key][metric].append(cases_results[case][str(key)][metric])
 
                    if cases_results[case][str(key)]["n_gt"] != 0:
                        having_results[key][metric].append(
                            cases_results[case][str(key)][metric]
                        )

                if cases_results[case][str(key)]["n_gt"] == 0:
                    Hallucinations[key]["num_of_case"] += 1

                if (
                    cases_results[case][str(key)]["n_gt"] == 0
                    and cases_results[case][str(key)]["n_pred"] != 0
                ):
                    Hallucinations[key]["num_of_case_have_hallucinations"] += 1

    for class_name in results.keys():
        results[class_name]["DSC_agg"] = (2 * np.sum(results[class_name]["TP"])) / (
            np.sum(results[class_name]["n_gt"]) + np.sum(results[class_name]["n_pred"])
        )
        for metric in results[class_name].keys():
            if metric == "DSC_agg":
                continue
            results[class_name][metric] = "{:f}+{:f}".format(
                np.nanmean(results[class_name][metric]),
                np.nanstd(results[class_name][metric]),
            )
            having_results[class_name][metric] = "{:f}+{:f}".format(
                np.nanmean(having_results[class_name][metric]),
                np.nanstd(having_results[class_name][metric]),
            )
        if Hallucinations[class_name]["num_of_case"] != 0:
            Hallucinations[class_name]["hallucinations_rate"] = (
                Hallucinations[class_name]["num_of_case_have_hallucinations"]
                / Hallucinations[class_name]["num_of_case"]
            )
        else:
            Hallucinations[class_name]["hallucinations_rate"] = np.nan

    recursive_fix_for_json_export(results)
    recursive_fix_for_json_export(having_results)
    recursive_fix_for_json_export(Hallucinations)
    summary = {
        "results": results,
        "results_with_positive_case": having_results,
        "Hallucinations": Hallucinations,
        "z_metric_per_case": metric_per_case,
    }
    save_json(summary, "./summary_analysis.json")
