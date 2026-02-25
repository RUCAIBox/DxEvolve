from os.path import join
import pickle
import glob

from dataset.utils import load_hadm_from_file
from utils.logging import read_from_pickle_file
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator
from evaluators.pathology_evaluator import best_effort_cover_fast


def extract_executed_actions(intermediate_steps):
    """
    从 agent intermediate_steps 中提取所有实际执行的动作
    返回 List[Dict]
    """
    actions = []

    for step in intermediate_steps:
        if not isinstance(step, tuple) or len(step) != 2:
            continue

        action, observation = step

        # 防御式检查
        if not hasattr(action, "tool"):
            continue

        actions.append({
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log,
            "observation": observation
        })

    return actions


def get_feedback(source, result, patho, correctness_only=False):
    evaluator = load_evaluator(
        patho
    )  # Reload every time to ensure no state is carried over
    if correctness_only:
        eval_result = evaluator._evaluate_diagnoise_correctness(
            prediction=result["output"],
            input=result["input"],
            reference=(source["Discharge Diagnosis"],),
        )
        return eval_result
    else:
        eval_result = evaluator._evaluate_agent_trajectory(
            prediction=result["output"],
            input=result["input"],
            reference=(
                source["Discharge Diagnosis"],
                source["ICD Diagnosis"],
                source["Procedures ICD9"],
                source["Procedures ICD10"],
                source["Procedures Discharge"],
            ),
            agent_trajectory=result["intermediate_steps"],
        )

    actions = extract_executed_actions(result["intermediate_steps"])
    has_pe = "Physical Examination" in source
    available_scans = [{'region': test['Region'], 'modality': test['Modality']} for test in source["Radiology"]]
    available_lab_tests, _ = best_effort_cover_fast(list(source['Laboratory Tests'].keys()))
    
    parts = []
    if has_pe:
        parts.append("The clinician used Physical Examination")

    if available_scans:
        scans_str = ", ".join(f'{x["modality"]} of the {x["region"]}' for x in available_scans)
        parts.append(f"For Imaging, they ordered {scans_str}")

    if available_lab_tests:
        parts.append(f"For Laboratory Tests, they ordered {', '.join(available_lab_tests)}")
    eval_result["clinician"] = ". ".join(parts) + "." if parts else None
    
    return eval_result


def load_evaluator(pathology):
    # Load desired evaluator
    if pathology == "appendicitis":
        evaluator = AppendicitisEvaluator()
    elif pathology == "cholecystitis":
        evaluator = CholecystitisEvaluator()
    elif pathology == "diverticulitis":
        evaluator = DiverticulitisEvaluator()
    elif pathology == "pancreatitis":
        evaluator = PancreatitisEvaluator()
    else:
        raise NotImplementedError
    return evaluator


def calculate_average(evals, field, pathology):
    average = 0
    for patient in evals.keys():
        average += evals[patient]["scores"][field]

    average /= len(evals)
    # print(f'{pathology}: {average:0.02} (n={len(evals)})'.rjust(30))
    return average, len(evals)


def calculate_percentages(evals, field):
    for patient in evals.keys():
        evals[patient]["scores"][field] = (
            evals[patient]["scores"][field[: -len(" Percentage")]]
            / evals[patient]["max_scores"][field[: -len(" Percentage")]]
        )
    return evals


def count_unnecessary(evals, field):
    for patient in evals.keys():
        evals[patient]["scores"][field] = len(evals[patient]["answers"][field])
    return evals


if __name__ == '__main__':

    base_hosp = join("mimic-iv", "hosp")

    id_difficulty = pickle.load(open("../MIMIC-Clinical-Decision-Making-Dataset/id_difficulty.pkl", "rb"))
    difficulty = "first_diag"
    models = [
        "Qwen_Qwen3-30B-A3B-Instruct-2507", 
    ]

    pre = "PUBMED_"

    experiment_results = {}
    experiment_evals = {}
    experiment_scores = {}
    for experiment in ["CDM_VANILLA"]: # , "CDM_NOSUMMARY"
        # print(experiment)
        model_scores = {}
        model_evals = {}
        model_results = {}

        for model in models:
            run = f"_ZeroShot_{model}_*_results.pkl"
            assert "result" in run

            all_evals = {}
            all_results = {}
            for patho in [
                "appendicitis",
                "cholecystitis",
                "diverticulitis",
                "pancreatitis",
            ]:
                # Load patient data
                hadm_info_clean = load_hadm_from_file(f"{patho}_hadm_info_clean", base_mimic="../MIMIC-Clinical-Decision-Making-Dataset/")
                all_evals[patho] = {}
                all_results[patho] = {}

                # results_log_path = f"evaluation_results/{experiment}/{patho}{run}"
                results_log_path = f"selected_results/{pre}{patho}{run}"

                results = []
                # print(results_log_path)
                if not glob.glob(results_log_path):
                    print(f"No results found for {patho}")
                    continue
                for r in read_from_pickle_file(glob.glob(results_log_path)[0]):
                    results.append(r)
                results = {k: v for d in results for k, v in d.items()}

                for _id in id_difficulty[patho][difficulty]:
                    if _id not in results:
                        print(f"Skipping {_id} | {glob.glob(results_log_path)[0]}")
                        continue
                    result = results[_id]
                    evaluator = load_evaluator(
                        patho
                    )  # Reload every time to ensure no state is carried over
                    eval = evaluator._evaluate_agent_trajectory(
                        prediction=result["output"],
                        input=result["input"],
                        reference=(
                            hadm_info_clean[_id]["Discharge Diagnosis"],
                            hadm_info_clean[_id]["ICD Diagnosis"],
                            hadm_info_clean[_id]["Procedures ICD9"],
                            hadm_info_clean[_id]["Procedures ICD10"],
                            hadm_info_clean[_id]["Procedures Discharge"],
                        ),
                        agent_trajectory=result["intermediate_steps"],
                    )
                    all_evals[patho][_id] = eval
                    all_results[patho][_id] = result

            model_evals[model] = all_evals
            model_results[model] = all_results
            avg_scores = {}
            avg_samples = {}

            for field in [
                "Diagnosis",
                "Gracious Diagnosis",
                "Physical Examination",
                "Late Physical Examination",
                "Action Parsing",
                "Treatment Parsing",
                "Diagnosis Parsing",
                "Rounds",
                "Invalid Tools",
                "Unnecessary Laboratory Tests",
                "Unnecessary Imaging",
            ]:
                avg_scores[field] = {}
                avg_samples[field] = {}
                for patho in [
                    "appendicitis",
                    "cholecystitis",
                    "diverticulitis",
                    "pancreatitis",
                ]:
                    if all_evals[patho] == {}:
                        continue
                    if field in ["Unnecessary Laboratory Tests", "Unnecessary Imaging"]:
                        all_evals[patho] = count_unnecessary(all_evals[patho], field)

                    avg, n = calculate_average(all_evals[patho], field, patho)

                    avg_scores[field][patho] = avg
                    avg_samples[field][patho] = n
            model_scores[model] = avg_scores

            if difficulty == "first_diag":
                pickle.dump(
                    all_evals,
                    open(
                        f"evaluation_results/{pre}{model}_evals.pkl",
                        "wb",
                    ),
                )
                pickle.dump(
                    all_results,
                    open(
                        f"evaluation_results/{pre}{model}_results.pkl",
                        "wb",
                    ),
                )
                pickle.dump(
                    avg_scores,
                    open(
                        f"evaluation_results/{pre}{model}_scores.pkl",
                        "wb",
                    ),
                )
        if difficulty == "first_diag":
            pickle.dump(
                model_evals,
                open(
                    f"evaluation_results/{pre}evals.pkl",
                    "wb",
                ),
            )
            pickle.dump(
                model_results,
                open(
                    f"evaluation_results/{pre}results.pkl",
                    "wb",
                ),
            )
            pickle.dump(
                model_scores,
                open(
                    f"evaluation_results/{pre}scores.pkl",
                    "wb",
                ),
            )
        experiment_results[experiment] = model_results
        experiment_evals[experiment] = model_evals
        experiment_scores[experiment] = model_scores
