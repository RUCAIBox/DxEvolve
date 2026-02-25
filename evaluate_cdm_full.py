from os.path import join
import json
import pickle
import glob
import random
import numpy as np

from dataset.utils import load_hadm_from_file
from utils.logging import read_from_pickle_file
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator


def get_feedback(source, result, patho):
    evaluator = load_evaluator(
        patho
    )  # Reload every time to ensure no state is carried over
    eval = evaluator._evaluate_agent_trajectory(
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
    return eval


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


def load_all_hadm_info(base_mimic_path):
    """加载所有四种疾病的患者数据"""
    all_hadm_info = {}
    
    pathologies = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]
    for patho in pathologies:
        hadm_info_clean = load_hadm_from_file(f"{patho}_hadm_info_clean", base_mimic=base_mimic_path)
        all_hadm_info.update(hadm_info_clean)
    
    return all_hadm_info


def get_patient_pathology(patient_id, id_difficulty, difficulty):
    """根据患者ID确定其疾病类型"""
    for pathology in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
        if patient_id in id_difficulty[pathology][difficulty]:
            return pathology
    return None

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

def check_exam(result, gold):
    reasoning = result['intermediate_steps']
    # print(reasoning)
    cnt_exam = 0
    cnt_actions = 0
    act_images = []
    actions = []

    # load actions in every patient
    for item in extract_executed_actions(result['intermediate_steps']):
        tool = item['tool']
        actions.append(tool)
        if tool == 'Imaging':
            # print(tool, item['tool_input'])
            act_images.append((item['tool_input']['action_input']['modality'], item['tool_input']['action_input']['region']))

        #     print(tool, item['tool_input']['action_input'], item['observation'])

        # elif tool == 'Laboratory Tests':
        #     print(tool, item['tool_input']['action_input'], item['observation'])
        # elif tool == 'Physical Examination':
        #     print(tool, item['observation'])
        # elif tool == 'PubMed Search':
        #     print(tool, item['tool_input']['action_input'], item['observation'])
        # elif tool == 'Experience Search':
        #     # print(tool, item['tool_input']['action_input'], item['observation'])
        #     1
    # print(actions, act_images)
    
    # calculate action coverage and lost exams
    lost_exams = []
    for exam in ['Physical Examination', 'Laboratory Tests', 'Radiology']:
        if exam in gold:
            cnt_exam += 1
            if exam == 'Radiology':
                gold_images = gold['Radiology']
                image_done = False
                lost_images = []
                for gold_image in gold_images:
                    if (gold_image['Modality'], gold_image['Region']) in act_images:
                        image_done = True
                    elif gold_image['Region'] == 'Abdomen':
                        lost_images.append('Abdomen ' + gold_image['Modality'])
                if image_done:
                    cnt_actions += 1
                lost_exams += lost_images
            else:
                if exam in actions:
                    cnt_actions += 1
                else:
                    lost_exams.append(exam)
    return cnt_exam, cnt_actions, lost_exams

def load_gold_data():
    # load all_hadm_info
    hadm_info_clean = {}
    hadm_info_clean_eval = {}
    random.seed(2023)
    np.random.seed(2023)

    pathologies = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']
    for pathology in pathologies:
        dic_temp = load_hadm_from_file(
            f"{pathology}_hadm_info_first_diag",
            base_mimic='hosp/',
        )
        for _id in dic_temp.keys():
            dic_temp[_id]["pathology"] = pathology
        hadm_info_clean = hadm_info_clean | dic_temp

        # dic_temp = load_hadm_from_file(
        #     f"{pathology}_hadm_info_clean",
        #     base_mimic='/root/MIMIC-Clinical-Decision-Making-Dataset',
        # )
        # hadm_info_clean_eval = hadm_info_clean_eval | dic_temp

    # for i, key in enumerate(hadm_info_clean):
    #     print(hadm_info_clean[key])
    #     print(hadm_info_clean_eval[key])
    #     exit()
    return hadm_info_clean

def eval_medical_tests(results):
    hadm_info_clean = load_gold_data()

    eval_medical_exam = {}
    tot_exams = 0
    tot_actions = 0
    lost_dict = {}
    cnt_miss_exams = 0
    # print(hadm_info_clean[23383965])
    # exit()
    for i, key in enumerate(results):
        # print(hadm_info_clean[key])
        cnt_exam, cnt_actions, lost_exams = check_exam(results[key], hadm_info_clean[key])
        # print(results[key])
        print(cnt_exam, cnt_actions, lost_exams)
        tot_exams += cnt_exam
        tot_actions += cnt_actions
        eval_medical_exam[key] = (cnt_exam, cnt_actions, lost_exams)
        for exam in lost_exams:
            if exam not in lost_dict:
                lost_dict[exam] = 1
            else:
                lost_dict[exam] += 1
        if cnt_exam < 3:
            cnt_miss_exams += 1
        # if i == 0:
        #     exit()
    for key in lost_dict:
        print(key, lost_dict[key])
    # print(cnt_miss_exams)
    print(float(tot_actions)/float(tot_exams))
    return eval_medical_exam


if __name__ == '__main__':

    base_hosp = join("mimic-iv", "hosp")

    id_difficulty = pickle.load(open("../MIMIC-Clinical-Decision-Making-Dataset/id_difficulty.pkl", "rb"))

    difficulty = "first_diag"
    models = [
        "Qwen_Qwen3-30B-A3B-Instruct-2507", 
    ]

    pre = "GUIDE_*"
    # pre = "PUBMED_"

    experiment_results = {}
    experiment_evals = {}
    experiment_scores = {}
    for experiment in ["CDM_VANILLA"]: # , "CDM_NOSUMMARY"
        # print(experiment)
        model_scores = {}
        model_evals = {}
        model_results = {}

        for model in models:
            run = f"EXP_GUIDE_PUBMED_TESTNUM=400_full_ZeroShot_Qwen3-30B-A3B-Instruct-2507_20-12-2025_00:06:47_results.pkl"
            assert "result" in run

            all_evals = {}
            all_results = {}
            
            # 加载所有患者数据
            all_hadm_info = load_all_hadm_info("../MIMIC-Clinical-Decision-Making-Dataset/")
            
            results_log_path = f"selected_results/{run}"

            results = []
            if not glob.glob(results_log_path):
                print(f"No results found for full")
                continue
                
            # 读取完整的结果文件
            for r in read_from_pickle_file(glob.glob(results_log_path)[0]):
                results.append(r)
            results = {k: v for d in results for k, v in d.items()}

            # 统计推理中完成医学检查的情况
            eval_medical_exam = eval_medical_tests(results)
        

            # 按疾病类型组织评估结果
            for pathology in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
                all_evals[pathology] = {}
                all_results[pathology] = {}

            # 对每个患者进行评估
            for _id in results:
                # 确定患者的疾病类型
                pathology = get_patient_pathology(_id, id_difficulty, difficulty)
                if pathology is None:
                    print(f"Could not determine pathology for patient {_id}, skipping...")
                    continue
                    
                if _id not in all_hadm_info:
                    print(f"Patient {_id} not found in hadm_info, skipping...")
                    continue
                    
                result = results[_id]
                evaluator = load_evaluator(pathology)
                eval = evaluator._evaluate_agent_trajectory(
                    prediction=result["output"],
                    input=result["input"],
                    reference=(
                        all_hadm_info[_id]["Discharge Diagnosis"],
                        all_hadm_info[_id]["ICD Diagnosis"],
                        all_hadm_info[_id]["Procedures ICD9"],
                        all_hadm_info[_id]["Procedures ICD10"],
                        all_hadm_info[_id]["Procedures Discharge"],
                    ),
                    agent_trajectory=result["intermediate_steps"],
                )
                all_evals[pathology][_id] = eval
                all_results[pathology][_id] = result
                print(result)
                print(eval_medical_exam[_id])
                exit()

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
                for pathology in [
                    "appendicitis",
                    "cholecystitis",
                    "diverticulitis",
                    "pancreatitis",
                ]:
                    if all_evals[pathology] == {}:
                        continue
                    if field in ["Unnecessary Laboratory Tests", "Unnecessary Imaging"]:
                        all_evals[pathology] = count_unnecessary(all_evals[pathology], field)

                    avg, n = calculate_average(all_evals[pathology], field, pathology)

                    avg_scores[field][pathology] = avg
                    avg_samples[field][pathology] = n
                    
                # 添加总体统计（所有疾病合并）
                all_patients_evals = {}
                for pathology in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
                    all_patients_evals.update(all_evals[pathology])
                
                if all_patients_evals:
                    if field in ["Unnecessary Laboratory Tests", "Unnecessary Imaging"]:
                        all_patients_evals = count_unnecessary(all_patients_evals, field)
                    
                    avg, n = calculate_average(all_patients_evals, field, "full")
                    avg_scores[field]["full"] = avg
                    avg_samples[field]["full"] = n

            model_scores[model] = avg_scores

            # if difficulty == "first_diag":
            #     pickle.dump(
            #         all_evals,
            #         open(
            #             f"evaluation_results/{pre}{model}_evals.pkl",
            #             "wb",
            #         ),
            #     )
            #     pickle.dump(
            #         all_results,
            #         open(
            #             f"evaluation_results/{pre}{model}_results.pkl",
            #             "wb",
            #         ),
            #     )
            #     pickle.dump(
            #         avg_scores,
            #         open(
            #             f"evaluation_results/{pre}{model}_scores.pkl",
            #             "wb",
            #         ),
            #     )
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
        print(json.dumps(model_scores, indent=5))
        experiment_results[experiment] = model_results
        experiment_evals[experiment] = model_evals
        experiment_scores[experiment] = model_scores