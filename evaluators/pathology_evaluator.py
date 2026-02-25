import re
from typing import Any, Optional, Sequence, Tuple, List, Dict
from abc import abstractmethod
from torch import Tensor

from thefuzz import fuzz

from langchain.evaluation import AgentTrajectoryEvaluator
from agents.AgentAction import AgentAction
from utils.nlp import keyword_positive, remove_punctuation
from agents.DiagnosisWorkflowParser import InvalidActionError
from models.utils import calculate_log_prob_confidence

from typing import Dict, List, Any
from collections import defaultdict
import pandas as pd
import ast
import pickle
import heapq

NAN_FLUID_KEY = "__nan__"


def _norm_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and x != x:  # NaN
        return NAN_FLUID_KEY
    return " ".join(str(x).strip().split())

def _as_list_of_ints(x: Any) -> List[int]:
    if x is None:
        return []
    if isinstance(x, float) and x != x:  # NaN
        return []
    if isinstance(x, int):
        return [x]
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
        except Exception:
            return []
        if isinstance(v, int):
            return [v]
        if isinstance(v, list):
            return [int(i) for i in v]
        return []
    return []

def _dedup_preserve_order(xs: List[int]) -> List[int]:
    return list(dict.fromkeys(xs))

def build_smart_label_key_to_itemids_map(
    df: pd.DataFrame,
    label_col: str = "label",
    fluid_col: str = "fluid",
    ids_col: str = "corresponding_ids",
) -> Dict[str, List[int]]:
    # 1) 统计 label 是否唯一
    label_counts = df[label_col].astype(str).value_counts(dropna=False)
    is_unique = {lab: (cnt == 1) for lab, cnt in label_counts.items()}

    # 2) 建 map
    tmp: Dict[str, List[int]] = defaultdict(list)
    for label, fluid, ids_cell in zip(df[label_col], df[fluid_col], df[ids_col]):
        label_s = _norm_text(label)
        fluid_s = _norm_text(fluid)

        if is_unique.get(str(label), False):
            key = label_s
        else:
            key = f"({fluid_s}) {label_s}"

        ids = _as_list_of_ints(ids_cell)
        if ids:
            tmp[key].extend(ids)

    return {k: _dedup_preserve_order(v) for k, v in tmp.items()}

lab_test_mapping_path = "../MIMIC-Clinical-Decision-Making-Dataset/lab_test_mapping.pkl"
with open(lab_test_mapping_path, "rb") as f:
    lab_test_mapping_df = pickle.load(f)
label2ids = build_smart_label_key_to_itemids_map(lab_test_mapping_df)
ids2labels = {}
for k, vs in label2ids.items():
    for v in vs:
        ids2labels.setdefault(int(v), []).append(k)

def best_effort_cover_fast(
    subB: List[Any] = [],
    prune_redundant: bool = True,
) -> Tuple[List[str], List[int]]:
    """
    Return (chosen_labels, uncovered_itemids)

    Guarantee:
    - uncovered_itemids only contains ids that have no entry in ids2labels
    - all ids that are coverable (exist in ids2labels) will be covered by chosen_labels
    - may choose extra labels. if prune_redundant=True, try to remove redundant labels safely
    """

    # 1) normalize + unique keep-order
    target: List[int] = []
    seen = set()
    for x in subB:
        try:
            xx = int(x)
        except Exception:
            continue
        if xx not in seen:
            seen.add(xx)
            target.append(xx)

    if not target:
        return [], []

    # 2) build label -> set(coverable ids) only for this target
    label_cover: Dict[str, set] = defaultdict(set)
    uncoverable: List[int] = []
    for v in target:
        ks = ids2labels.get(v)
        if not ks:
            uncoverable.append(v)
            continue
        for k in ks:
            label_cover[k].add(v)

    coverable_set = set(target) - set(uncoverable)
    if not coverable_set:
        return [], uncoverable

    # 3) greedy cover all coverable ids
    uncovered_set = set(coverable_set)
    heap = []
    for k, cov in label_cover.items():
        heapq.heappush(heap, (-len(cov & uncovered_set), k))

    chosen: List[str] = []
    chosen_set = set()

    while uncovered_set and heap:
        neg_gain, k = heapq.heappop(heap)
        cov = label_cover[k]
        true_gain = len(cov & uncovered_set)
        if true_gain == 0:
            continue
        if -neg_gain != true_gain:
            heapq.heappush(heap, (-true_gain, k))
            continue
        chosen.append(k)
        chosen_set.add(k)
        uncovered_set -= cov

    # 兜底。理论上 greedy 结束后 uncovered_set 应为空
    if uncovered_set:
        for v in list(uncovered_set):
            ks = ids2labels.get(v, [])
            if ks:
                chosen_set.add(ks[0])
                uncovered_set.remove(v)

    # 4) prune redundant labels while keeping full coverage
    if prune_redundant and len(chosen_set) > 1:
        # counts[id] = how many chosen labels cover this id
        counts = defaultdict(int)
        for k in chosen_set:
            for v in label_cover.get(k, set()):
                counts[v] += 1

        # try remove smaller covers first
        removable_order = sorted(list(chosen_set), key=lambda k: len(label_cover.get(k, set())))

        for k in removable_order:
            cov = label_cover.get(k, set())
            if not cov:
                chosen_set.discard(k)
                continue

            can_remove = True
            for v in cov:
                if counts[v] <= 1:
                    can_remove = False
                    break

            if can_remove:
                for v in cov:
                    counts[v] -= 1
                chosen_set.discard(k)

    # 5) stable output order by label2ids key order
    chosen_out = [k for k in label2ids.keys() if k in chosen_set]
    return chosen_out, uncoverable

class PathologyEvaluator(AgentTrajectoryEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of <PATHOLOGY>."""

    pathology: str
    alternative_pathology_names: List[Dict]
    gracious_alternative_pathology_names: List[Dict]
    required_lab_tests: Dict[str, List[str]]
    neutral_lab_tests: List[str]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answers = {
            "Diagnosis": "",
            "Diagnostic Confidence": None,
            "Treatment": "",
            "Unnecessary Laboratory Tests": [],
            "Correct Laboratory Tests": {},
            "Unnecessary Imaging": [],
            "Correct Imaging": [],
        }
        self.scores = {
            "Late Physical Examination": 0,
            "Physical Examination": 0,
            "Laboratory Tests": 0,
            "Imaging": 0,
            "Diagnosis": 0,
            "Gracious Diagnosis": 0,
            "Action Parsing": 0,
            "Treatment Parsing": 0,
            "Diagnosis Parsing": 0,
            "Invalid Tools": 0,
            "Rounds": 0,
        }

    def _evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        reference: Optional[
            Tuple[str, List[str], List[str], List[str], List[str], List[str]]
        ] = None,
        diagnosis_probabilities: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> dict:
        self.discharge_diagnosis = reference[0]
        self.icd_diagnoses = reference[1]
        self.procedures_icd9 = reference[2]
        self.procedures_icd10 = reference[3]
        self.procedures_discharge = reference[4]

        for indx, actions in enumerate(agent_trajectory):
            action, observation = actions

            # Keep running tally of custom parsings done
            if isinstance(action.tool_input, dict):
                self.scores["Action Parsing"] = 1

            # Keep running tally of invalid tool requests
            if action.tool == InvalidActionError.invalid_tool_str:
                self.scores["Invalid Tools"] += 1

            # First action should be physical examination
            if action.tool == "Physical Examination":
                self.score_physical_examination(action, indx)

            if action.tool == "Laboratory Tests":
                self.score_laboratory_tests(action)

            if action.tool == "Imaging":
                self.score_imaging_action(action)

        # Record number of rounds needed to reach diagnosis
        self.scores["Rounds"] = len(agent_trajectory)

        # Parse and score diagnosis (and confidence if available)
        self.parse_diagnosis(prediction)
        self.score_diagnosis()
        if diagnosis_probabilities is not None:
            self.answers["Diagnostic Confidence"] = diagnosis_probabilities.squeeze()

        # Parse and score treatment
        if len(agent_trajectory) > 0:
            self.parse_treatment(prediction)
            self.score_treatment()

        return {
            "scores": self.scores,
            "answers": self.answers,
            "message": self.generate_feedback()
        }

    def parse_treatment(self, prediction: str) -> None:
        """Takes prediction string and parses it for treatment. Checks how well the LLM held itself to the desired format.

        Args:
            prediction (str): The prediction string.
        """
        regex = r"Treatment(.*)?:(.*)"
        match = re.search(regex, prediction, flags=re.DOTALL | re.IGNORECASE)
        if match:
            # Instructions were to only write Treatment. If not followed, penalize
            if match.group(1):
                self.scores["Treatment Parsing"] = 1

            treatment = match.group(2).strip()
            self.answers["Treatment"] = treatment

    def parse_diagnosis(self, prediction: str) -> str:
        """Takes prediction string and parses it for diagnosis. Checks how well the LLM held itself to the desired format.

        Args:
            prediction (str): The prediction string.
        """
        custom_parsing = False
        regex = r"(Final )?Diagnosis:([\s\S]*?)(?=[\.\n].*Treatment.*:|$)"
        match = re.search(regex, prediction, flags=re.IGNORECASE)
        if match:
            # Instructions were to write Final Diagnosis. If not followed, penalize
            if not match.group(1):
                custom_parsing = True

            diagnosis = match.group(2).strip()

            # Llama 2 Chat has an intro sentence we want to remove
            modify_check = diagnosis
            diagnosis = re.sub(r"^Based on.*:\n\n", "", diagnosis)
            if modify_check != diagnosis:
                custom_parsing = True

            # Delete extra sections so they are not parsed as diagnosis accidentally
            modify_check = diagnosis
            for section in [
                "rationale",
                "note",
                "recommendation",
                "explanation",
                "finding",
                "other.*diagnos.*include",
                "other.*diagnos.*considered(?: were)?",
                "management",
                "action",
                "plan",
                "reasoning",
                "assessment",
                "justification",
                "tests",
                "additional diagnoses",
                "notification",
                "impression",
                "background",
                "additional findings include",
            ]:
                diagnosis = re.sub(
                    rf"{section}[s]?:.*", "", diagnosis, flags=re.IGNORECASE | re.DOTALL
                )
            if modify_check != diagnosis:
                custom_parsing = True

            # Check for lists with numbers. Extract the first entry
            modify_check = diagnosis
            match = re.search(r"^1\.(.*)", diagnosis, flags=re.MULTILINE)
            if match:
                diagnosis = match.group(1).strip()
            if modify_check != diagnosis:
                custom_parsing = True
                # Remove unwanted explanations
                diagnosis = re.sub(r"[-:].*", "", diagnosis)

            # Check for lists using stars. Extract the first entry
            modify_check = diagnosis
            match = re.search(r"^\*(.*)", diagnosis, flags=re.MULTILINE)
            if match:
                diagnosis = match.group(1).strip()
            if modify_check != diagnosis:
                custom_parsing = True
                # Remove unwanted explanations
                diagnosis = re.sub(r"[-:] .*", "", diagnosis)

            # Llama 2 Chat often does a double newline with an explanation after the diagnosis
            modify_check = diagnosis
            diagnosis = re.sub(r"\n\n.*", "", diagnosis)
            if modify_check != diagnosis:
                custom_parsing = True

            # Check for diagnosis contained in the sentence
            modify_check = diagnosis
            match = re.search(
                r".*?diagnosis[^.\n]*?\bis\b(.*?)[.\n]", diagnosis, flags=re.DOTALL
            )
            if match:
                diagnosis = match.group(1).strip()
            if modify_check != diagnosis:
                custom_parsing = True

            # Check for diagnosis contained in the sentence
            modify_check = diagnosis
            diagnosis = re.sub(
                r".*?patient has", "", diagnosis, count=1, flags=re.DOTALL
            )
            if modify_check != diagnosis:
                custom_parsing = True

            # Check for multiple diagnoses.
            diagnoses = re.split(r"[,.\n]|(?:\s*\b(?:and|or|vs[.]?)\b\s*)", diagnosis)
            diagnoses = [d for d in diagnoses if d != ""]

            # Instructions were to just write a single diagnosis. If not followed, penalize
            if len(diagnoses) > 1:
                custom_parsing = True

            # We do not allow for more than two diagnoses. Indicated great uncertainty and task was to provide a single diagnosis
            # if len(diagnoses) > 2 or len(diagnoses) == 0:
            if len(diagnoses) == 0:
                diagnosis = ""
            # If multiple diagnoses provided, take first as the more likely diagnosis and use for evaluation purposes
            else:
                diagnosis = diagnoses[0]

            if custom_parsing:
                self.scores["Diagnosis Parsing"] = 1

            self.answers["Diagnosis"] = diagnosis.strip()

    def score_diagnosis(self) -> None:
        """Checks predicted pathology against that of the patient. Ensures diagnosis does not contain negation"""
        answer = remove_punctuation(self.answers["Diagnosis"].lower())
        for word in answer.split():
            if fuzz.ratio(word, self.pathology) > 90 and keyword_positive(
                self.answers["Diagnosis"], word
            ):
                self.scores["Diagnosis"] = 1
                self.scores["Gracious Diagnosis"] = 1
                break
        for alternative_patho in self.alternative_pathology_names:
            patho_loc = alternative_patho["location"]
            for patho_mod in alternative_patho["modifiers"]:
                if (
                    patho_loc in answer
                    and patho_mod in answer
                    and keyword_positive(self.answers["Diagnosis"], patho_loc)
                    and keyword_positive(self.answers["Diagnosis"], patho_mod)
                ):
                    self.scores["Diagnosis"] = 1
                    self.scores["Gracious Diagnosis"] = 1
                    break
        # Can be more or less gracious with what is accepted as a correct diagnosis
        for alternative_patho in self.gracious_alternative_pathology_names:
            patho_loc = alternative_patho["location"]
            for patho_mod in alternative_patho["modifiers"]:
                if (
                    patho_loc in answer
                    and patho_mod in answer
                    and keyword_positive(self.answers["Diagnosis"], patho_loc)
                    and keyword_positive(self.answers["Diagnosis"], patho_mod)
                ):
                    self.scores["Gracious Diagnosis"] = 1
                    self.scores["Contains"] = [patho_loc, patho_mod]
                    break

    def score_physical_examination(self, action: AgentAction, indx: int) -> None:
        """A physical examination must be done. It should be done first. Full points only awarded if done first

        Args:
          action (AgentAction): The physical examination action.
          indx (int): The index of the action in the trajectory.
        """
        if indx == 0:
            self.scores["Physical Examination"] = 1
            self.scores["Late Physical Examination"] = 1
        else:
            self.scores["Late Physical Examination"] = 1

    def score_laboratory_tests(self, action: AgentAction) -> None:
        """Score all laboratory tests according to diagnostic guidelines. Each category of test should be performed at least once. Multiple tests in the same category are ignored. Unnecessary tests (that are not neutral i.e. used for eliminating differential diagnoses) are recorded.

        Args:
          action (AgentAction): The laboratory test action.
        """
        for test in action.tool_input["action_input"]:
            for test_category, valid_test_names in self.required_lab_tests.items():
                if test in valid_test_names:
                    # Only provide points for first test occurance in each category
                    if (
                        len(self.answers["Correct Laboratory Tests"][test_category])
                        == 0
                    ):
                        self.scores["Laboratory Tests"] += 1
                    self.answers["Correct Laboratory Tests"][test_category].append(test)
                    break
            else:
                if test not in self.neutral_lab_tests:
                    self.answers["Unnecessary Laboratory Tests"].append(test)

    def score_imaging_action(
        self,
        action: AgentAction,
    ) -> None:
        region = action.tool_input["action_input"]["region"]
        modality = action.tool_input["action_input"]["modality"]
        imaging_dict = {"region": region, "modality": modality}

        # Run patho specific scoring and check if valid modality + region combination
        valid_imaging_combination = self.score_imaging(region, modality)
        if (
            not valid_imaging_combination
            or imaging_dict in self.answers["Correct Imaging"]
        ):
            self.answers["Unnecessary Imaging"].append(imaging_dict)
        else:
            self.answers["Correct Imaging"].append(imaging_dict)

    @abstractmethod
    def score_imaging(self, region: str, modality: str) -> None:
        """Score all imaging tests according to diagnostic guidelines. Unnecessary tests are recorded. Also check if valid modality + region combination.

        Args:
          region (str): The region of the body to be scanned.
          modality (str): The imaging modality.

        Returns:
          bool: True if valid modality + region combination, False otherwise.
        """

    @abstractmethod
    def score_treatment(self) -> None:
        """Score a treatment according to diagnostic guidelines."""


    def generate_feedback(self) -> str:
        messages: List[str] = []

        pe = self.scores.get("Physical Examination", 0)
        late_pe = self.scores.get("Late Physical Examination", 0)

        if pe == 1:
            pass
        elif late_pe == 1 and pe == 0:
            messages.append(
                "History and examination: A physical examination should be done at the beginning of the encounter."
            )
        else:
            messages.append(
                "A focused physical examination is expected for this "
                "case."
            )

        lab_summary_parts = []

        for category, required_tests in self.required_lab_tests.items():
            done_tests = self.answers["Correct Laboratory Tests"].get(category, [])
            
            done_chosen, done_uncovered = best_effort_cover_fast(done_tests)
            done_str = ", ".join(done_chosen) if done_chosen else ""
            missing_tests = [t for t in required_tests if t not in done_tests]
            missing_chosen, missing_uncovered = best_effort_cover_fast(missing_tests)
            missing_str = ", ".join(missing_chosen) if missing_chosen else ""

            if done_chosen and missing_chosen:
                lab_summary_parts.append(
                    f"{category}: done ({done_str}), missing ({missing_str})"
                )
            elif done_chosen:
                lab_summary_parts.append(
                    f"{category}: done ({done_str})"
                )
            elif missing_chosen:
                lab_summary_parts.append(
                    f"{category}: missing ({missing_str})"
                )

        if lab_summary_parts:
            messages.append(
                "Laboratory tests: " + "; ".join(lab_summary_parts) + "."
            )

        imaging_score = self.scores.get("Imaging", 0)
        if imaging_score == 0:
            messages.append(
                "Imaging: No appropriate imaging study for this pathology was recorded. "
            )

        unnecessary_imaging = self.answers.get("Unnecessary Imaging", [])
        if unnecessary_imaging:
            formatted: List[str] = []
            for item in unnecessary_imaging:
                if isinstance(item, dict):
                    region = item.get("region", "Unknown region")
                    modality = item.get("modality", "Unknown modality")
                    formatted.append(f"{modality} of {region}")
                else:
                    formatted.append(str(item))
            unique_imaging = ", ".join(sorted(set(formatted)))
            messages.append(
                "Imaging: Some imaging requests were not considered appropriate for "
                f"this case: {unique_imaging}. Non targeted imaging leads to "
                "deductions in the imaging section."
            )

        diag_answer = self.answers.get("Diagnosis", "").strip()
        diag_score = self.scores.get("Diagnosis", 0)
        gracious_score = self.scores.get("Gracious Diagnosis", 0)

        target_diag = getattr(self, "pathology", "") or ""
        target_diag = target_diag.strip()
        contains = self.scores.get("Contains", [])

        if not diag_answer:
            messages.append(
                "Diagnosis: No final diagnosis could be extracted from the model "
                "output."
            )
        elif diag_score == 1:
            pass
        elif gracious_score == 1:
            if target_diag and isinstance(contains, list) and len(contains) == 2:
                loc, mod = contains
                messages.append(
                    f"Diagnosis: You wrote '{diag_answer}', while the target diagnosis is '{target_diag}'. "
                    f"Your answer correctly included '{loc}' and '{mod}', so it is counted as a reasonable "
                    "alternative rather than completely wrong. However, it still does not clearly name the "
                    "target diagnosis, so it does not receive full credit."
                )
            elif target_diag:
                messages.append(
                    f"Diagnosis: You wrote '{diag_answer}', which is close to the target '{target_diag}'. "
                    "It is counted as a reasonable alternative, but not a strict exact match."
        )
        else:
            pass
            
        if not messages:
            return (
                "No major deficiencies were detected across examination, testing, "
                "imaging, diagnosis and treatment according to the current scoring "
                "rules."
            )

        return "\n\n".join(messages)

    def _evaluate_diagnoise_correctness(
        self,
        prediction: str,
        input: str = None,
        reference: Tuple[str] = None,
    ):
        if reference is not None:
            self.discharge_diagnosis = reference[0]
        self.scores["Diagnosis"] = 0
        self.scores["Gracious Diagnosis"] = 0
        self.scores["Diagnosis Parsing"] = 0
        self.answers["Diagnosis"] = ""

        self.parse_diagnosis(prediction)
        self.score_diagnosis()

        return {
            "scores": self.scores,
            "answers": self.answers,
            "message": None,
        }
