from evaluators.pathology_evaluator import PathologyEvaluator, best_effort_cover_fast
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from utils.nlp import (
    keyword_positive,
    procedure_checker,
    treatment_alternative_procedure_checker,
)
from icd.procedure_mappings import (
    APPENDECTOMY_PROCEDURES_ICD9,
    APPENDECTOMY_PROCEDURES_ICD10,
    APPENDECTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_APPENDECTOMY_KEYWORDS,
)
from typing import List


class AppendicitisEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of appendicitis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "appendicitis"
        self.alternative_pathology_names = [
            {
                "location": "appendi",
                "modifiers": [
                    "gangren",
                    "infect",
                    "inflam",
                    "abscess",
                    "rupture",
                    "necros",
                    "perf",
                ],
            }
        ]
        self.gracious_alternative_pathology_names = []

        self.required_lab_tests = {
            "Inflammation": INFLAMMATION_LAB_TESTS,
        }
        for req_lab_test_name in self.required_lab_tests:
            self.answers["Correct Laboratory Tests"][req_lab_test_name] = []

        self.neutral_lab_tests = []
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Complete Blood Count (CBC)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Liver Function Panel (LFP)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Renal Function Panel (RFP)"]
        )
        self.neutral_lab_tests.extend(ADDITIONAL_LAB_TEST_MAPPING["Urinalysis"])
        self.neutral_lab_tests = [
            t
            for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Inflammation"]
        ]

        self.answers["Treatment Requested"] = {
            "Appendectomy": False,
            "Antibiotics": False,
            "Support": False,
        }
        self.answers["Treatment Required"] = {
            "Appendectomy": False,
            "Antibiotics": True,
            "Support": True,
        }

    def score_imaging(
        self,
        region: str,
        modality: str,
    ) -> None:
        # Region must be abdomen
        if region == "Abdomen":
            # TODO: Score according to what was done in case and not blindly following guidelines? i.e. if only CT was done by Dr, then give full points
            # Preferred imaging is US and should be done first
            if modality == "Ultrasound":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # CT is acceptable but should be done after US
            if modality == "CT":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
            # MRI is similar to CT and preferred for pregnant patients but should be done after US
            if modality == "MRI":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def score_treatment(self) -> None:
        ### APPENDECTOMY ###
        if (
            procedure_checker(APPENDECTOMY_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(APPENDECTOMY_PROCEDURES_ICD10, self.procedures_icd10)
            or procedure_checker(
                APPENDECTOMY_PROCEDURES_KEYWORDS,
                self.procedures_discharge,
            )
        ):
            self.answers["Treatment Required"]["Appendectomy"] = True

        if procedure_checker(
            APPENDECTOMY_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_APPENDECTOMY_KEYWORDS, self.answers["Treatment"]
        ):
            self.answers["Treatment Requested"]["Appendectomy"] = True

        ### ANTIBIOTICS ###
        # TODO: Check antibiotics against medications table
        if keyword_positive(self.answers["Treatment"], "antibiotic"):
            self.answers["Treatment Requested"]["Antibiotics"] = True

        ### SUPPORT ###
        if (
            keyword_positive(self.answers["Treatment"], "fluid")
            or keyword_positive(self.answers["Treatment"], "analgesi")
            or keyword_positive(self.answers["Treatment"], "pain")
        ):
            self.answers["Treatment Requested"]["Support"] = True

    def generate_feedback(self) -> str:
        messages = []

        imaging_score = self.scores.get("Imaging", 0)
        if imaging_score == 2:
            pass
        elif imaging_score == 1:
            messages.append(
                "Imaging: abdominal imaging was partially appropriate. Set region='Abdomen' and request ultrasound first. CT or MRI are also recommended as follow up options."
            )

        elif imaging_score == 0:
            messages.append(
                "Imaging: no appropriate abdominal imaging was requested. Set region='Abdomen' and request ultrasound as first line. CT or MRI are also recommended as follow up options."
            )
        done_lab_tests: set[int] = set()
        for k in (
            "Laboratory Tests",
            "Laboratory Tests Requested",
            "Requested Laboratory Tests",
            "Labs",
            "Labs Requested",
        ):
            v = self.answers.get(k, None)
            if isinstance(v, list):
                done_lab_tests.update(x for x in v if isinstance(x, int))

        correct = self.answers.get("Correct Laboratory Tests", {}) or {}
        if isinstance(correct, dict):
            for vs in correct.values():
                if isinstance(vs, list):
                    done_lab_tests.update(x for x in vs if isinstance(x, int))

        done_inflam_tests = set(correct.get("Inflammation", []) if isinstance(correct, dict) else [])
        required_inflam_tests = set(self.required_lab_tests.get("Inflammation", []))
        missing_inflam_tests = required_inflam_tests - done_inflam_tests

        if missing_inflam_tests:
            missing_targets = sorted(missing_inflam_tests)
            chosen_keys, _ = best_effort_cover_fast(missing_targets)

            missing_str = (
                ", ".join(chosen_keys)
                if chosen_keys
                else ", ".join(str(x) for x in missing_targets)
            )

            messages.append(
                "Laboratory tests: key inflammation related tests were incomplete. "
                f"Consider ordering: {missing_str}."
            )

        neutral_panel_defs = {
            "Complete Blood Count (CBC)": set(ADDITIONAL_LAB_TEST_MAPPING["Complete Blood Count (CBC)"]),
            "Liver Function Panel (LFP)": set(ADDITIONAL_LAB_TEST_MAPPING["Liver Function Panel (LFP)"]),
            "Renal Function Panel (RFP)": set(ADDITIONAL_LAB_TEST_MAPPING["Renal Function Panel (RFP)"]),
            "Urinalysis": set(ADDITIONAL_LAB_TEST_MAPPING["Urinalysis"]),
        }

        missing_neutral_panel_names = [
            name for name, ids in neutral_panel_defs.items()
            if ids.isdisjoint(done_lab_tests)
        ]

        if missing_neutral_panel_names:
            messages.append(
                "Laboratory tests: the following neutral baseline panels were not obtained. "
                "They are neutral tests, optional but recommended to complete the workup: "
                + ", ".join(missing_neutral_panel_names)
                + "."
            )

        if not messages:
            return (
                "In this case, imaging evaluation and laboratory tests "
                "are broadly consistent with the current scoring rules, "
                "and no major deficiencies were identified."
            )

        return "\n\n".join(messages)
