from evaluators.pathology_evaluator import PathologyEvaluator
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from utils.nlp import (
    keyword_positive,
    procedure_checker,
    treatment_alternative_procedure_checker,
)
from icd.procedure_mappings import (
    CHOLECYSTECTOMY_PROCEDURES_ICD9,
    CHOLECYSTECTOMY_PROCEDURES_ICD10,
    CHOLECYSTECTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_CHOLECYSTECTOMY_KEYWORDS,
)
from typing import List


class CholecystitisEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of cholecystitis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "cholecystitis"
        self.alternative_pathology_names = [
            {
                "location": "gallbladder",
                "modifiers": [
                    "gangren",
                    "infect",
                    "inflam",
                    "abscess",
                    "necros",
                    "perf",
                ],
            },
            {
                "location": "cholangitis",
                "modifiers": [
                    "cholangitis",
                ],
            },
        ]
        self.gracious_alternative_pathology_names = [
            {"location": "acute gallbladder", "modifiers": ["disease", "attack"]},
            {"location": "acute biliary", "modifiers": ["colic"]},
        ]

        self.required_lab_tests = {
            "Inflammation": INFLAMMATION_LAB_TESTS,
            "Liver": [
                50861,  # "Alanine Aminotransferase (ALT)",
                50878,  # "Asparate Aminotransferase (AST)",
            ],
            "Gallbladder": [
                50883,  # "Bilirubin",
                50927,  # "Gamma Glutamyltransferase",
            ],
        }
        for req_lab_test_name in self.required_lab_tests:
            self.answers["Correct Laboratory Tests"][req_lab_test_name] = []

        self.neutral_lab_tests = []
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Complete Blood Count (CBC)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Renal Function Panel (RFP)"]
        )
        self.neutral_lab_tests.extend(
            [
                50863,  # "Alkaline Phosphatase"
            ]
        )
        self.neutral_lab_tests.extend(ADDITIONAL_LAB_TEST_MAPPING["Urinalysis"])
        self.neutral_lab_tests = [
            t
            for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Inflammation"]
            and t not in self.required_lab_tests["Liver"]
            and t not in self.required_lab_tests["Gallbladder"]
        ]

        self.answers["Treatment Requested"] = {
            "Cholecystectomy": False,
            "Antibiotics": False,
            "Support": False,
        }
        self.answers["Treatment Required"] = {
            "Cholecystectomy": False,
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
            # Preferred is regular US but MRI or endoscopic US is acceptable
            if modality == "Ultrasound" or modality == "HIDA":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            if modality == "MRI" or modality == "EUS":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def score_treatment(self) -> None:
        ### CHOLECYSTECTOMY ###
        if (
            procedure_checker(CHOLECYSTECTOMY_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(
                CHOLECYSTECTOMY_PROCEDURES_ICD10, self.procedures_icd10
            )
            or procedure_checker(
                CHOLECYSTECTOMY_PROCEDURES_KEYWORDS, self.procedures_discharge
            )
        ):
            self.answers["Treatment Required"]["Cholecystectomy"] = True

        if procedure_checker(
            CHOLECYSTECTOMY_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_CHOLECYSTECTOMY_KEYWORDS, self.answers["Treatment"]
        ):
            self.answers["Treatment Requested"]["Cholecystectomy"] = True

        ### SUPPORT ###
        if (
            keyword_positive(self.answers["Treatment"], "fluid")
            or keyword_positive(self.answers["Treatment"], "analgesi")
            or keyword_positive(self.answers["Treatment"], "pain")
        ):
            self.answers["Treatment Requested"]["Support"] = True

        ### ANTIBIOTICS ###
        # TODO: Check antibiotics against medications table
        if keyword_positive(self.answers["Treatment"], "antibiotic"):
            self.answers["Treatment Requested"]["Antibiotics"] = True

    def generate_feedback(self) -> str:
        """
        Cholecystitis specific feedback:
        - Start from generic feedback (no treatment)
        - Add cholecystitis specific imaging and treatment comments.
        """
        base_feedback = super().generate_feedback()
        messages: List[str] = []

        generic_msg = (
            "No major deficiencies were detected across examination, testing, "
            "imaging, diagnosis and treatment according to the current scoring "
            "rules."
        )

        if base_feedback and base_feedback.strip() and base_feedback.strip() != generic_msg:
            messages.append(base_feedback.strip())

        imaging_score = self.scores.get("Imaging", 0)
        correct_imaging = self.answers.get("Correct Imaging", []) or []

        has_ruq_primary = any(
            isinstance(item, dict)
            and item.get("region") == "Abdomen"
            and item.get("modality") in ("Ultrasound", "HIDA")
            for item in correct_imaging
        )
        has_mri_eus = any(
            isinstance(item, dict)
            and item.get("region") == "Abdomen"
            and item.get("modality") in ("MRI", "EUS")
            for item in correct_imaging
        )

        if imaging_score == 1:
            if has_mri_eus and not has_ruq_primary:
                messages.append(
                    "Imaging (cholecystitis): set region='Abdomen' and start with ultrasound or HIDA as first line. MRI/EUS are also recommended as follow up options. Aim to complete appropriate biliary imaging rather than relying on MRI/EUS alone."
                )
            else:
                messages.append(
                    "Imaging (cholecystitis): abdominal imaging was partially appropriate. Set region='Abdomen' and prioritize ultrasound or HIDA as first line. MRI/EUS are also recommended as follow up options. Consider completing the recommended imaging workup for biliary disease."
                )
        elif imaging_score == 0:
            messages.append(
                "Imaging (cholecystitis): no appropriate abdominal imaging was done. Set region='Abdomen' and request ultrasound or HIDA as first line. MRI/EUS are also recommended as follow up options. Consider completing the recommended imaging workup for biliary disease."
            )

        if not messages:
            if base_feedback and base_feedback.strip():
                return base_feedback.strip()
            return generic_msg

        return "\n\n".join(messages)