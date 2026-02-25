from evaluators.pathology_evaluator import PathologyEvaluator, best_effort_cover_fast
from utils.nlp import (
    diagnosis_checker,
    procedure_checker,
    keyword_positive,
    treatment_alternative_procedure_checker,
)
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from icd.procedure_mappings import (
    ERCP_PROCEDURES_ICD10,
    ERCP_PROCEDURES_ICD9,
    ERCP_PROCEDURES_KEYWORDS,
    DRAINAGE_PROCEDURES_ICD9,
    DRAINAGE_PROCEDURES_ALL_ICD10,
    DRAINAGE_PROCEDURES_PANCREATITIS_ICD10,
    DRAINAGE_PROCEDURES_KEYWORDS,
    DRAINAGE_LOCATIONS_PANCREATITIS,
    ALTERNATE_DRAINAGE_KEYWORDS_PANCREATITIS,
    CHOLECYSTECTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_CHOLECYSTECTOMY_KEYWORDS,
)
from typing import List


class PancreatitisEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of pancreatitis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "pancreatitis"
        self.alternative_pathology_names = [
            {
                "location": "pancrea",
                "modifiers": [
                    "gangren",
                    "infect",
                    "inflam",
                    "abscess",
                    "necros",
                ],
            }
        ]
        self.gracious_alternative_pathology_names = []

        self.required_lab_tests = {
            "Inflammation": INFLAMMATION_LAB_TESTS,
            "Pancreas": [
                50867,  # Amylase
                50956,  # Lipase
            ],
            "Seriousness": [
                51480,  # "Hematocrit",
                50810,
                51221,
                51638,
                51006,  # "Urea Nitrogen",
                52647,
                51000,  # "Triglycerides",
                50893,  # "Calcium, Total",
                50824,  # "Sodium",
                52623,
                50983,
                52610,  # "Potassium",
                50971,
                50822,
            ],
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
            and t not in self.required_lab_tests["Pancreas"]
            and t not in self.required_lab_tests["Seriousness"]
        ]

        self.answers["Treatment Requested"] = {
            "Support": False,
            "Drainage": False,
            "ERCP": False,
            "Cholecystectomy": False,
        }
        self.answers["Treatment Required"] = {
            "Support": True,
            "Drainage": False,
            "ERCP": False,
            "Cholecystectomy": False,
        }

    def score_imaging(
        self,
        region: str,
        modality: str,
    ) -> None:
        # Region must be abdomen
        if region == "Abdomen":
            # Preferred imaging is US
            if modality == "Ultrasound":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # CT is acceptable but should be done after US
            if modality == "CT":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
            # EUS is neutral if not biliary etiology otherwise +1
            if modality == "EUS":
                if diagnosis_checker(
                    self.discharge_diagnosis, self.icd_diagnoses, "biliary"
                ):
                    self.scores["Imaging"] += 1
                return True
        return False

    def score_treatment(self) -> None:
        ### SUPPORT ###
        if (
            keyword_positive(self.answers["Treatment"], "fluid")
            and (
                keyword_positive(self.answers["Treatment"], "analgesi")
                or keyword_positive(self.answers["Treatment"], "pain")
            )
            # and keyword_positive(self.answers["Treatment"], "nutrition")
            and keyword_positive(self.answers["Treatment"], "monitor")
        ):
            self.answers["Treatment Requested"]["Support"] = True

        ### DRAINAGE ###
        if (
            procedure_checker(DRAINAGE_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(DRAINAGE_PROCEDURES_ALL_ICD10, self.procedures_icd10)
            or procedure_checker(
                DRAINAGE_PROCEDURES_PANCREATITIS_ICD10, self.procedures_icd10
            )
            or (
                procedure_checker(
                    DRAINAGE_PROCEDURES_KEYWORDS, self.procedures_discharge
                )
                and procedure_checker(
                    DRAINAGE_LOCATIONS_PANCREATITIS, self.procedures_discharge
                )
            )
        ):
            self.answers["Treatment Required"]["Drainage"] = True

        if treatment_alternative_procedure_checker(
            ALTERNATE_DRAINAGE_KEYWORDS_PANCREATITIS, self.answers["Treatment"]
        ):
            self.answers["Treatment Requested"]["Drainage"] = True

        ### BILIARY ###
        if diagnosis_checker(self.discharge_diagnosis, self.icd_diagnoses, "biliary"):
            ### CHOLECYSTECTOMY ###
            # TODO: Cholecystectomy is recommended in FUTURE (once stable/cured) if biliary origin to remove future risk of gallstone pancreatitis. Again, manual check done here as no temporal check is stable enough
            self.answers["Treatment Required"]["Cholecystectomy"] = True
            if procedure_checker(
                CHOLECYSTECTOMY_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]
            ) or treatment_alternative_procedure_checker(
                ALTERNATE_CHOLECYSTECTOMY_KEYWORDS, self.answers["Treatment"]
            ):
                self.answers["Treatment Requested"]["Cholecystectomy"] = True

        ### ERCP ###
        if (
            procedure_checker(ERCP_PROCEDURES_ICD9, self.procedures_icd9)
            or procedure_checker(ERCP_PROCEDURES_ICD10, self.procedures_icd10)
            or procedure_checker(ERCP_PROCEDURES_KEYWORDS, self.procedures_discharge)
        ):
            self.answers["Treatment Required"]["ERCP"] = True

        if procedure_checker(ERCP_PROCEDURES_KEYWORDS, [self.answers["Treatment"]]):
            self.answers["Treatment Requested"]["ERCP"] = True

    def generate_feedback(self) -> str:
        """
        Pancreatitis specific feedback:
        - Start from generic feedback (no treatment).
        - Add pancreatitis specific imaging and treatment comments.
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
        imgs = self.answers.get("Correct Imaging", []) or []

        has_abd_us = any(
            isinstance(d, dict)
            and d.get("region") == "Abdomen"
            and d.get("modality") == "Ultrasound"
            for d in imgs
        )
        has_abd_ct = any(
            isinstance(d, dict)
            and d.get("region") == "Abdomen"
            and d.get("modality") == "CT"
            for d in imgs
        )
        has_abd_eus = any(
            isinstance(d, dict)
            and d.get("region") == "Abdomen"
            and d.get("modality") == "EUS"
            for d in imgs
        )

        is_biliary = diagnosis_checker(
            self.discharge_diagnosis, self.icd_diagnoses, "biliary"
        )

        if imaging_score >= 1:
            if has_abd_ct and not has_abd_us:
                messages.append(
                    "Imaging: set region='Abdomen' and request ultrasound first. CT is also recommended but should be considered as a second line option."
                )
            if is_biliary and has_abd_eus and not has_abd_us:
                messages.append(
                    "Imaging: for biliary pancreatitis, start with region='Abdomen' ultrasound. EUS is also recommended but should be used as a follow up option."
                )
        elif imaging_score == 0:
            messages.append(
                "Imaging: no useful abdominal imaging. Next time set region='Abdomen' and use ultrasound as first choice. CT/EUS are also appropriate options but should be considered after ultrasound."
            )

        if not messages:
            if base_feedback and base_feedback.strip():
                return base_feedback.strip()
            return generic_msg

        return "\n\n".join(messages)
