#!/usr/bin/env python3
"""Run the dependency-free DxEvolve synthetic trajectory demonstration."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any


ALLOWED_ACTIONS = {
    "Physical Examination",
    "Laboratory Tests",
    "Imaging",
    "Experience Search",
    "Guideline Search",
    "PubMed Search",
}
MEDICAL_ACTIONS = {"Physical Examination", "Laboratory Tests", "Imaging"}
SEARCH_ACTIONS = {"Experience Search", "Guideline Search", "PubMed Search"}


def normalize_diagnosis(value: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return " ".join(value.casefold().translate(table).split())


def extract_final_diagnosis(model_output: str) -> str:
    match = re.search(r"^\s*Final\s+Diagnosis\s*:\s*(.+)$", model_output, re.I | re.M)
    if not match:
        raise ValueError("The model output does not contain a Final Diagnosis field.")
    diagnosis = match.group(1).strip()
    return re.split(r"\s*(?:;|\||\n)\s*", diagnosis, maxsplit=1)[0].strip()


def load_case(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("The demo case must be a JSON object.")
    return data


def validate_case(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata", {})
    if metadata.get("synthetic") is not True:
        raise ValueError("The demo accepts only data explicitly marked as synthetic.")
    if "not derived from a patient record" not in metadata.get("source_notice", "").casefold():
        raise ValueError("The synthetic-data provenance statement is missing.")

    trajectory = data.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("The demo trajectory is empty.")

    available_evidence = {"initial_presentation"}
    medical_actions = 0
    search_actions = 0
    for index, step in enumerate(trajectory, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Step {index} is not a JSON object.")
        action = step.get("action")
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Step {index} contains an unsupported action: {action!r}.")
        if not str(step.get("action_input", "")).strip():
            raise ValueError(f"Step {index} has no action input.")
        if not str(step.get("observation", "")).strip():
            raise ValueError(f"Step {index} has no observation.")

        evidence_used = set(step.get("uses_evidence_from", []))
        unavailable = evidence_used - available_evidence
        if unavailable:
            missing = ", ".join(sorted(unavailable))
            raise ValueError(f"Step {index} uses evidence not yet released: {missing}.")

        medical_actions += action in MEDICAL_ACTIONS
        search_actions += action in SEARCH_ACTIONS
        available_evidence.add(f"step_{index}")

    scoring = data.get("scoring", {})
    if scoring.get("reference_visibility") != "after_model_output":
        raise ValueError("The reference diagnosis must remain unavailable until scoring.")

    reference = normalize_diagnosis(str(scoring.get("reference_diagnosis", "")))
    visible_before_prediction = normalize_diagnosis(
        json.dumps(
            {
                "initial_presentation": data.get("initial_presentation", ""),
                "trajectory": trajectory,
            },
            ensure_ascii=False,
        )
    )
    if reference and reference in visible_before_prediction:
        raise ValueError("The reference diagnosis appears in the model-visible trajectory.")

    extracted = extract_final_diagnosis(str(data.get("model_output", "")))
    accepted = {
        normalize_diagnosis(str(value))
        for value in scoring.get("accepted_diagnoses", [])
    }
    diagnosis_match = normalize_diagnosis(extracted) in accepted

    return {
        "case_id": metadata.get("case_id", "unknown"),
        "steps": len(trajectory),
        "medical_actions": medical_actions,
        "search_actions": search_actions,
        "diagnosis": extracted,
        "reference": scoring.get("reference_diagnosis", ""),
        "diagnosis_match": diagnosis_match,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a synthetic staged trajectory and its final diagnosis."
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=Path(__file__).with_name("synthetic_biliary_case.json"),
        help="Path to a clearly synthetic JSON case.",
    )
    args = parser.parse_args()

    summary = validate_case(load_case(args.case))
    if not summary["diagnosis_match"]:
        raise ValueError("The extracted diagnosis does not match an accepted reference term.")

    print("DxEvolve synthetic demo")
    print(f"Case: {summary['case_id']}")
    print(
        "Actions: "
        f"{summary['steps']} "
        f"({summary['medical_actions']} medical-evaluation, "
        f"{summary['search_actions']} contextual-search)"
    )
    print("Evidence boundary: PASS")
    print(f"Final diagnosis: {summary['diagnosis']}")
    print(f"Reference category: {summary['reference']}")
    print("Automated diagnosis match: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
