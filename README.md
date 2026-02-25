# DxEvolve

This repository contains the implementation code for **DxEvolve**.

## Directory structure and data access

Place the code and the MIMIC Clinical Decision-Making dataset in the same parent directory:

```text
.
├── DxEvolve
└── MIMIC-Clinical-Decision-Making-Dataset
```

Access to MIMIC-IV and MIMIC-CDM is governed by the applicable PhysioNet credentialing and data-use requirements. See the [MIMIC Clinical Decision-Making dataset repository](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset) for access instructions. Controlled clinical records are not redistributed here.

## Tested environment

The code was run with Python 3.11 on four servers running Ubuntu 22.04, each equipped with eight NVIDIA A100 GPUs (80 GB). The resources required for an individual evaluation depend on the selected backbone and serving configuration; this study hardware is not a universal minimum requirement.

## Installation

```bash
conda create -n dxevolve python=3.11
conda activate dxevolve
pip install -r requirements.txt
```

Environment creation and dependency installation typically require approximately 30-60 min on a networked Linux workstation with a compatible GPU driver. Actual time depends on network and package-mirror conditions and excludes system-level CUDA configuration, model weights and controlled-access data.

The following command checks that the main command-line entry point and installed dependencies can be loaded without starting an evaluation:

```bash
python run.py --help
```

## Synthetic demo

The repository includes one wholly synthetic biliary-presentation trajectory that follows the staged action-observation structure used by DxEvolve. It was authored solely for software demonstration and is not derived from MIMIC, the external hospital cohort or any other patient record. The offline demo validates the serialized action contract, sequential evidence-release boundary and automated final-diagnosis matching without loading an LLM, accessing clinical data or using a network service.

From the repository root, run:

```bash
python demo/run_demo.py
```

Expected output:

```text
DxEvolve synthetic demo
Case: synthetic-biliary-001
Actions: 4 (3 medical-evaluation, 1 contextual-search)
Evidence boundary: PASS
Final diagnosis: Acute calculous cholecystitis
Reference category: cholecystitis
Automated diagnosis match: PASS
```

The demo is CPU-only and completes in less than 1 s on the tested Ubuntu 22.04 environment. It verifies the serialized trajectory and final-diagnosis contract; it is not a clinical evaluation and does not reproduce a manuscript accuracy estimate.

## Local vLLM backend

Full evaluations use a locally deployed vLLM server. In `models/models.py`, set `base_url` to the local vLLM endpoint, for example `http://127.0.0.1:8002/v1/`. Deployment-specific placeholder values in that file should be adjusted locally.

Optional retrieval settings, when enabled, should be provided at run time through environment variables or command-line arguments rather than stored in the repository.

## Running an evaluation

A standard 400-encounter evaluation is launched with:

```bash
python run.py \
  --model "$MODEL_NAME" \
  --test_num 400
```

Available model identifiers and run-specific settings are defined by the released model configuration files. Dataset paths can be adjusted through the corresponding command-line options shown by `python run.py --help`.

## Outputs and analysis boundary

Each run creates a timestamped directory under `new_logs/` containing the execution log and serialized per-encounter results (`*_results.pkl`); runs with experience retrieval also retain the corresponding experience index.

The automated encounter-level scoring implementation is provided in `evaluate_cdm.py` and `evaluate_cdm_full.py`. Manuscript-level summaries were assembled from the exported evaluation records, and the numerical data underlying the reported figures and tables are provided as Source Data with the paper. The repository does not claim a single command that reconstructs every manuscript panel from raw logs.
