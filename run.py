import os
from os.path import join
import re
import random
from datetime import datetime
import time
from typing import List, Dict, Any, Union, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import pickle
import traceback
from tools.retrieve import ExperienceStore

import numpy as np
import torch
from loguru import logger
import langchain
import argparse
import yaml

from dataset.utils import load_hadm_from_file
from utils.logging import append_to_pickle_file
from evaluate_cdm import get_feedback
from models.models import CustomLLM
from agents.agent import build_agent_executor_ZeroShot

MODEL_CFG_DIR = "./configs/model"
_RUN_TS_RE = re.compile(r"\d{2}-\d{2}-\d{4}_\d{2}:\d{2}:\d{2}$")


def refresh_run_name_timestamp(run_name: str) -> str:
    now_ts = datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y_%H:%M:%S")
    if _RUN_TS_RE.search(run_name):
        return _RUN_TS_RE.sub(now_ts, run_name)
    return f"{run_name}_{now_ts}"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def resolve_pubmed_api_key(explicit_key: Optional[str]) -> Optional[str]:
    if explicit_key and str(explicit_key).strip():
        return str(explicit_key).strip()
    for env_name in ("NCBI_API_KEY", "PUBMED_API_KEY"):
        env_value = os.getenv(env_name)
        if env_value and env_value.strip():
            return env_value.strip()
    return None


def _list_model_cfg_options(cfg_dir: str) -> list[str]:
    if not os.path.isdir(cfg_dir):
        return []
    out = []
    for fn in os.listdir(cfg_dir):
        if fn.endswith(".yaml") or fn.endswith(".yml"):
            out.append(os.path.splitext(fn)[0])
    return sorted(out)


def load_model_cfg(model_key: str, cfg_dir: str = MODEL_CFG_DIR) -> dict:
    path = os.path.join(cfg_dir, f"{model_key}.yaml")
    if not os.path.isfile(path):
        opts = _list_model_cfg_options(cfg_dir)
        raise FileNotFoundError(
            f"Model config not found: {path}\n"
            f"Available options in '{cfg_dir}':\n  - " + "\n  - ".join(opts)
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def apply_model_cfg_to_args(args, model_cfg: dict):
    """
    Keep Hydra-like behavior:
    - Always load configs/model/<model_key>.yaml
    - If CLI explicitly provides an override (arg is not None), keep CLI value
    """
    # core model fields
    if getattr(args, "model_name", None) is None:
        args.model_name = model_cfg.get("model_name", args.model_name)

    if getattr(args, "max_context_length", None) is None:
        args.max_context_length = model_cfg.get("max_context_length", args.max_context_length)

    if getattr(args, "exllama", None) is None:
        args.exllama = model_cfg.get("exllama", args.exllama)

    if getattr(args, "base_models", None) is None:
        args.base_models = model_cfg.get("base_models", args.base_models)

    # tags and stop words
    for k in (
        "system_tag_start",
        "system_tag_end",
        "user_tag_start",
        "user_tag_end",
        "ai_tag_start",
        "ai_tag_end",
    ):
        if getattr(args, k, None) in (None, "") and k in model_cfg:
            setattr(args, k, model_cfg.get(k, getattr(args, k)))

    # stop_words: if CLI didn't pass any, use config
    if (not getattr(args, "stop_words", None)) and ("stop_words" in model_cfg):
        args.stop_words = list(model_cfg.get("stop_words") or [])

    return args


def parse_int_range(spec: str):
    # supports "a:b" with open ends
    if spec is None:
        return (None, None)
    spec = spec.strip()
    if ":" not in spec:
        raise ValueError(f"Invalid --exp_seq_range '{spec}', expected a:b")
    a, b = spec.split(":", 1)
    start = int(a) if a.strip() else None
    end = int(b) if b.strip() else None
    if start is not None and end is not None and start > end:
        raise ValueError(f"Invalid --exp_seq_range '{spec}', start > end")
    return start, end


def args_to_flag_lines(args: Any) -> str:
    """
    Return a string with one arg per line:
    --name value\n
    Skips None. Booleans are true/false.
    Lists/tuples become space-joined.
    """
    d = vars(args) if hasattr(args, "__dict__") else dict(args)

    lines = []
    for name in sorted(d.keys()):
        val = d[name]
        if val is None:
            continue

        flag = f"--{name}"
        if isinstance(val, bool):
            lines.append(f"{flag} {str(val).lower()}")
        elif isinstance(val, (list, tuple)):
            lines.append(f"{flag} " + " ".join(map(str, val)))
        else:
            lines.append(f"{flag} {val}")

    return " \\\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("run_cdm")

    # ===== model config (Hydra-like) =====
    p.add_argument(
        "--model",
        type=str,
        default="Qwen3-235B-A22B-Instruct-2507",
        help=f"Model config key under {MODEL_CFG_DIR} (without .yaml).",
    )

    # Optional overrides. If not provided, they come from configs/model/<model>.yaml
    p.add_argument("--model_name", type=str, default=None, help="Override HF model_name from model yaml.")
    p.add_argument("--max_context_length", type=int, default=None, help="Override max_context_length from model yaml.")
    p.add_argument("--exllama", type=str2bool, default=None, help="Override exllama from model yaml.")

    # ===== defaults you provided =====
    p.add_argument("--embedding_model_name", type=str, default="BAAI/bge-large-en-v1.5")
    p.add_argument(
        "--pathologies",
        nargs="+",
        default=["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"],
    )
    p.add_argument("--agent", type=str, default="ZeroShot")
    p.add_argument("--prompt_template", type=str, default="VANILLA")

    p.add_argument("--summarize", type=str2bool, default=True)
    p.add_argument("--second_thought", type=str2bool, default=False)
    p.add_argument("--fewshot", type=str2bool, default=False)
    p.add_argument("--include_ref_range", type=str2bool, default=False)
    p.add_argument("--bin_lab_results", type=str2bool, default=False)
    p.add_argument("--bin_lab_results_abnormal", type=str2bool, default=False)
    p.add_argument("--provide_diagnostic_criteria", type=str2bool, default=False)
    p.add_argument("--include_tool_use_examples", type=str2bool, default=False)
    p.add_argument("--abbreviated", type=str2bool, default=True)
    p.add_argument("--self_consistency", type=str2bool, default=False)
    p.add_argument("--only_abnormal_labs", type=str2bool, default=False)

    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--data_seed", type=int, default=2023)
    p.add_argument("--local_logging", type=str2bool, default=True)
    p.add_argument("--run_descr", type=str, default="")

    p.add_argument("--first_patient", type=str, default=None)
    p.add_argument("--patient_list_path", type=str, default=None)

    p.add_argument("--order", type=str, default="pli")
    p.add_argument("--diagnostic_criteria", type=str, default="")
    p.add_argument("--rr_name", type=str, default="RR")
    p.add_argument("--diag_crit_writer_openai_api_key", type=str, default=None)
    p.add_argument("--confirm_diagnosis", type=str2bool, default=False)
    p.add_argument("--save_probabilities", type=str2bool, default=False)

    p.add_argument(
        "--pubmed_api_key",
        type=str,
        default=os.getenv("NCBI_API_KEY"),
        help="PubMed/NCBI API key. Defaults to NCBI_API_KEY or PUBMED_API_KEY when unset.",
    )
    p.add_argument("--use_pubmed", type=str2bool, default=True)
    p.add_argument("--use_exp", type=str2bool, default=True)
    p.add_argument("--use_guidelines", type=str2bool, default=True)
    p.add_argument("--use_full_info", type=str2bool, default=True)

    p.add_argument("--test_num", type=int, default=-100)
    p.add_argument("--last_patho", type=str, default=None)

    # ===== paths =====
    p.add_argument("--base_mimic", type=str, default="hosp/")
    p.add_argument("--base_mimic_eval", type=str, default="../MIMIC-Clinical-Decision-Making-Dataset")
    p.add_argument("--lab_test_mapping_path", type=str, default="../MIMIC-Clinical-Decision-Making-Dataset/lab_test_mapping.pkl")

    # ===== model loading helper =====
    p.add_argument("--base_models", type=str, default=None)
    p.add_argument("--local_logging_dir", type=str, default="./new_logs")

    # stop words override (if you pass any, we keep them and ignore yaml stop_words)
    p.add_argument("--stop_words", nargs="*", default=None, help="Override stop words (space separated).")

    # Chat tags overrides (if empty, we will fill from model yaml)
    p.add_argument("--system_tag_start", type=str, default="")
    p.add_argument("--system_tag_end", type=str, default="")
    p.add_argument("--user_tag_start", type=str, default="")
    p.add_argument("--user_tag_end", type=str, default="")
    p.add_argument("--ai_tag_start", type=str, default="")
    p.add_argument("--ai_tag_end", type=str, default="")

    p.add_argument("--exp_seq_range", type=str, default=None,
                help="Keep only experiences with exp_seq in [a:b] when loading exp store in resume mode. Example: 0:100, :200, 50:")

    p.add_argument("--test_lang", type=str, default=None,
                help="Select en/zh")
    p1 = p.add_argument("--gg", type=str2bool, default=False)

    return p


def load_processed_ids(results_pkl_path: str) -> set:
    if not os.path.isfile(results_pkl_path):
        return set()
    done = set()
    with open(results_pkl_path, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            except Exception:
                break
            if isinstance(obj, dict):
                done.update(obj.keys())
    return done


def run(args):
    if not args.self_consistency:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    hadm_info_clean = {}
    hadm_info_clean_eval = {}

    for pathology in args.pathologies:
        dic_temp = load_hadm_from_file(
            f"{pathology}_hadm_info_first_diag",
            base_mimic=args.base_mimic,
        )
        for _id in dic_temp.keys():
            dic_temp[_id]["pathology"] = pathology
        hadm_info_clean = hadm_info_clean | dic_temp

        dic_temp = load_hadm_from_file(
            f"{pathology}_hadm_info_clean",
            base_mimic=args.base_mimic_eval,
        )
        hadm_info_clean_eval = hadm_info_clean_eval | dic_temp

    items = list(hadm_info_clean.items())
    random.seed(args.data_seed)
    random.shuffle(items)

    if args.last_patho and (args.last_patho in args.pathologies):
        items = sorted(items, key=lambda item: item[1]["pathology"] == args.last_patho)
        non_last_patho_count = sum(1 for _, info in items if info["pathology"] != args.last_patho)
    elif args.test_num == -1:
        dr_idx = []
        with open("./tests/dr_idx.txt") as f:
            for line in f.readlines():
                dr_idx.append(int(line.strip()))
        dr_idx_set = set(dr_idx)
        items = sorted(items, key=lambda item: (item[0] in dr_idx_set))

        non_last_patho_count = sum(1 for idx, info in items if idx not in dr_idx_set)
    else:
        non_last_patho_count = len(items)
        if args.test_num:
            non_last_patho_count -= args.test_num

    hadm_info_clean = dict(items)

    tags = {
        "system_tag_start": args.system_tag_start,
        "user_tag_start": args.user_tag_start,
        "ai_tag_start": args.ai_tag_start,
        "system_tag_end": args.system_tag_end,
        "user_tag_end": args.user_tag_end,
        "ai_tag_end": args.ai_tag_end,
    }

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )

    llm = CustomLLM(
        model_name=args.model_name,
        tags=tags,
        max_context_length=args.max_context_length,
        exllama=args.exllama,
        use_vllm=True,
        seed=args.seed,
        self_consistency=args.self_consistency,
        model=None,
        generator=None,
        tokenizer=None,
    )
    llm.load_model(args.base_models)

    date_time = datetime.fromtimestamp(time.time())
    str_date = date_time.strftime("%d-%m-%Y_%H:%M:%S")
    safe_model_name = str(args.model).replace("/", "_")

    done_ids = set()
    str_date = datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y_%H:%M:%S")
    run_name = f"full_{args.agent}_{safe_model_name}_{str_date}"

    if args.last_patho:
        run_name = f"LASTPATHO={args.last_patho}_" + run_name
    elif args.test_num:
        run_name = f"TESTNUM={args.test_num}_" + run_name

    if args.use_pubmed:
        run_name = "PUBMED_" + run_name
    if args.use_guidelines:
        run_name = "GUIDE_" + run_name
    if args.use_exp:
        run_name = "EXP_" + run_name

    run_dir = join(args.local_logging_dir, run_name)
    experience_index_path = os.path.join(run_dir, "experience_faiss")

    os.makedirs(run_dir, exist_ok=True)
    if args.use_exp:
        os.makedirs(experience_index_path, exist_ok=True)

    results_log_path = join(run_dir, f"{run_name}_results.pkl")
    log_path = join(run_dir, f"{run_name}.log")
    exp_store = None
    logger.add(log_path, enqueue=True, backtrace=True, diagnose=True)
    run_data = hadm_info_clean

    first_patient_seen = False
    for idx, _id in tqdm(enumerate(list(run_data.keys())), total=len(list(run_data.items()))):
        if args.first_patient and not first_patient_seen:
            if _id == args.first_patient:
                first_patient_seen = True
            else:
                continue

        logger.info(f"Processing patient: {_id}")

        agent_executor = build_agent_executor_ZeroShot(
            patient=hadm_info_clean[_id],
            llm=llm,
            lab_test_mapping_path=args.lab_test_mapping_path,
            logfile=log_path,
            max_context_length=args.max_context_length,
            tags=tags,
            include_ref_range=args.include_ref_range,
            bin_lab_results=args.bin_lab_results,
            include_tool_use_examples=args.include_tool_use_examples,
            provide_diagnostic_criteria=args.provide_diagnostic_criteria,
            experience_index_path=experience_index_path,
            summarize=args.summarize,
            model_stop_words=args.stop_words,
            pubmed_api_key=args.pubmed_api_key,
            embeddings=embeddings,
            use_guidelines=args.use_guidelines,
            use_pubmed=args.use_pubmed,
            use_experience_search=args.use_exp,
            exp_store=exp_store,
            init_store=None,
            summarize_observation=args.second_thought,
            encourage_guideline=args.gg,
            use_full_info=args.use_full_info,
        )

        result = agent_executor({"input": hadm_info_clean[_id]["Patient History"].strip()})
        result["idx"] = idx
        feedback = get_feedback(
            source=hadm_info_clean_eval[_id],
            result=result,
            patho=hadm_info_clean[_id]["pathology"],
            correctness_only=(idx>=non_last_patho_count),
        )

        feedback.update(result)
        feedback.update(
            {
                "ground_truth": hadm_info_clean[_id]["pathology"],
                "correct": bool(feedback["scores"]['Gracious Diagnosis']),
                "message": feedback.get("message", ""),
                "clinician": feedback.get("clinician", ""),
            }
        )
        result.update(feedback)
        patho_name = hadm_info_clean[_id]["pathology"]

        if idx < non_last_patho_count and args.use_exp:
                exp_store = agent_executor.agent.construct_experience(feedback)
        if exp_store is not None and args.use_exp:
            result["experience_retrieval_records"] = exp_store.get_records()
        append_to_pickle_file(results_log_path, {_id: result})
        done_ids.add(_id)

    if exp_store:
        exp_store.save()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    model_cfg = load_model_cfg(args.model, MODEL_CFG_DIR)
    args = apply_model_cfg_to_args(args, model_cfg)

    if args.stop_words is None:
        args.stop_words = []

    args.pubmed_api_key = resolve_pubmed_api_key(args.pubmed_api_key)
    if args.use_pubmed and not args.pubmed_api_key:
        logger.warning(
            "PubMed search is enabled but no API key was supplied. "
            "Set NCBI_API_KEY or pass --pubmed_api_key to use authenticated requests."
        )

    run(args)
