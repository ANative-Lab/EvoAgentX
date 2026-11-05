# Acknowledgement: 
# This file is modified from the original AFlow repository: https://github.com/geekan/MetaGPT/blob/main/metagpt/ext/aflow/scripts/optimizer_utils/data_utils.py

import os 
import json 
import random
import tarfile
import requests
import datetime
import numpy as np 
import pandas as pd 
from ..utils import make_parent_folder 
from ...core.logging import logger
from ...core.module_utils import load_json, save_json 


AFLOW_DATASET_FILES_MAP = {
    "hotpotqa": {"train": None, "dev": "hotpotqa_validate.jsonl", "test": "hotpotqa_test.jsonl"},
    "humaneval": {"train": None, "dev": "humaneval_validate.jsonl", "test": "humaneval_test.jsonl", "test_cases": "humaneval_public_test.jsonl"},
    "mbpp": {"train": None, "dev": "mbpp_validate.jsonl", "test": "mbpp_test.jsonl", "test_cases": "mbpp_public_test.jsonl"},
    "gsm8k": {"train": None, "dev": "gsm8k_validate.jsonl", "test": "gsm8k_test.jsonl"},
    "math": {"train": None, "dev": "math_validate.jsonl", "test": "math_test.jsonl"},
}

def extract_tar_gz(filename: str, extract_path: str) -> None:
    """Extract a tar.gz file to the specified path."""
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=extract_path)


def download_aflow_benchmark_data(dataset: str, save_folder: str):

    candidate_datasets = list(AFLOW_DATASET_FILES_MAP.keys()) + ["all"]
    lower_candidate_datasets = [dataset.lower() for dataset in candidate_datasets]
    if dataset.lower() not in lower_candidate_datasets:
        raise ValueError(f"Invalid value for dataset: {dataset}. Available choices: {candidate_datasets}")
    
    url = "https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e"
    logger.info(f"Downloading AFlow benchmark data from {url} ...")
    aflow_data_save_file = os.path.join(save_folder, "aflow_data.tar.gz")
    # download_file(url=url, save_file=aflow_data_save_file)
    make_parent_folder(aflow_data_save_file)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(aflow_data_save_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    logger.info(f"Extracting data for {dataset} dataset(s) from {aflow_data_save_file} ...")
    extract_tar_gz(aflow_data_save_file, extract_path=save_folder)

    if dataset != "all":
        dataset_files = [file for file in list(AFLOW_DATASET_FILES_MAP[dataset].values()) if file is not None]
        for file in os.listdir(save_folder):
            if file not in dataset_files:
                os.remove(os.path.join(save_folder, file))
    
    if os.path.exists(aflow_data_save_file):
        logger.info(f"Remove {aflow_data_save_file}")
        os.remove(aflow_data_save_file)


class DataUtils:

    def __init__(self, root_path: str):
        self.root_path = root_path
        self.top_scores = []

    def load_results(self, path: str) -> list:
        result_path = os.path.join(path, "results.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as json_file:
                try:
                    return json.load(json_file)
                except json.JSONDecodeError:
                    return []
        return []

    def get_top_rounds(self, sample: int, path=None, mode="Graph"):

        self._load_scores(path, mode)
        unique_rounds = set()
        unique_top_scores = []

        first_round = next((item for item in self.top_scores if item["round"] == 0), None)
        if first_round:
            unique_top_scores.append(first_round)
            unique_rounds.add(0)

        for item in self.top_scores:
            if item["round"] not in unique_rounds:
                unique_top_scores.append(item)
                unique_rounds.add(item["round"])

                if len(unique_top_scores) >= sample:
                    break

        return unique_top_scores

    def select_round(self, items):

        if not items:
            raise ValueError("Item list is empty.")

        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
        scores = [item["score"] * 100 for item in sorted_items]

        probabilities = self._compute_probabilities(scores)
        logger.info(f"\nMixed probability distribution: {probabilities}")
        logger.info(f"\nSorted rounds: {sorted_items}")

        selected_index = np.random.choice(len(sorted_items), p=probabilities)
        logger.info(f"\nSelected index: {selected_index}, Selected item: {sorted_items[selected_index]}")

        return sorted_items[selected_index]

    def _compute_probabilities(self, scores, alpha=0.2, lambda_=0.3):

        scores = np.array(scores, dtype=np.float64)
        n = len(scores)

        if n == 0:
            raise ValueError("Score list is empty.")

        uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)

        max_score = np.max(scores)
        shifted_scores = scores - max_score
        exp_weights = np.exp(alpha * shifted_scores)

        sum_exp_weights = np.sum(exp_weights)
        if sum_exp_weights == 0:
            raise ValueError("Sum of exponential weights is 0, cannot normalize.")

        score_prob = exp_weights / sum_exp_weights

        mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob

        total_prob = np.sum(mixed_prob)
        if not np.isclose(total_prob, 1.0):
            mixed_prob = mixed_prob / total_prob

        return mixed_prob

    def load_log(self, cur_round, path=None, mode: str = "Graph"):
        if mode == "Graph":
            log_dir = os.path.join(self.root_path, f"round_{cur_round}", "log.json")
        else:
            log_dir = path

        if not os.path.exists(log_dir):
            return "" 
        logger.info(log_dir)
        # data = read_json_file(log_dir, encoding="utf-8")
        data = load_json(log_dir, type="json")

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data)

        if not data:
            return ""

        sample_size = min(3, len(data))
        random_samples = random.sample(data, sample_size)

        log = ""
        for sample in random_samples:
            log += json.dumps(sample, indent=4, ensure_ascii=False) + "\n\n"

        return log

    def get_results_file_path(self, graph_path: str) -> str:
        return os.path.join(graph_path, "results.json")

    def create_result_data(self, round: int, score: float, avg_cost: float, total_cost: float) -> dict:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {"round": round, "score": score, "avg_cost": avg_cost, "total_cost": total_cost, "time": now}

    def save_results(self, json_file_path: str, data: list):
        save_json(data, json_file_path, type="json", use_indent=True)

    def _load_scores(self, path=None, mode="Graph"):
        if mode == "Graph":
            rounds_dir = self.root_path # os.path.join(self.root_path, "workflows")
        else:
            rounds_dir = path

        result_file = os.path.join(rounds_dir, "results.json")
        self.top_scores = []

        data = load_json(result_file, type="json")
        data = [row for row in data if row.get("split", "validation") == "validation"]
        if not data:
            return self.top_scores
        df = pd.DataFrame(data)

        scores_per_round = df.groupby("round")["score"].mean().to_dict()

        for round_number, average_score in scores_per_round.items():
            self.top_scores.append({"round": round_number, "score": average_score})

        self.top_scores.sort(key=lambda x: x["score"], reverse=True)

        return self.top_scores
    

def test_case_2_test_function(solution: str, test_case: str, entry_point: str):
    tester_function = f"""
{solution}


def check(candidate):
    {test_case}

def test_check():
    check({entry_point})

test_check()
"""
    return tester_function
