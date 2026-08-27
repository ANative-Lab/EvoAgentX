import os
import json
from typing import Any, Dict, Callable, List
from .benchmark import Benchmark
from .measures import exact_match_score, f1_score, acc_score
from ..core.logging import logger
from ..core.module_utils import load_json
from datasets import load_dataset

# WorfBench dataset file mapping
WORFBENCH_FILES_MAP = {
    "train": "worfbench_train.json",
    "test": "worfbench_test.json"
}
VALID_WORFBENCH_FILES = list(WORFBENCH_FILES_MAP.values())

def evaluate_workflow_sequence(prediction: List[Any], ground_truth: List[Any]) -> float:
    """Evaluate F1 score for sequence workflow."""
    from .measures import f1_score
    return f1_score(prediction=prediction, ground_truth=ground_truth)

def evaluate_workflow_graph(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Evaluate F1 score for graph workflow."""
    pred_nodes = set(prediction.get("nodes", []))
    true_nodes = set(ground_truth.get("nodes", []))
    pred_edges = set(tuple(edge) for edge in prediction.get("edges", []))
    true_edges = set(tuple(edge) for edge in ground_truth.get("edges", []))
    
    node_precision = len(pred_nodes & true_nodes) / len(pred_nodes) if pred_nodes else 0
    node_recall = len(pred_nodes & true_nodes) / len(true_nodes) if true_nodes else 0
    edge_precision = len(pred_edges & true_edges) / len(pred_edges) if pred_edges else 0
    edge_recall = len(pred_edges & true_edges) / len(true_edges) if true_edges else 0
    
    node_f1 = 2 * (node_precision * node_recall) / (node_precision + node_recall) if (node_precision + node_recall) > 0 else 0
    edge_f1 = 2 * (edge_precision * edge_recall) / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0
    
    return (node_f1 + edge_f1) / 2

def download_worfbench_data(dataset: str, save_folder: str) -> None:
    """
    Download WorfBench dataset from Hugging Face.

    Args:
        dataset (str): Dataset name ("worfbench").
        save_folder (str): Directory to save data.
    """
    datasets_map = {
        "train": {"repo_id": "zjunlp/WorFBench_train", "filename": "worfbench_train.json", "split": "train"},
        "test": {"repo_id": "zjunlp/WorFBench_test", "filename": "worfbench_test.json", "split": "test"}
    }
    
    os.makedirs(save_folder, exist_ok=True)
    for split, info in datasets_map.items():
        repo_id = info["repo_id"]
        filename = info["filename"]
        dataset_split = info["split"]
        save_path = os.path.join(save_folder, filename)
        
        if not os.path.exists(save_path):
            logger.info(f"Downloading {split} split of {dataset} from {repo_id}...")
            try:
                # Load dataset
                ds = load_dataset(repo_id, split=dataset_split)
                # Convert dataset to list and save as JSON
                data = [item for item in ds]
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Successfully downloaded and saved {filename} to {save_path}")
            except Exception as e:
                logger.error(f"Failed to download or save {filename}: {e}")
                raise
        else:
            logger.info(f"File {save_path} already exists, skipping download.")

class WorfBench(Benchmark):
    """
    WorfBench evaluation class for assessing LLM agents on complex workflow generation tasks.
    Assumed data structure:
    {
        "id": str,
        "task": str,
        "context": list of dicts (e.g., [{"title": str, "content": list of str}]),
        "expected_output": str or dict (sequence or graph),
        "type": str,
        "level": str
    }
    """
    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/worfbench")
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str) -> Dict:
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_worfbench_data(dataset="worfbench", save_folder=self.path)
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} still does not exist after download attempt!")
            return None
        logger.info(f"Loading WorfBench data from {file_path} ...")
        data = load_json(path=file_path, type="json")
        if data is None:
            logger.error(f"Failed to load data from {file_path}")
            return None
        # The raw train split (unlabelled chat/SFT data) has no "id" field,
        # unlike the test split. Synthesize a stable id from the file's type
        # and position for any example that doesn't already have one.
        typ = "train" if "train" in file_name else "test"
        for i, example in enumerate(data):
            if not example.get("id"):
                example["id"] = f"{typ}-{i+1}"
        return data

    def _load_data(self) -> None:
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=WORFBENCH_FILES_MAP["train"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=WORFBENCH_FILES_MAP["test"])

    def _get_label(self, example: Dict) -> Any:
        return example.get("expected_output", "")

    def _get_id(self, example: Dict) -> Any:
        return example["id"]

    def evaluate(self, prediction: Any, label: Any) -> Dict:
        if isinstance(prediction, list) and isinstance(label, list):
            f1 = evaluate_workflow_sequence(prediction, label)
        elif isinstance(prediction, dict) and isinstance(label, dict):
            f1 = evaluate_workflow_graph(prediction, label)
        else:
            f1 = f1_score(prediction=str(prediction), ground_truth=str(label))
        em = exact_match_score(prediction=prediction, ground_truth=label)
        acc = acc_score(prediction=prediction, ground_truths=[label])
        return {"em": em, "f1": f1, "acc": acc}

    async def async_evaluate(self, graph: Callable, example: Dict) -> float:
        task = example.get("task", "")
        context = "\n".join(
            f"{ctx.get('title', '')}: {' '.join(ctx.get('content', []))}"
            for ctx in example.get("context", [])
            if isinstance(ctx, dict)
        )
        inputs = f"Task: {task}\nContext: {context}\nGenerate workflow:\nAnswer:"
        try:
            generated_workflow = await graph(inputs)
        except Exception as e:
            logger.error(f"Error generating workflow: {e}")
            generated_workflow = ""
        label = self._get_label(example)
        metrics = self.evaluate(prediction=generated_workflow, label=label)
        return metrics["f1"]