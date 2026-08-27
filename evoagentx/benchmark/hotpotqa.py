import os 
from typing import Any, Callable
from .benchmark import Benchmark
from .scorers import QAScorer
from ..core.logging import logger
from ..core.module_utils import load_json
from ..utils.utils import download_file
from ..utils.aflow_utils.data_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data


HOTPOTQA_FILES_MAP = {"train": "hotpot_train_v1.1.json", "dev": "hotpot_dev_distractor_v1.json", "test": None}
VALIDE_RAW_HOTPOTQA_FILES = [file for file in list(HOTPOTQA_FILES_MAP.values()) if file is not None]

def download_raw_hotpotqa_data(name: str, save_folder: str):

    assert name in VALIDE_RAW_HOTPOTQA_FILES, f"'{name}' is an invalid hotpotqa file name. Available file names: {VALIDE_RAW_HOTPOTQA_FILES}"
    # The original curtis.ml.cmu.edu host is no longer reliably reachable.
    # This public mirror serves byte-identical copies of the official files.
    url = f"https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/{name}?download=true"
    typ = "train" if "train" in name else "dev"
    logger.info(f"Downloading HotPotQA {typ} data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, name))


class HotPotQA(Benchmark):

    """Benchmark class for evaluating multi-hop question answering on HotPotQA dataset.
    
    Each HotPotQA example has the following structure:
    {
        "_id": str, 
        "question": str, 
        "answer": str, 
        "context": [["context_title", ["context_sentence", "another_sentence"]]],
        "supporting_facts": [["supporting_title", supporting_sentence_index]],
        "type": str,
        "level": str
    }
    
    The benchmark evaluates answers using exact match, F1 score, and accuracy metrics.
    """

    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/hotpotqa")
        self.scorer = QAScorer(metrics=["f1", "em", "acc"])
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_raw_hotpotqa_data(name=file_name, save_folder=self.path)
        logger.info(f"loading HotPotQA data from {file_path} ...")
        return load_json(path=file_path, type="json")

    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP["train"])
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP["dev"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP["test"])
    
    def _get_label(self, example: Any) -> Any:
        return example["answer"]
    
    def _get_id(self, example: Any) -> Any:
        # Mostly 24-hex strings, but the AFlow export carries a handful of int
        # _id values, so normalise to str.
        return str(example["_id"])
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        return self.scorer.score(prediction=prediction, label=label)
    

class AFlowHotPotQA(HotPotQA):

    """
    AFlow-specific implementation of HotPotQA benchmark.
    """

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset="hotpotqa", save_folder=self.path)
        logger.info(f"loading data from {file_path} ...")
        return load_json(path=file_path, type="jsonl")

    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["hotpotqa"]["train"])
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["hotpotqa"]["dev"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["hotpotqa"]["test"])
    
    async def async_evaluate(self, graph: Callable, example: Any) -> float:

        # generate solution 
        prompt = example["question"]
        paragraphs = [item[1] for item in example["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        inputs = f"Context: {context_str}\n\nQuestion: {prompt}\n\nAnswer:"
        solution = await graph(inputs)
        label = self._get_label(example)
        metrics = await super().async_evaluate(prediction=solution, label=label)
        return metrics["f1"]
    
