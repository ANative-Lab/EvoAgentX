import os
from typing import Any, List, Callable
from ..core.logging import logger
from .benchmark import Benchmark
from .scorers import MathScorer
from ..utils.utils import download_file
from ..core.module_utils import load_json
from ..utils.aflow_utils.data_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data


GSM8K_FILES_MAP = {"train": "train.jsonl", "dev": None, "test": "test.jsonl"}
VALID_RAW_GSM8K_FILES = [file for file in list(GSM8K_FILES_MAP.values()) if file is not None]

def download_raw_gsm8k_data(name: str, save_folder: str):

    assert name in VALID_RAW_GSM8K_FILES, f"'{name}' is an invalid GSM8K file name. Available file names: {VALID_RAW_GSM8K_FILES}"
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{name}"
    typ = "train" if "train" in name else "test"
    logger.info(f"Downloading GSM8K {typ} data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, name))


def load_gsm8k_data(file_path: str) -> List[dict]:

    base_name = os.path.basename(file_path)
    file_type_map = {file_name: typ for typ, file_name in GSM8K_FILES_MAP.items()}
    assert base_name in file_type_map, f"'{base_name}' is an invalid gsm8k file name. Available file names: {VALID_RAW_GSM8K_FILES}"

    typ = file_type_map[base_name]
    data = load_json(path=file_path, type="jsonl")
    new_data = [] 
    for i, example in enumerate(data):
        item = {"id": f"{typ}-{i+1}"}
        item.update(example)
        new_data.append(item)
    return new_data
    

class GSM8K(Benchmark):

    """Benchmark class for evaluating math reasoning on GSM8K dataset.
    
    GSM8K (Grade School Math 8K) is a dataset of math word problems that
    test a model's ability to solve grade school level math problems requiring
    multi-step reasoning. This class handles loading the dataset, evaluating
    solutions, and computing metrics based on answer accuracy.
    
    Each GSM8K example has the following structure:
    {
        "id": "test-1", 
        "question": "the question", 
        "answer": "the answer"
    }
    
    The benchmark evaluates answers by extracting the final numerical value
    and comparing it to the ground truth answer.
    """
    
    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/gsm8k")
        self.scorer = MathScorer(mode="numeric", tol=1e-6)
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_raw_gsm8k_data(name=file_name, save_folder=self.path)
        # loading data from file 
        logger.info(f"loading GSM8K data from {file_path} ...")
        return load_gsm8k_data(file_path=file_path)
    
    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=GSM8K_FILES_MAP["train"])
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=GSM8K_FILES_MAP["dev"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=GSM8K_FILES_MAP["test"])
            
    def _get_label(self, example: Any) -> Any:
        return example["answer"]
    
    def _get_id(self, example: Any) -> Any:
        return example["id"]
    
    def extract_last_number(self, text: str) -> float:
        """
        Extract the last number from a text.
        """
        return self.scorer._extract_numeric(text)
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        return self.scorer.score(prediction=prediction, label=label)


class AFlowGSM8K(GSM8K): 

    """AFlow-specific implementation of GSM8K benchmark.
    
    This class extends the GSM8K benchmark with features specific to the
    AFlow framework, including loading from AFlow-formatted data files and
    supporting asynchronous evaluation for workflows.
    
    Attributes:
        path: Path to the directory containing AFlow-formatted GSM8K files.
        mode: Data loading mode ("train", "dev", "test", or "all").
        _train_data: Training dataset loaded from AFlow format.
        _dev_data: Development dataset loaded from AFlow format.
        _test_data: Test dataset loaded from AFlow format.
    """

    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/aflow/gsm8k")
        super().__init__(path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset="gsm8k", save_folder=self.path)
        return load_json(path=file_path, type="jsonl")
        
    def _load_data(self):

        if self.mode == "train" or self.mode == "all":
            logger.info(f"Loading train data from {AFLOW_DATASET_FILES_MAP['gsm8k']['train']}")
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["gsm8k"]["train"])
        if self.mode == "dev" or self.mode == "all":
            logger.info(f"Loading dev data from {AFLOW_DATASET_FILES_MAP['gsm8k']['dev']}")
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["gsm8k"]["dev"])
        if self.mode == "test" or self.mode == "all":
            logger.info(f"Loading test data from {AFLOW_DATASET_FILES_MAP['gsm8k']['test']}")
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["gsm8k"]["test"])       
    
    async def async_evaluate(self, graph: Callable, example: Any) -> float:

        input_text = example["question"] 
        label = self._get_label(example) 
        output = await graph(input_text)
        metrics = await super().async_evaluate(prediction=output, label=label)
        return metrics["solve_rate"]