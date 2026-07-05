import os
import zipfile
import requests
from typing import Any, List, Callable

from ..core.logging import logger
from .benchmark import Benchmark
from .scorers import MathScorer
from ..utils.utils import make_parent_folder
from ..core.module_utils import load_json
from ..utils.aflow_utils.data_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data


def download_raw_math_data(save_folder: str):
    """
    Download the MATH data from the modelscope website.
    """
    url = "https://www.modelscope.cn/datasets/opencompass/competition_math/resolve/master/data/MATH.zip"
    logger.info(f"Downloading MATH data from {url} ...")
    save_file_path = os.path.join(save_folder, "MATH.zip")

    make_parent_folder(save_file_path)
    if not os.path.exists(save_file_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    
    with zipfile.ZipFile(save_file_path, "r") as zip_ref:
        zip_ref.extractall(save_folder)
    if os.path.exists(save_file_path):
        os.remove(save_file_path)


class MATH(Benchmark):

    """Benchmark class for evaluating mathematical reasoning on the MATH dataset.
    
    MATH is a dataset of challenging competition mathematics problems,
    spanning various difficulty levels and subject areas. This class handles
    loading the dataset, extracting answers, evaluating solutions through
    symbolic and numerical comparisons, and computing accuracy metrics.
    
    The dataset includes problems across 7 subject areas (Algebra, Geometry, etc.)
    and 5 difficulty levels. Each problem contains LaTeX-formatted
    questions and solutions.
    
    Each MATH example has the following structure:
    {
        "id": "test-1", 
        "problem": "the problem", 
        "solution": "the solution",
        "level": "Level 1", # "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level ?"
        "type": "Algebra", # 'Geometry', 'Algebra', 'Intermediate Algebra', 'Counting & Probability', 'Precalculus', 'Number Theory', 'Prealgebra'
    }
    
    The benchmark evaluates answers using symbolic math equality checking
    and numerical approximation to handle equivalent mathematical expressions.
    """

    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/math")
        self.scorer = MathScorer(mode="symbolic", tol=1e-3)
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)
    
    def _load_data_from_folders(self, data_folder: str) -> List[dict]:
        if data_folder is None:
            return None
        data = []
        typ = "train" if "train" in data_folder else "test"
        sub_data_folders = os.listdir(data_folder)
        i = 0
        logger.info(f"loading MATH data from {data_folder} ...")
        for sub_data_folder in sub_data_folders:
            if os.path.isdir(os.path.join(data_folder, sub_data_folder)):
                files = os.listdir(os.path.join(data_folder, sub_data_folder))
                for file in files:
                    if file.endswith(".json"):
                        example = {"id": f"{typ}-{i+1}"}
                        example.update(load_json(os.path.join(data_folder, sub_data_folder, file), type="json"))
                        data.append(example)
                        i += 1
        return data
                
    def _load_data(self):
        if not os.path.exists(os.path.join(self.path, "MATH")):
            download_raw_math_data(save_folder=self.path)
        data_folder = os.path.join(self.path, "MATH")

        # load data 
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_folders(data_folder=os.path.join(data_folder, "train"))
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = None 
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_folders(data_folder=os.path.join(data_folder, "test"))
    
    def _get_label(self, example: Any) -> Any:
        return example["solution"]
    
    def _get_id(self, example: Any) -> Any:
        return example["id"] 
    
    def extract_answer(self, text: str) -> str: 
        return self.scorer._extract_symbolic(text)
    
    # Acknowledgement: https://github.com/geekan/MetaGPT/blob/main/metagpt/ext/aflow/benchmark/math.py#L40 
    def math_equal(self, prediction: Any, reference: Any) -> bool:
        return self.scorer._symbolic_equal(prediction, reference)
    
    def is_digit(self, num: Any) -> bool:
        return self.scorer._is_digit(num)
    
    def parse_digits(self, num: Any) -> float:
        return self.scorer._parse_digits(num)

    def symbolic_equal(self, a: Any, b: Any) -> bool:
        return self.scorer._symbolic_expression_equal(a, b)

    def evaluate(self, prediction: Any, label: Any) -> dict:
        return self.scorer.score(prediction=prediction, label=label)
    

class AFlowMATH(MATH):

    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/aflow/math")
        super().__init__(path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset="math", save_folder=self.path)
        return load_json(path=file_path, type="jsonl")

    def _load_data(self):

        if self.mode == "train" or self.mode == "all":
            logger.info(f"Loading train data from {AFLOW_DATASET_FILES_MAP['math']['train']}")
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["math"]["train"])
        if self.mode == "dev" or self.mode == "all":
            logger.info(f"Loading dev data from {AFLOW_DATASET_FILES_MAP['math']['dev']}")
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["math"]["dev"])
        if self.mode == "test" or self.mode == "all":
            logger.info(f"Loading test data from {AFLOW_DATASET_FILES_MAP['math']['test']}")
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["math"]["test"])       
    
    async def async_evaluate(self, graph: Callable, example: Any) -> float:

        problem = example["problem"]
        label = self._get_label(example)
        output = await graph(problem)
        metrics = await super().async_evaluate(prediction=output, label=label)
        return metrics["solve_rate"]
