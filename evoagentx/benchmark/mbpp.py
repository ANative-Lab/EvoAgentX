import os 
import regex
from typing import Union, Any, List, Callable
from ..core.logging import logger
from .benchmark import Benchmark
from ..utils.utils import download_file
from ..core.module_utils import load_json
from ..utils.aflow_utils.data_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data
from .scorers import CodeScorer


def download_raw_mbpp_data(name: str, save_folder: str):
    url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"
    logger.info(f"Downloading MBPP data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, name))


def load_mbpp_data(data_path: str):

    """
    load MBPP data from the given path and convert to HumanEval format
    """

    def extract_func_name(func_header: str) -> str:
        func_name_pattern = r"def\s+([a-zA-Z_]\w*)\s*\("
        match = regex.search(func_name_pattern, func_header)
        if match:
            return match.group(1)
        else:
            return None

    def extract_func_header(code: str, test_list: List[str]) -> str:
        lines = code.split("\n")
        imports, defs = [], []
        for line in lines:
            if line.startswith("def "):
                break
            imports.append(line)
        for line in lines:
            if line.startswith("def "):
                defs.append(line)
        func_head = None
        for header in defs:
            func_name = extract_func_name(header)
            if func_name is None:
                continue 
            if all(func_name in test for test in test_list):
                func_head = header 
                break 
        if func_head is None:
            logger.warning(f"No function header found for {code}")
        return ("\n".join(imports) + "\n\n" + func_head).strip() 

    data = load_json(data_path, type="json")

    for example in data:
        original_prompt = example["prompt"] 
        code = example["code"]
        test_list = [assert_str.strip() for assert_str in example["test_list"]]
        func_header = extract_func_header(code, test_list)

        if example["task_id"] == 56:
            # change the `check` function to `check_answer`
            func_header = func_header.replace("check", "check_answer")
            code = code.replace("check", "check_answer")
            test_list = [test.replace("check", "check_answer") for test in test_list]

        prompt = example["prompt"] + "\n\n" + func_header + "\n"
        canonical_solution = code 
        test = "def check(candidate):\n    " + "\n    ".join(test_list) + "\n" 
        entry_point = extract_func_name(func_header)

        example["prompt"] = prompt 
        example["entry_point"] = entry_point 
        example["canonical_solution"] = canonical_solution 
        example["test"] = test 
        example["original_prompt"] = original_prompt 
    
    return data


class MBPP(Benchmark):

    """Benchmark class for evaluating code generation on the MBPP dataset.
    
    MBPP (Mostly Basic Python Programming) is a collection of Python programming 
    problems designed to test a model's ability to generate functionally correct 
    code from natural language descriptions. This class handles loading the dataset, 
    evaluating solutions, and computing metrics such as pass@k.
    
    The original MBPP format is transformed to be compatible with the HumanEval
    benchmark format, allowing for consistent evaluation infrastructure.
    
    Each MBPP example has the following structure:
    {
        "task_id" (int): 2, 
        "prompt" (str): "Write a function to find the shared elements from the given two lists.",
        "code" (str): "def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res) ", 
        "test_imports": [] 
        "test_list" (List[str]): ['assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))', 'assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))', 'assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))']
    }
    
    Attributes:
        k: An integer or list of integers specifying which pass@k metrics to compute
    """

    def __init__(self, path: str = None, mode: str = "all", timeout: int = 60, k: Union[int, list] = 1,**kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/mbpp")
        self.k = k 
        self.timeout = timeout
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)
        self.scorer = CodeScorer(
            k=self.k,
            timeout=self.timeout,
            prompt_joiner="\n",
            prompt_getter=lambda task_id: self.get_example_by_id(str(task_id))["prompt"],
        )
    
    def _load_data(self):

        data_path = os.path.join(self.path, "sanitized-mbpp.json")
        if not os.path.exists(data_path):
            download_raw_mbpp_data(name="sanitized-mbpp.json", save_folder=self.path)
        
        # load data 
        if self.mode == "train" or self.mode == "all":
            self._train_data = None 
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = None 
        if self.mode == "test" or self.mode == "all":
            self._test_data = load_mbpp_data(data_path)
    
    def _get_id(self, example: Any) -> Any:
        # task_id is an int in the raw data; example ids must be strings.
        return str(example["task_id"])

    def _get_label(self, example: Any) -> Any:
        # return the unit test code
        return {
            "task_id": example["task_id"],
            "canonical_solution": example["canonical_solution"],
            "test": example["test"],
            "entry_point": example["entry_point"]
        }
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        """
        Evaluate the solution code.

        Args:
            prediction (str | List[str]): The solution code(s).
            label (dict | List[dict]): The unit test code(s).

        Returns:
            dict: The evaluation metrics (pass@k).
        """
        return self.scorer.score(prediction=prediction, label=label)
    


class AFlowMBPP(MBPP):

    """
    AFlow-specific implementation of MBPP benchmark.
    """

    def __init__(self, path: str = None, mode: str = "all", timeout: int = 60, k: Union[int, list] = 1,**kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/aflow/mbpp")
        super().__init__(path=path, mode=mode, timeout=timeout, k=k, **kwargs)
        self.scorer = CodeScorer(
            k=self.k,
            timeout=self.timeout,
            prompt_joiner="\n",
            prompt_getter=lambda task_id: self.get_example_by_id(str(task_id))["prompt"],
            use_entrypoint_as_input=False,
        )

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset="mbpp", save_folder=self.path)
        
        return load_json(path=file_path, type="jsonl")

    def _load_data(self):

        if self.mode == "train" or self.mode == "all":
            logger.info(f"Loading train data from {AFLOW_DATASET_FILES_MAP['mbpp']['train']}")
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["mbpp"]["train"])
        if self.mode == "dev" or self.mode == "all":
            logger.info(f"Loading dev data from {AFLOW_DATASET_FILES_MAP['mbpp']['dev']}")
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["mbpp"]["dev"])
        if self.mode == "test" or self.mode == "all":
            logger.info(f"Loading test data from {AFLOW_DATASET_FILES_MAP['mbpp']['test']}")
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["mbpp"]["test"])
        
        # load test cases 
        self._test_cases = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["mbpp"]["test_cases"])
    
    def _get_label(self, example: Any):
        return {
            "task_id": example["task_id"], 
            "canonical_solution": example["code"], 
            "test": example["test"], 
            "entry_point": example["entry_point"]
        }
    
    def extract_test_cases_with_entry_point(self, entry_point: str):

        hardcoded_cases = {
            "remove_odd": "",
            "replace_spaces": "",
            "snake_to_camel": "",
            "Split": "",
            "swap_List": "",
            "square_Sum": "",
            "sort_sublists": "",
            "unique_sublists": "",
        }
        if entry_point in hardcoded_cases:
            return hardcoded_cases[entry_point]
        
        for case in self._test_cases:
            if case["entry_point"] == entry_point:
                return case["test"]
        
        return None
    
    async def async_evaluate(self, graph: Callable, example: Any) -> float:

        # generate solution 
        prompt, entry_point = example["prompt"], example["entry_point"]
        solution = await graph(prompt, entry_point)
        label = self._get_label(example)
        metrics = await super().async_evaluate(prediction=solution, label=label)
        return metrics["pass@1"]
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        """
        Evaluate the solution code.

        Args:
            prediction (str | List[str]): The solution code(s).
            label (dict | List[dict]): The unit test code(s).

        Returns:
            dict: The evaluation metrics (pass@k).
        """
        return self.scorer.score(prediction=prediction, label=label)