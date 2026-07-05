import os 
import gzip 
import shutil
from typing import Union, Any, Callable
from .benchmark import Benchmark
from ..core.logging import logger 
from ..utils.utils import download_file 
from ..core.module_utils import load_json
from ..utils.aflow_utils.data_utils import AFLOW_DATASET_FILES_MAP, download_aflow_benchmark_data
from .scorers import CodeScorer


def download_raw_humaneval_data(save_folder: str): 
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    logger.info(f"Downloading HumanEval data from {url} ...")
    save_file_path = os.path.join(save_folder, "HumanEval.jsonl.gz")
    download_file(url=url, save_file=save_file_path)
    with gzip.open(save_file_path, "rb") as f_in, open(os.path.join(save_folder, "HumanEval.jsonl"), "wb") as f_out: 
        shutil.copyfileobj(f_in, f_out) 
    if os.path.exists(save_file_path):
        os.remove(save_file_path)


def load_humaneval_data(data_path: str):
    data = load_json(data_path, type="jsonl") 
    # Handle 115 prompt to make its docstring well-formed
    for example in data:
        if example["task_id"] == "HumanEval/115":
            example["prompt"] = "import math\n" + example["prompt"].replace("import math", "")
    return data 


class HumanEval(Benchmark):

    """Benchmark class for evaluating code generation on HumanEval.
    
    HumanEval is a collection of Python programming problems designed to test
    a model's ability to generate functionally correct code from natural language
    descriptions. This class handles loading the dataset, evaluating solutions,
    and computing metrics such as pass@k.
    
    Each HumanEval example has the following structure:
    {
        "task_id": "HumanEval/0", 
        "prompt": "from typing import List\n\ndef func_name(*args, **kwargs) -> return_type\n    "function description"\n\n", 
        "entry_point": "func_name",
        "canonical_solution": "canonical solution (code)",
        "test": "METADATA = {xxx}\n\n\ndef check(candidate):\n assert candidate(inputs) == output\n"
    }
    
    Attributes:
        k: An integer or list of integers specifying which pass@k metrics to compute
    """

    def __init__(self, path: str = None, mode: str = "all", timeout: int = 60, k: Union[int, list] = 1, **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/humaneval")
        self.k = k 
        self.timeout = timeout
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)
        self.scorer = CodeScorer(
            k=self.k,
            timeout=self.timeout,
            prompt_joiner="",
            prompt_getter=lambda task_id: self.get_example_by_id(task_id)["prompt"],
            special_cases_handler=self.handle_special_cases,
        )

    def _load_data(self):

        data_path = os.path.join(self.path, "HumanEval.jsonl")
        if not os.path.exists(data_path):
            download_raw_humaneval_data(self.path)
        
        # load data  
        if self.mode == "train" or self.mode == "all":
            self._train_data = None 
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = None 
        if self.mode == "test" or self.mode == "all":
            self._test_data = load_humaneval_data(data_path)

    def _get_label(self, example: Any):
        # return the unit test code
        return {
            "task_id": example["task_id"],
            "canonical_solution": example["canonical_solution"],
            "test": example["test"],
            "entry_point": example["entry_point"]
        }
    
    def _get_id(self, example: Any):
        return example["task_id"]
    
    def handle_special_cases(self, task_id: str, solution: str, test: str) -> bool:
        """
        Handle special cases for HumanEval.
        """
        if task_id == "HumanEval/50":
            solution = (
                '\n\ndef encode_shift(s: str):\n    """\n    returns encoded string by shifting every character by 5 in the alphabet.\n    """\n    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n\n'
                + solution
            )
            return solution, test 
        
        return solution, test

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
    

class HumanEvaluPlus(HumanEval):

    """Extended version of HumanEval with additional test cases and inputs.
    
    HumanEvalPlus extends the original HumanEval benchmark with additional
    test cases, input validation contracts, and more rigorous testing.
    
    Each HumanEvalPlus example has the following structure:
    {
        "task_id": "HumanEvalPlus/0",
        "prompt": "function signature with docstring such as: from typing import List\n\ndef func_name(*args, **kwargs) -> return_type\n    "function description"\n\n", 
        "entry_point": "func_name",
        "canonical_solution": "canonical solution (code)",
        "test": "METADATA = {xxx}\n\n\ndef check(candidate):\n assert candidate(inputs) == output\n", 
        "contract": "string", # the assertions for the function's input (validity)
        "base_input": list, # the test inputs from original HumanEval
        "plus_input": list, # the test inputs brought by EvalPlus
        "atol": int, # the absolute tolerance for diff-testing
    }
    """
    pass 


class AFlowHumanEval(HumanEval):

    """
    AFlow-specific implementation of HumanEval benchmark.
    """

    def __init__(self, path: str = None, mode: str = "all", timeout: int = 60, k: Union[int, list] = 1, **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/aflow/humaneval")
        super().__init__(path=path, mode=mode, timeout=timeout, k=k, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset="humaneval", save_folder=self.path)
        
        return load_json(path=file_path, type="jsonl")

    def _load_data(self):

        if self.mode == "train" or self.mode == "all":
            logger.info(f"Loading train data from {AFLOW_DATASET_FILES_MAP['humaneval']['train']}")
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["train"])
        if self.mode == "dev" or self.mode == "all":
            logger.info(f"Loading dev data from {AFLOW_DATASET_FILES_MAP['humaneval']['dev']}")
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["dev"])
        if self.mode == "test" or self.mode == "all":
            logger.info(f"Loading test data from {AFLOW_DATASET_FILES_MAP['humaneval']['test']}")
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["test"])
        
        # load test cases 
        self._test_cases = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP["humaneval"]["test_cases"])
    
    def extract_test_cases_with_entry_point(self, entry_point: str):
        """
        Extract test cases with the given entry point.
        """

        hardcoded_cases = {
            "find_zero": "",
            "decode_cyclic": "",
            "decode_shift": "",
            "by_length": "",
            "add": "",
            "triangle_area": "",
            "correct_bracketing": "",
            "solve": "",
            "sum_squares": "",
            "starts_one_ends": "",
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
    
