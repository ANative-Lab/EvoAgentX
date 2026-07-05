"""
BIG-Bench Hard Benchmark Module

This module implements the BIGBenchHard benchmark evaluation framework.
BIGBenchHard is a challenging subset of 23 tasks from the BIG-bench evaluation suite,
designed to test reasoning capabilities of language models.
"""

import os
import random  
import numpy as np
import torch
from typing import Any, List, Optional

from .benchmark import Benchmark
from .scorers import QAScorer
from ..core.logging import logger
from ..core.module_utils import load_json
from ..utils.utils import download_file

# Task categorization for different evaluation types
MULTIPLE_CHOICE_TASKS = [
    'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table', 
    'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects', 
    'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'movie_recommendation', 
    'salient_translation_error_detection', 'reasoning_about_colored_objects', 
]

FREE_FORM_TASKS = [
    'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding', 
    'boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies', 
]

# Complete task mapping to data files
ALL_TASKS = {task: f"{task}.json" for task in MULTIPLE_CHOICE_TASKS + FREE_FORM_TASKS}

def download_raw_bigbenchhard_data(task_name: str, save_folder: str):
    """
    Download raw BIGBenchHard data for a specific task.
    
    Args:
        task_name: The name of the task to download
        save_folder: Directory to save the downloaded data file
        
    Raises:
        AssertionError: If task_name is not a valid BIGBenchHard task
    """
    assert task_name in ALL_TASKS, f"'{task_name}' is an invalid bigbenchhard task name. Available tasks: {list(ALL_TASKS.keys())}"
    file_name = ALL_TASKS[task_name]
    url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{file_name}"
    logger.info(f"Downloading BIGBenchHard '{task_name}' data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, file_name))
    
def set_seed(seed: int):
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed: The random seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BIGBenchHard(Benchmark):
    """
    Benchmark class for BIGBenchHard dataset evaluation.
    
    BIGBenchHard is a subset of 23 challenging tasks from the BIG-bench evaluation suite.
    Each task example has the following structure:
    {
        "input": str,    # The input question/problem
        "target": str    # The expected answer/output
    }
    
    The benchmark supports automatic data splitting for training/validation purposes
    and evaluates predictions using exact match scoring.
    """

    def __init__(self, task: str, path: str = None, mode: str = "all", dev_sample_num: int = 0, seed: int = 10, **kwargs):
        """
        Initialize BIGBenchHard benchmark.
        
        Args:
            task: The specific BIGBenchHard task name
            path: Path to store the dataset. Defaults to ~/.evoagentx/data/bigbenchhard/{task}
            mode: Data loading mode. Defaults to "all"
            dev_sample_num: Number of samples to use for dev set. If 0, all data goes to test set
            seed: Random seed for reproducibility. Defaults to 10
            **kwargs: Additional parameters for customization
            
        Raises:
            ValueError: If task is not a valid BIGBenchHard task name
        """
        if task not in ALL_TASKS:
            raise ValueError(f"Unknown task '{task}'. Available tasks: {list(ALL_TASKS.keys())}")
            
        self.task = task
        self.file_name = ALL_TASKS[task]
        self.dev_sample_num = dev_sample_num
        self.seed = seed
        self.scorer = QAScorer(metrics=["em"], normalize_fn=self._normalize_for_scoring)
        
        # Set default path if not provided
        path = os.path.expanduser(path or f"~/.evoagentx/data/bigbenchhard/{task}")
        
        super().__init__(name=f"BIGBenchHard-{self.task}", path=path, mode=mode, **kwargs)

    def _normalize_for_scoring(self, value: Any) -> Any:
        if self.task == "dyck_languages":
            return str(value).replace(" ", "")
        return value

    def _load_data_from_file(self, file_name: str) -> Optional[List[dict]]:
        """
        Load data from a specific file.
        
        Args:
            file_name: Name of the file to load
            
        Returns:
            List of loaded examples or None if file doesn't exist
        """
        if file_name is None:
            return None
            
        file_path = os.path.join(self.path, file_name)
        
        # Download data if not exists locally
        if not os.path.exists(file_path):
            download_raw_bigbenchhard_data(task_name=self.task, save_folder=self.path)
            
        logger.info(f"Loading BIGBenchHard data from {file_path}...")
        data = load_json(path=file_path, type="json")
        return data.get("examples", [])

    def _load_data(self):
        """
        Load and split data according to mode and dev_sample_num settings.
        
        Data splitting logic:
        - If dev_sample_num > 0: randomly samples examples for dev set, rest go to test set
        - If dev_sample_num = 0: all data goes to test set for evaluation
        - No training data provided (BIGBenchHard is designed for few-shot evaluation)
        """
        # Load the raw task data
        task_data = self._load_data_from_file(file_name=self.file_name)
        
        # Handle case where no data is loaded
        if task_data is None:
            logger.warning(f"No data loaded for task {self.task}")
            self._train_data = []
            self._dev_data = []
            self._test_data = []
            return
        
        # BIGBenchHard doesn't provide training data - designed for few-shot evaluation
        self._train_data = []
        
        # Split data based on dev_sample_num parameter
        if self.dev_sample_num > 0 and len(task_data) > self.dev_sample_num:
            logger.info(f"Sampling {self.dev_sample_num} examples for dev set, rest for test set.")
            if self.seed is not None:
                set_seed(self.seed)
            dev_subset = random.sample(task_data, self.dev_sample_num)
            self._dev_data = dev_subset
            self._test_data = [item for item in task_data if item not in dev_subset]
            
        else:
            # Handle edge cases
            if self.dev_sample_num > 0:
                logger.warning(f"dev_sample_num ({self.dev_sample_num}) >= total data size ({len(task_data)}). "
                             f"Using all data for dev set, none for test set.")
                self._dev_data = task_data
                self._test_data = []
            else:
                logger.info("dev_sample_num is 0, using all data for test set.")
                self._dev_data = []
                self._test_data = task_data
                
    def get_input_keys(self) -> List[str]:
        """
        Return the input keys expected by the benchmark.
        
        Returns:
            List containing "input" as the key for the problem text
        """
        return ["input"]
    
    def _get_label(self, example: Any) -> Any:
        """
        Extract the ground truth label from an example.
        
        Args:
            example: The benchmark example
            
        Returns:
            The target answer/label
        """
        return example["target"]
    
    def _get_id(self, example: Any) -> Any:
        """
        Extract the unique identifier from an example.
        
        BIGBenchHard examples don't have explicit IDs, so we use input text as identifier.
        
        Args:
            example: The benchmark example
            
        Returns:
            The input text as a unique identifier
        """
        return example.get("input", None)
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        """
        Score a prediction against the ground truth label.
        
        Uses exact match scoring with task-specific handling for certain tasks.
        
        Args:
            prediction: The predicted answer
            label: The ground truth answer
            
        Returns:
            Dictionary containing the exact match score
        """
        return self.scorer.score(prediction=prediction, label=label)
