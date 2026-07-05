import os 
from typing import Any, List
from .benchmark import Benchmark
from .scorers import QAScorer
from ..core.logging import logger
from ..utils.utils import download_file


NQ_FILES_MAP = {"train": "nq-train.qa.csv", "dev": "nq-dev.qa.csv", "test": "nq-test.qa.csv"}
VALID_RAW_NQ_FILES = [file for file in list(NQ_FILES_MAP.values()) if file is not None]

def download_raw_nq_data(name: str, save_folder: str):
    assert name in VALID_RAW_NQ_FILES, f"'{name}' is an invalid nq file name. Available file names: {VALID_RAW_NQ_FILES}"
    file_type_map = {file_name: typ for typ, file_name in NQ_FILES_MAP.items()}
    typ = file_type_map[name]
    url = f"https://dl.fbaipublicfiles.com/dpr/data/retriever/{name}"
    logger.info(f"Downloading NQ {typ} data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, name))


def load_tsv_data(file_path: str) -> List[dict]:

    base_name = os.path.basename(file_path)
    file_type_map = {file_name: typ for typ, file_name in NQ_FILES_MAP.items()}
    assert base_name in file_type_map, f"'{base_name}' is an invalid nq file name. Available file names: {VALID_RAW_NQ_FILES}"

    typ = file_type_map[base_name]

    data = [] 
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            question, answers = line.strip().split("\t")
            answers = eval(answers)
            data.append({"id": f"{typ}-{i+1}", "question": question, "answers": answers})
    return data


class NQ(Benchmark):

    """Benchmark class for evaluating question answering on Natural Questions dataset.
    
    Natural Questions (NQ) is a dataset for open-domain question answering,
    containing real questions from Google Search and answers from Wikipedia.
    This class handles loading the dataset, evaluating answers, and computing
    metrics like exact match and F1 score.
    
    Each NQ example has the following structure:
    {
        "id": str, 
        "question": str, 
        "answers": List[str]
    }
    
    The benchmark evaluates answers using exact match, F1 score, and accuracy metrics.
    """

    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        path = os.path.expanduser(path or "~/.evoagentx/data/nq")
        self.scorer = QAScorer(multi_reference=True, metrics=["f1", "em", "acc"])
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_raw_nq_data(name=file_name, save_folder=self.path)
        logger.info(f"loading NQ data from {file_path} ...")
        return load_tsv_data(file_path=file_path)
            
    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=NQ_FILES_MAP["train"])
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=NQ_FILES_MAP["dev"])
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=NQ_FILES_MAP["test"])

    def _get_label(self, example: Any) -> Any:
        return example["answers"]
    
    def _get_id(self, example: Any) -> Any:
        return example["id"]
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        return self.scorer.score(prediction=prediction, label=label)