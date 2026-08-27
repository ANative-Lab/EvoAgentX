# copied from: https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/benchmarks/code_generation.py

import json
import os
import zlib
import pickle
import base64
import warnings
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from datasets import load_dataset


_RELEASE_FILES = {
    "release_v1": ("test.jsonl",),
    "release_v2": ("test.jsonl", "test2.jsonl"),
    "release_v3": ("test.jsonl", "test2.jsonl", "test3.jsonl"),
    "release_v4": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"),
    "release_v5": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"),
    "release_v6": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"),
    "release_latest": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"),
}


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except Exception:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def load_code_generation_dataset(release_version="release_v1", cache_dir: str = None, start_date=None, end_date=None) -> list[CodeGenerationProblem]:
    release_files = _RELEASE_FILES.get(release_version)
    if release_files is None:
        raise ValueError(
            f"Unsupported LiveCodeBench release version {release_version!r}. "
            f"Available versions: {sorted(_RELEASE_FILES)}"
        )

    local_files = [] if cache_dir is None else [
        os.path.join(cache_dir, file_name) for file_name in release_files
    ]
    if local_files and all(os.path.isfile(file_path) for file_path in local_files):
        records = []
        for file_path in local_files:
            with open(file_path, encoding="utf-8") as file:
                records.extend(json.loads(line) for line in file if line.strip())
        dataset = [CodeGenerationProblem(**record) for record in records]
    else:
        try:
            dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag=release_version, trust_remote_code=True, cache_dir=cache_dir)
            dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
        except RuntimeError as error:
            message = (
                "LiveCodeBench code_generation could not be loaded remotely. "
                "The official dataset still uses a loading script, which datasets>=4 no longer supports. "
                "Download the official dataset files into the directory passed as path (for example: "
                "hf download livecodebench/code_generation_lite --repo-type dataset --local-dir <path>), "
                f"including {', '.join(release_files)} for {release_version}."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            raise RuntimeError(message) from error
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]
    
    # print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    # print(f"Loaded {len(dataset)} problems")
    return dataset


if __name__ == "__main__":
    dataset = load_code_generation_dataset()
