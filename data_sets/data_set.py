import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class Data_set(ABC):
    """
    Abstract class for datasets. All datasets should inherit from this class.

    We use the underline to differentiate from huggingface
    datasets (dataset) and Datasets (Dataset).
    """

    def __init__(
        self,
        cache_root: str,
        dataset_name: str,
        model_name: str,
        approach_name: str,
        tokenizer_name: str,
        num_data_preprocess=int(1e6),
        num_data_test=int(1e6),
        **kwargs,
    ) -> None:
        """
        Load dataset either locally or online.

        Args:
            cache_root: The folder to store the cache.
            dataset_name: The name of the dataset.
            model_name: The model used to generate the output.
            approach_name: The approach used to generate the output.
            tokenizer_name: The tokenizer used to process the data.
            num_data_preprocess: The number of data to preprocess. Preprocessed data is a subset of the online (total) data.
            num_data_test: The number of data to test. Test data is a subset of the preprocessed data.
        Return:
            None
        """

        # Imutable attributes
        self.cache_root = cache_root
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.approach_name = approach_name
        self.cache_folder = os.path.join(cache_root, dataset_name, model_name)
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        self.preprocessed_data_path = os.path.join(self.cache_folder, "preprocessed_data.json")
        self.result_path = os.path.join(self.cache_folder, f"{approach_name}.json")
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_data_preprocess = num_data_preprocess
        self.num_data_test = num_data_test
        assert (
            self.num_data_test <= self.num_data_preprocess
        ), "num_data_test should be less than or equal to num_data_preprocess"
        self.kwargs = kwargs

        # Mutable attributes
        self.data: pd.DataFrame = None  # Hold the preprocessed data
        self.result: pd.DataFrame = (
            None  # Hold the result of the test using (dataset_name, model_name, approach_name, tokenizer_name)
        )

        # Construct mutable attributes
        if os.path.exists(
            self.preprocessed_data_path
        ):  # If we need to force-reload the data, we need to remove the preprocessed_data.json file first
            self.data = pd.read_json(self.preprocessed_data_path)
            logger.info(f"Loaded dataset locally from {self.preprocessed_data_path}")
        else:
            self.data = self.load_data_online()  # Implemented in the subclass
            logger.info("Loaded dataset online")

            # Common processing for all datasets
            self.data = self.data[: self.num_data_preprocess]
            self.data = self.data.apply(
                lambda row: self.create_groundtruth_fields(row), axis=1
            ).apply(lambda row: self.create_prompt_fields(row), axis=1)

            self.data.to_json(self.preprocessed_data_path, orient="records", indent=4)
        self.data = self.data[:num_data_test]

        if os.path.exists(self.result_path):
            self.result = pd.read_json(self.result_path)
            # We update the "finished" field each time we load the result
            self.result = self.result.apply(lambda row: self.check_finished(row), axis=1)
        else:
            # self.result must have the column "finished", with 0 indicating unfinished and 1 indicating finished
            self.result = pd.DataFrame({"finished": [0] * len(self.data)})
        num_data_test_finished = self.result["finished"].sum()
        logger.info(f"The progress is {num_data_test_finished}/{num_data_test}")

    @staticmethod
    def create_dataset(
        cache_root: str,
        dataset_name: str,
        model_name: str,
        approach_name: str,
        tokenizer_name: str,
        num_data_preprocess=int(1e6),
        num_data_test=int(1e6),
        **kwargs,
    ):
        if dataset_name == "sharegpt":
            Class = ShareGPT
        elif dataset_name == "mooncake":
            Class = Mooncake
        else:
            raise NotImplementedError

        return Class(
            cache_root=cache_root,
            dataset_name=dataset_name,
            model_name=model_name,
            approach_name=approach_name,
            tokenizer_name=tokenizer_name,
            num_data_preprocess=num_data_preprocess,
            num_data_test=num_data_test,
            **kwargs,
        )

    @abstractmethod
    def load_data_online(self) -> pd.DataFrame:
        """
        Load raw data. Inline function.

        Returns:
            The raw data in pandas DataFrame format.
        """
        raise NotImplementedError

    @abstractmethod
    def create_groundtruth_fields(self, row: Dict) -> Dict:
        """
        Create the row["groundtruth"], row["groundtruth_token_ids"], row["groundtruth_length"] columns in the dataset. To be used with the pandas apply function.

        Args:
            row: One row of the dataset.

        Returns:
            The updated row.
        """
        raise NotImplementedError

    @abstractmethod
    def create_prompt_fields(self, row: Dict) -> Dict:
        """
        Create the row["prompt"], row["prompt_token_ids"], row["prompt_length"] columns in the dataset. To be used with the pandas apply function.

        Args:
            row: One row of the dataset.

        Returns:
            The updated row.
        """
        raise NotImplementedError

    @abstractmethod
    def _calc_accuracy(self, row: Dict) -> Dict:
        """
        Check the correctness of the model output and store the result in the f"accuracy" column.

        Args:
            row: One row of the dataset.

        Returns:
            The row with the row["accuracy"] updated.
        """
        raise NotImplementedError

    """
    The following methods are common to all subclasses and SHOULD NOT BE OVERRIDDEN.
    This constraint is to ensure consistent interfaces.
    If you have a strong need to override some of the following methods,
    please refer to how calc_accruacy and _calc_accuracy methods are implemented.
    """

    def calc_accuracy(self) -> None:
        """
        Compare the model output with the groundtruth and calculate the accuracy.
        Store the accuracy in the "accuracy" column.

        Args:
            approach: The approach used to generate the output.
        """
        assert "output" in self.result.columns, f"output not in the result"
        assert "groundtruth" in self.result.columns, "groundtruth not in the result"

        self.result = self.result.apply(lambda row: self._calc_accuracy(row), axis=1)
        self.result.to_json(self.result_path, orient="records", indent=4)

    def update_result(
        self,
        row: int,
        column: str,
        value: Any,
    ) -> None:
        """
        Update the result with self.result.loc[row, column] = value. If column does not exists,
        create the column filled with None value.

        Args:
            row: The row index.
            column: The column name.
            value: The value to be updated.

        Returns:
            None
        """
        if column not in self.data.columns:
            self.result[column] = [None] * len(self.data)
        self.result.loc[row, column] = value
        # TODO(hjh): If this line slows down the process, save the result every X updates.
        self.result.to_json(self.result_path, orient="records", indent=4)

    def check_finished(self, row: Dict) -> Dict:
        """
        Check if the row is finished. If the row is finished, set row["finished"] = 1.
        The condition can be changed according to users' needs.

        Args:
            row: One row of the dataset.

        Returns:
            The updated row.
        """
        # TODO(hjh): Change the condition according to the needs
        row["finished"] = 1 if len(row) > 1 and row.notna().all() else 0
        return row

    def __iter__(self):
        for idx, row in self.data.iterrows():
            if row["finished"]:
                continue
            yield idx, row["prompt_token_ids"], row["groundtruth_token_ids"]

    def __len__(self) -> int:
        return len(self.data)


class Mooncake(Data_set):

    BLOCK_SIZE = 512

    def load_data_online(self) -> pd.DataFrame:
        return pd.read_json(
            Path(__file__).parent / "online_data" / "mooncake_trace.jsonl", lines=True
        )

    def create_groundtruth_fields(self, row: Dict) -> Dict:
        row["groundtruth"] = 0

        # The following lines take into account the dependecy of each request, which relates to the inter-request KV reuse
        token_ids_2Dlist = [
            [row["hash_ids"][i]] * Mooncake.BLOCK_SIZE for i in range(len(row["hash_ids"]))
        ]
        token_ids_1Dlist = [item for sublist in token_ids_2Dlist for item in sublist]
        row["groundtruth_token_ids"] = token_ids_1Dlist[
            row["input_length"] : row["input_length"] + row["output_length"]
        ]

        row["groundtruth_length"] = row["output_length"]
        return row

    def create_prompt_fields(self, row: Dict) -> Dict:
        row["prompt"] = 0

        # The following lines take into account the dependecy of each request, which relates to the inter-request KV reuse
        token_ids_2Dlist = [
            [row["hash_ids"][i]] * Mooncake.BLOCK_SIZE for i in range(len(row["hash_ids"]))
        ]
        token_ids_1Dlist = [item for sublist in token_ids_2Dlist for item in sublist]
        row["prompt_token_ids"] = token_ids_1Dlist[: row["input_length"]]

        row["prompt_length"] = row["input_length"]
        return row

    def _calc_accuracy(self, row: Dict, approach: str) -> Dict:
        row["accuracy"] = 0
        return row


class ShareGPT(Data_set):
    # TODO(hjh): This dataset is not ready
    def load_data_online(self) -> pd.DataFrame:
        # We first download the dataset from the huggingface hub and put the json file in the following position
        part1 = pd.read_json(Path(__file__).parent / "online_data" / "sg_90k_part1.json")
        part2 = pd.read_json(Path(__file__).parent / "online_data" / "sg_90k_part2.json")
        return pd.concat([part1, part2])

    def create_groundtruth_fields(self, row: Dict) -> Dict:
        # row["groundtruth"] = row["answer"].split("####")[-1].strip()
        row["groundtruth"] = 0
        return row

    def create_prompt_fields(self, row: Dict) -> Dict:
        # row["prompt"] = (
        #     row["question"] + "\nMake sure the final answer is standalone and in latex format."
        # )
        row["prompt"] = 0
        return row

    def _calc_accuracy(self, row: Dict, approach: str) -> Dict:

        # model_output: str = extract_answer(row[f"output_{approach}"], "aime")
        # groundtruth: str = str(row["groundtruth"])

        # row[f"accuracy_{approach}"] = math_equal(model_output, groundtruth)
        row["accuracy"] = 0
        return row


if __name__ == "__main__":

    dataset = Data_set.create_dataset(
        cache_root="/data0/hujunhao/temp",
        dataset_name="mooncake",
        model_name="Llama-3.1-8B-Instruct",
        approach_name="rr",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
    )
    import pdb

    pdb.set_trace()
