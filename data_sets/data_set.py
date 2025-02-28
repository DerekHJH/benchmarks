import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List
from transformers import AutoTokenizer
import pandas as pd
from datasets import load_dataset
from benchmarks.data_sets.utils import f1_score

logger = logging.getLogger(__name__)


class Data_set(ABC):
    """
    Abstract class for datasets. All datasets should inherit from this class.

    We use the underline to differentiate from huggingface
    datasets (dataset) and Datasets (Dataset).
    """

    def __init__(self, 
                cache_root: str,
                dataset_name: str,
                model_name: str,
                approach_name: str,
                tokenizer_name: str, 
                num_data_preprocess=int(1e6), 
                num_data_test=int(1e6),
                **kwargs
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
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_data_preprocess = num_data_preprocess
        self.num_data_test = num_data_test
        assert self.num_data_test <= self.num_data_preprocess, "num_data_test should be less than or equal to num_data_preprocess"
        self.kwargs = kwargs

        # Mutable attributes
        self.data: pd.DataFrame = None # Hold the preprocessed data
        preprocessed_data_path = os.path.join(self.cache_folder, "preprocessed_data.json")
        if os.path.exists(preprocessed_data_path):
            self.data = pd.read_json(preprocessed_data_path)
            logger.info(f"Loaded dataset locally from {preprocessed_data_path}")
        else:
            self.data = self.load_data_online() # Implemented in the subclass
            logger.info("Loaded dataset online")
            
            # Common processing for all datasets
            self.data = self.data[:self.num_data_preprocess]
            self.data = self.data.apply(
                lambda row: self.create_groundtruth_field(row), axis=1
            ).apply(lambda row: self.create_prompt_field(row), axis=1)

            
            self.data.to_json(preprocessed_data_path, orient="records", indent=4)
        self.data = self.data[:num_data_test]

        self.result: pd.DataFrame = None # Hold the result of the test using (dataset_name, model_name, approach_name, tokenizer_name)
        result_path = os.path.join(self.cache_folder, f"{approach_name}.json")
        if os.path.exists(result_path):
            self.result = pd.read_json(result_path)
        else:
            self.result = pd.DataFrame()
        logger.info(f"The progress is {len(self.result)}/{num_data_test}")


    @staticmethod
    def create_dataset(
        cache_root: str,
        dataset_name: str,
        model_name: str,
        approach_name: str,
        tokenizer_name: str,
        num_data_preprocess=int(1e6), 
        num_data_test=int(1e6),
        **kwargs
    ):
        if dataset_name == "sharegpt":
            Class = ShareGPT
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
            **kwargs
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
    def create_groundtruth_field(self, row: Dict) -> Dict:
        """
        Create the row["groundtruth"] column in the dataset. To be used with the pandas apply function.

        Args:
            row: One row of the dataset.

        Returns:
            The row with the row["groundtruth"] updated.
        """
        raise NotImplementedError

    @abstractmethod
    def create_prompt_field(self, row: Dict) -> Dict:
        """
        Create the row["prompt"] column in the dataset. To be used with the pandas apply function.

        Args:
            row: One row of the dataset.

        Returns:
            The row with the row["prompt"] updated.
        """
        raise NotImplementedError

    @abstractmethod
    def _calc_accuracy(self, row: Dict, approach: str) -> Dict:
        """
        Check the correctness of the model output and store the result in the f'accuracy_{approach}' column.
        And store the final output in the f'final_output_{approach}' column.

        Args:
            row: One row of the dataset.
            approach: The approach used to generate the output.

        Returns:
            The row with the row[f'accuracy_{approach}'], row[f'final_output_{approach}'] updated.
        """
        raise NotImplementedError


    """
    The following methods are common to all subclasses and SHOULD NOT BE OVERRIDDEN.
    This constraint is to ensure consistent interfaces.
    If you have a strong need to override some of the following methods,
    please refer to how calc_accruacy and _calc_accuracy methods are implemented.
    """

    def calc_accuracy(self, approach: str) -> None:
        """
        Compare the model output with the groundtruth and calculate the accuracy.
        Store the accuracy in the "accuracy" column.

        Args:
            approach: The approach used to generate the output.
        """
        result_path = os.path.join(self.cache_folder, f"{approach}.json")
        assert os.path.exists(result_path), f"{result_path} not found"
        self.result = pd.read_json(result_path)
        assert "output" in result.columns, f"output not in the result"
        assert "groundtruth" in result.columns, "groundtruth not in the result"

        result = result.apply(lambda row: self._calc_accuracy(row, approach), axis=1)
        result.to_json(result_path, orient="records", indent=4)

    def update(self, new_data: Dict[str, List]) -> None:
        """
        Update the dataset with new dictionary data. If the length of the new data is less than the original data,
        fill the rest with None.

        Args:
            new_data: The new dictionary data to update the dataset.
        """
        for key, value in new_data.items():
            if len(value) < len(self.data):
                logger.warning(
                    (
                        "Length of new data is less than the original data: "
                        f"{len(value)} < {len(self.data)}."
                        "We fill the rest with None."
                    )
                )
            self.data[key] = value + (len(self.data) - len(value)) * [None]
        

    def __iter__(self):
        for idx, row in self.data.iterrows():
            yield idx, row["prompt"], row["groundtruth"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]


class ShareGPT(Data_set):

    def load_data_online(self) -> pd.DataFrame:
        # return load_dataset("RyokoAI/ShareGPT52K", split="test").to_pandas()
        return pd.read_json("/data0/hujunhao/.cache/huggingface/hub/datasets--RyokoAI--ShareGPT52K/snapshots/6f9b78cc1dd15dbb51d3c51ccc219c558962fd77/sg_90k_part1.json")

    def create_groundtruth_field(self, row: Dict) -> Dict:
        row["groundtruth"] = row["answer"].split("####")[-1].strip()
        return row

    def create_prompt_field(self, row: Dict) -> Dict:
        # TODO(hjh): Create more complex prompts
        row["prompt"] = (
            row["question"] + "\nMake sure the final answer is standalone and in latex format."
        )
        return row

    def _calc_accuracy(self, row: Dict, approach: str) -> Dict:

        model_output: str = extract_answer(row[f"output_{approach}"], "aime")
        groundtruth: str = str(row["groundtruth"])

        row[f"accuracy_{approach}"] = math_equal(model_output, groundtruth)
        row[f"final_output_{approach}"] = model_output
        # row[f"accuracy_{approach}"] = rouge_score(model_output, groundtruth)
        return row


if __name__ == "__main__":

    dataset = Data_set.create_dataset(
        cache_root="/data0/hujunhao/benchmarks",
        dataset_name="sharegpt",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        approach_name="raas",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        num_data_preprocess=10,
        num_data_test=10,
    )

    import pdb
    pdb.set_trace()
