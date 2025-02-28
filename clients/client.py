import logging
import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List
from transformers import AutoTokenizer
import pandas as pd
from benchmarks.data_sets.data_set import Data_set

logger = logging.getLogger(__name__)


class Client(ABC):
    """
    Abstract class for clients. All clients should inherit from this class.

    The client 
    1) Sends all requests from dataset in a certain request rate.
    2) Gets all responses and stores them into dataset.
    """

    def __init__(self, 
                dataset: Data_set,
                host: str = "localhost",
                port: int = 8000,
                request_rate: float = 20,
                num_data_test: int = int(1e6),
                **kwargs
            ) -> None:
        """
        Initialize the client.

        Args:
            dataset: The dataset used for the client.
            host: The host of the server.
            port: The port of the server.
            request_rate: The rate of requests per second.
            num_data_test: The number of data/requests to test.
        Return:
            None
        """

        # Imutable attributes
        self.host = host
        self.port = port
        self.request_rate = request_rate
        self.num_data_test = num_data_test

        # Mutable attributes
        self.dataset = dataset


    @staticmethod
    def create_dataset(
        dataset: Data_set,
        host: str = "localhost",
        port: int = 8000,
        request_rate: float = 20,
        num_data_test: int = int(1e6),
        client_type: str = "open",
        **kwargs
    ):
        if client_type == "open":
            Class = OpenLoopClient
        elif client_type == "closed":
            Class = ClosedLoopClient
        else:
            raise NotImplementedError

        return Class(
            dataset=dataset,
            host=host,
            port=port,
            request_rate=request_rate,
            num_data_test=num_data_test,
            **kwargs
        )
    
    @abstractmethod
    def run(self) -> None:
        """
        Run the client. 
        1) Sends all requests from dataset in a certain request rate.
        2) Gets all responses and stores them into dataset.

        Returns:
            None
        """
        raise NotImplementedError




class OpenLoopClient(Client):

    def run(self) -> None:
        raise NotImplementedError






class ClosedLoopClient(Client):

    def run(self) -> None:
        raise NotImplementedError


if __name__ == "__main__":

    dataset = Data_set.create_dataset(
        cache_root="/data0/hujunhao/temp",
        dataset_name="mooncake",
        model_name="Llama-3.1-8B-Instruct",
        approach_name="raas",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        # num_data_preprocess=10,
        # num_data_test=10,
    )
    client = Client.create_client(
        dataset=dataset,
        host="localhost",
        port=8000,
        request_rate=20,
        num_data_test=int(1e6),
        client_type="open",
    )
    client.run()
