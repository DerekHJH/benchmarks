from benchmarks.utils import set_seed
from benchmarks.data_sets.data_set import Data_set
from benchmarks.clients.client import Client
import asyncio
from transformers import AutoTokenizer
import argparse
import sys


def get_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache-root", type=str, default="/data/hujunhao/FlowServe/datasets")
    parser.add_argument("--dataset-name", type=str, default="mooncake", choices=["sharegpt", "mooncake"])
    parser.add_argument("--model-name", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--approach-name", type=str, default="rr", choices=["rr", "ll"]) # Round Robin (rr), Least Load (ll)
    parser.add_argument("--tokenizer-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num-data-preprocess", type=int, default=int(1e6))
    parser.add_argument("--num-data-test", type=int, default=int(1e6)) # num_requests

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    # parser.add_argument("--stream", action="store_true")
    parser.add_argument("--request-rate", type=float, default=20)
    parser.add_argument("--client-type", type=str, default="open", choices=["open", "closed"])

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return args
    
if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)
    dataset = Data_set.create_dataset(
        **vars(args)
    )
    client = Client.create_client(
        **vars(args)
    )

    
    asyncio.run(run(args, reqs, multi_conversations_range))