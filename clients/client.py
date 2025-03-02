import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import aiohttp
import pandas as pd
from transformers import AutoTokenizer

from benchmarks.data_sets.data_set import Data_set

logger = logging.getLogger(__name__)


class Client(ABC):
    """
    Abstract class for clients. All clients should inherit from this class.

    The client
    1) Sends all requests from dataset in a certain request rate.
    2) Gets all responses and stores them into dataset.
    3) Save per-request information such as JCT, TTFT, tec. into dataset.

    Warning: The client is asynchronous and single-threaded.
    And that is why we go wild in using global variables.
    """

    def __init__(
        self,
        dataset: Data_set,
        host: str = "localhost",
        port: int = 8000,
        request_rate: float = 20,
        **kwargs
    ) -> None:
        """
        Initialize the client.

        Args:
            dataset: The dataset used for the client.
            host: The host of the server.
            port: The port of the server.
            request_rate: The rate of requests per second.
        Return:
            None
        """

        # Imutable attributes
        self.host = host
        self.port = port
        self.request_rate = request_rate

        # Mutable attributes
        self.dataset = dataset
        self.start_time = None  # The start time of the client run function

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

        return Class(dataset=dataset, host=host, port=port, request_rate=request_rate, **kwargs)

    @abstractmethod
    async def run(self) -> None:
        """
        Run the client with asyncio.run(client.run()).
        1) Sends all requests from dataset in a certain request rate.
        2) Gets all responses and stores them into dataset.
        3) Save per-request information such as JCT, TTFT, etc. into self.dataset.

        Returns:
            None
        """
        raise NotImplementedError


class OpenLoopClient(Client):

    async def run(self) -> None:

        self.start_time = time.perf_counter()
        coroutines = [
            asyncio.create_task(
                self.send_single_request(request_id, prompt_token_ids, ground_truth_token_ids)
            )
            for request_id, prompt_token_ids, ground_truth_token_ids in self.dataset
        ]
        await asyncio.gather(*coroutines)

    async def send_single_request(
        request_id: int, prompt_token_ids: List[int], ground_truth_token_ids: List[int]
    ):
        pass


class ClosedLoopClient(Client):

    async def run(self) -> None:
        raise NotImplementedError


if __name__ == "__main__":

    dataset = Data_set.create_dataset(
        cache_root="/data0/hujunhao/temp",
        dataset_name="mooncake",
        model_name="Llama-3.1-8B-Instruct",
        approach_name="rr",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        num_data_test=100,
    )
    client = Client.create_client(
        dataset=dataset,
        host="localhost",
        port=8000,
        request_rate=20,
        client_type="open",
    )
    asyncio.run(client.run())


# AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
# G_URL = "http://127.0.0.1:8081/add_request"  #GS服务器的地址 P

# async def asyc_forward_request(request_dict, api_url):
#     headers = {"User-Agent": "Test Client"}
#     async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
#         async with session.post(url=api_url, json=request_dict,
#                                 headers=headers) as response:
#             if response.status == 200:
#                 delimiter=b"\0"
#                 buffer = b''  # 用于缓存数据块中的部分消息
#                 async for chunk in response.content.iter_any():
#                     buffer += chunk  # 将新的数据块添加到缓冲区中
#                     while delimiter in buffer:
#                         index = buffer.index(delimiter)  # 查找分隔符在缓冲区中的位置
#                         message = buffer[:index]  # 提取从缓冲区起始位置到分隔符位置的消息
#                         yield message.strip()  # 返回提取的消息
#                         buffer = buffer[index + len(delimiter):]  # 从缓冲区中移除已提取的消息和分隔符

# async def post_request_and_get_response(args, req, waiting_time):

#     if args.test_type == "open":
#         await asyncio.sleep(waiting_time)

#     pload = {
#         "prompt_token_ids": req[1],
#         "request_id": random_uuid(),
#         "n": args.n,
#         "use_beam_search": False,
#         "temperature": 0.0,
#         "max_tokens": req[-1],
#         "logprobs": 1,
#         "ignore_eos": True,
#         "stream":True
#     }

#     response = asyc_forward_request(pload, G_URL)
#     start_time = 0
#     end_time = 0
#     ttft = 0
#     tbt = []
#     completion_token_ids = []
#     async for resp in response:
#         resp = resp.decode('utf-8')
#         resp = json.loads(resp)
#         if resp['n'] == 0:
#             start_time = resp['start_time']
#             ttft = resp['ttft']
#             if resp['finished'] == True:
#                 end_time = resp['end_time']
#                 completion_token_ids.extend(resp['prefilled_token_id'])
#         else:
#             if resp['finished'] != True:
#                 tbt.append(resp['tbt'])
#             elif resp['finished'] == True:
#                 end_time = resp['end_time']
#                 completion_token_ids.extend(resp['prefilled_token_id'])

#     # print('completion_token_ids', completion_token_ids)
#     # print('completion_token_ids length', len(completion_token_ids))
#     # print('ground truth length', req[-1])
#     assert len(completion_token_ids) == req[-1], 'Fail to keep the length of completion token ids'
#         # yield (json.dumps(resp, ensure_ascii=False) + "\0").encode("utf-8")
#     return (end_time-start_time, ttft, tbt[1:], tbt, tbt[0], req[-2] , req[-1], completion_token_ids)


# async def dummy_post_request_and_get_response(args, req, waiting_time, **kwargs):

#     if args.test_type == "open":
#         await asyncio.sleep(waiting_time)

#     main_request_id = kwargs['main_request_id']
#     sub_request_id = kwargs['sub_request_id']
#     print(f"main_request_id {main_request_id} sub_request_id {sub_request_id}")
#     await asyncio.sleep(3) # Do some work here

#     return (0.1, 1.1, [1.1, 2.2], [3.3, 4.4], 1.1, req[-2], req[-1], list(range(req[-1])))


# # Handle all subrequests of one main request
# async def handle_main_request(main_request_id, reqs, args):
#     global waiting_time
#     global time_start
#     global response
#     res = None
#     for i in range(len(reqs)):
#         if i != 0:
#             prev_prompt_len = reqs[i-1][-2]
#             prev_completion_len = reqs[i-1][-1]
#             prev_completion_token_ids = res[-1]
#             reqs[i][1][prev_prompt_len:prev_prompt_len+prev_completion_len] = prev_completion_token_ids
#         waiting_time = waiting_time + np.random.exponential(1.0 / args.request_rate)
#         time_elapsed = time.perf_counter() - time_start
#         if waiting_time < time_elapsed:
#             print(f"\033[93m Warning: main_request_id {main_request_id} sub_request_id {i}: Poisson violation\033[0m", file=sys.stderr)
#             print(f"\033[93m Should have been sent at {time_elapsed - waiting_time:.3} seconds ago\033[0m", file=sys.stderr)
#         res = await post_request_and_get_response(args, reqs[i], waiting_time - time_elapsed)
#         # res = await dummy_post_request_and_get_response(
#         #     args,
#         #     reqs[i],
#         #     waiting_time - time_elapsed,
#         #     main_request_id=main_request_id,
#         #     sub_request_id=i
#         # )
#         response.append(res)

# async def run(args, reqs, multi_conversations_range):

#     coroutines = [
#         asyncio.create_task(handle_main_request(
#         i, reqs[multi_conversations_range[i]:multi_conversations_range[i+1]], args))
#         for i in range(first_few_sessions)
#     ]

#     # Start global timer
#     time_start = time.perf_counter()
#     # Kick start the first few sessions
#     await asyncio.sleep(0)

#     # Deal with the rest of the sessions, add them when the first few sessions cannot meet the Poisson speed
#     main_request_id = first_few_sessions
#     while main_request_id < len(multi_conversations_range) - 1:
#         coroutines.append(asyncio.create_task(handle_main_request(
#             main_request_id,
#             reqs[multi_conversations_range[main_request_id]:multi_conversations_range[main_request_id+1]],
#             args
#         )))
#         main_request_id += 1
#         # Sleep for enough time to avoid too many sessions to enter at the same time
#         # To avoid oversleep, we subtract 0.1 seconds
#         await asyncio.sleep(waiting_time - (time.perf_counter() - time_start) - 0.1)
#         while waiting_time - (time.perf_counter() - time_start) - 0.1 > 0:
#             await asyncio.sleep(waiting_time - (time.perf_counter() - time_start) - 0.1)

#     await asyncio.gather(*coroutines)

#     global response
#     assert len(response) == len(reqs), 'Fail to handle all requests'
#     for res in response:
#         jct.append(res[0])
#         ttft.append(res[1])
#         tbt_no_second_token.extend(res[2])
#         tbt_with_second_token.extend(res[3])
#         second_token.append(res[4])
#         # print("Res ", res)
#     print("average_jct,p99_jct,average_ttft,p99_ttft,average_tbt_no_second_token,p99_tbt_no_second_token,average_tbt_with_second_token,p99_tbt_with_second_token")
#     print("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(np.average(jct), np.percentile(jct, 99), np.average(ttft), np.percentile(ttft, 99), np.average(tbt_no_second_token), np.percentile(tbt_no_second_token, 99), np.average(tbt_with_second_token), np.percentile(tbt_with_second_token, 99)))s
