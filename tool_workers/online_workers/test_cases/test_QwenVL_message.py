"""Send a test message."""
import argparse
import json
import time
import requests
import torch
import uuid
import os
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *


def main():
    model_name = args.model_name

    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        # ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        print(f"{model_name} worker_addr: {worker_addr}")

    if worker_addr == "":
        print(f"No available workers for {model_name}")
        return

    headers = {"User-Agent": "FastChat Client"}
    if args.send_image:
        img = load_image(args.image_path)
        img_arg = encode(img)
    else:
        img_arg = args.image_path
    datas = {
        "model": model_name,
        "query":args.query,
        "image": img_arg,
    }
    tic = time.time()
    # breakpoint()
    response = requests.post(
        worker_addr + "/worker_generate",
        headers=headers,
        json=datas,
    )
    toc = time.time()
    print(f"Time: {toc - tic:.3f}s")
    print(f"QwenVL response: {response[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # worker parameters
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:20001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default='QwenVL')

    # model parameters
    parser.add_argument(
        "--max_new_tokens", type=str, default=128
    )
    parser.add_argument(
        "--query", type=str, default="describe this image shortly."
    )
    parser.add_argument(
        "--send_image", action="store_true",
    )
    parser.add_argument(
        "--image_path", type=str, default="/home/fangly/mllm/ljd/OpenThinkIMG/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/mathvista_35.jpg"
    )
    args = parser.parse_args()
    args.send_image = False

    main()
