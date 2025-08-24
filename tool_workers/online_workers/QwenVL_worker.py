import uuid
import os
import re
import io
import argparse
import torch
import numpy as np
from PIL import Image

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
import matplotlib.pyplot as plt

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
import traceback

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None


class QwenVLWorker(BaseToolWorker):
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_path = "/home/fangly/models/Qwen2.5-VL-7B-Instruct", 
                 model_base = "", 
                 model_name = "QwenVL",
                 load_8bit = False, 
                 load_4bit = False, 
                 device = "cuda",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 model_semaphore = None,
                 max_new_tokens = 128,
                 args = None,
                 ):
        self.max_new_tokens = max_new_tokens  # 新增特有参数
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            args=args
            )

    def init_model(self):
        logger.info(f"Initializing model {self.model_name}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
               self.model_path , torch_dtype="auto", device_map=self.device
            )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate(self, params):
        image = params.get("image", None)
        query = params.get("query", None)
        if query is None:
            logger

        messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image ,
                                },
                                {"type": "text", "text": query},
                            ],
                        }
                    ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def get_tool_instruction(self):
        instruction = {
                "name": "QwenVL",
                "description": "A MLLM model",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The path of the image."
                        },
                        "query": {
                            "type": "string",
                            "description": "your query"
                        }
                    },
                    "required": ["query"]
                }
            }
        return instruction 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20077)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--model_path", type=str, default="/home/fangly/models/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = QwenVLWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
        model_path = args.model_path,
        max_new_tokens = args.max_new_tokens
    )
    worker.run()

