#!/usr/bin/env python3
"""
LongBench-Pro Evaluation Tool
"""
import argparse
import os
import sys
import json
import random
import numpy as np
import logging
from typing import Optional, Dict, Any
from modules.data_loader import DataLoader
from modules.model_manager import ModelManagerOpenAI
from modules.inference import InferenceEngine
from modules.evaluation import Evaluator

# configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """parse arguments"""
    parser = argparse.ArgumentParser(description='LongBench-Pro Evaluation Tool')
    
    # basic parameters
    parser.add_argument("--output_dir", type=str, default="output", help="output directory")
    parser.add_argument("--n_proc", type=int, default=8, help="number of processes in one shard")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # dataset parameters
    parser.add_argument("--dataset_path", type=str, default="dataset/longbench_pro.json", help="dataset path")
    parser.add_argument("--total_shards", type=int, default=1, help="number of shards")
    parser.add_argument("--shard_id", type=int, default=1, help="shard id, begin from 1")
    
    # model parameters
    parser.add_argument("--model_manager", type=str, default="openai", help="model manager")
    parser.add_argument("--model_name", type=str, default="Qwen3-235B-A22B-Instruct-2507", help="model name")
    parser.add_argument("--thinking_model", action='store_true', help="this model is a thinking model, which opens thinking mode by default, use non-thinking prompt to evaluate")
    parser.add_argument("--tokenizer_path", type=str, default="model/Tokenizers/qwen", help="tokenizer path")
    parser.add_argument("--context_max_length", type=int, default=224000, help="context max length")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/v1", help="url")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="api key")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--max_new_tokens", type=int, default=32768, help="max new tokens")
    parser.add_argument("--extra_body", type=str, default=None, help="extra body")
    parser.add_argument("--timeout", type=int, default=1200, help="timeout")
    parser.add_argument("--max_tries", type=int, default=1, help="max tries")
    parser.add_argument("--time_sleep", type=float, default=0.0, help="time sleep")

    # inference parameters
    parser.add_argument("--only_infer", action='store_true', help="only infer")
    parser.add_argument("--thinking_enabled", action='store_true', help="thinking enabled")
    
    # evaluation parameters
    parser.add_argument("--only_eval", action='store_true', help="only eval")
    parser.add_argument("--embedding_model_path", type=str, default="model/Qwen3-Embedding-8B", help="embedding model path for summary task")
    parser.add_argument("--bon_num", type=int, default=3, help="Evaluation inference iterations, used to compute general performance (average) and upper-bound performance (best-of-n), default is 3.")

    args = parser.parse_args()
    try:
        args.extra_body = json.loads(args.extra_body) if args.extra_body else None
    except json.JSONDecodeError as e:
        logger.error(f"extra_body is not a valid JSON format - {e}")
        sys.exit(1)
    
    return args

def set_random_seed(seed: int):
    """set all random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def main():
    """main function"""

    args = parse_arguments()
    
    # set random seed
    set_random_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # create output directory
    if args.thinking_enabled or args.thinking_model:
        inference_file = os.path.join(args.output_dir, args.model_name, f"thinking_context-{args.context_max_length}_bon-{args.bon_num}_inference_{args.shard_id}-of-{args.total_shards}.jsonl")
        evaluation_file = os.path.join(args.output_dir, args.model_name, f"thinking_context-{args.context_max_length}_bon-{args.bon_num}_evaluation.jsonl")
        summary_file = os.path.join(args.output_dir, args.model_name, f"thinking_context-{args.context_max_length}_bon-{args.bon_num}_summary.json")
    else:
        inference_file = os.path.join(args.output_dir, args.model_name, f"nonthinking_context-{args.context_max_length}_bon-{args.bon_num}_inference_{args.shard_id}-of-{args.total_shards}.jsonl")
        evaluation_file = os.path.join(args.output_dir, args.model_name, f"nonthinking_context-{args.context_max_length}_bon-{args.bon_num}_evaluation.jsonl")
        summary_file = os.path.join(args.output_dir, args.model_name, f"nonthinking_context-{args.context_max_length}_bon-{args.bon_num}_summary.json")
    # get all inference files in shard group, use for evaluation
    path_prefix = inference_file.rsplit("_", 1)[0]
    inference_files_in_shard_group = [f"{path_prefix}_{shard_id}-of-{args.total_shards}.jsonl" for shard_id in range(1, args.total_shards+1)]
    
    logger.info(f"Inference file this time: {inference_file}")
    logger.info(f"Inference files in shard group: {inference_files_in_shard_group}")
    logger.info(f"Evaluation file: {evaluation_file}")
    logger.info(f"Summary file: {summary_file}")
    os.makedirs(os.path.dirname(inference_file), exist_ok=True)

    # data loader
    data_loader = DataLoader(args.dataset_path, args.bon_num, args.total_shards, args.shard_id, args.seed)
    inference_samples_num = data_loader.inference_samples_num # total inference samples num = bon_num * total question num / total_shards
    evaluation_samples_num = data_loader.evaluation_samples_num # total evaluation samples num = bon_num * total question num

    if not args.only_eval:
        # data processing
        # filter data that needs processing
        new_data = data_loader.filter_new_data(inference_file) # samples that need to be processed, some samples may have been processed before
        # split data for multi-process
        data_subsets = data_loader.split_data_for_multi_process(new_data, args.n_proc)
        
        # model manager
        if args.model_manager == "openai":
            model_manager = ModelManagerOpenAI(
                args.model_name, args.tokenizer_path, args.context_max_length, args.url, args.api_key,
                args.temperature, args.max_new_tokens, args.timeout, args.max_tries, args.time_sleep, args.extra_body
            )
        else:
            # TODO: You can add your own model manager here!
            raise ValueError(f"Unsupported model manager: {args.model_manager}")
        
        # infer
        inference_engine = InferenceEngine(model_manager, args.thinking_enabled)
        inference_engine.infer(data_subsets, inference_file)
    
    if not args.only_infer:
        # evaluate
        evaluation_engine = Evaluator(evaluation_samples_num=evaluation_samples_num, embedding_model_path=args.embedding_model_path)
        fail_items_num = evaluation_engine.evaluate(inference_files_in_shard_group, evaluation_file)

        # summary
        evaluation_engine.metric_summary(evaluation_file, fail_items_num, summary_file, args.bon_num)


if __name__ == "__main__":
    main()
