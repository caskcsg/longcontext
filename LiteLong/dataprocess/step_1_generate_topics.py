import os
import json
import argparse
import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from openai import OpenAI
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import random

class DebateModelPool1:
    def __init__(self):
        self.call_times = 0
        self.models = []
        self.models.append(self.call_model1)
    def call(self, messages, temperature=0.6, top_p=0.95, max_tokens=8192):
        return self.models[0](messages, temperature, top_p, max_tokens)
    def call_model1(self, messages, temperature=0.6, top_p=0.95, max_tokens=8192):
        client = OpenAI(api_key="demo_api_1", base_url="http://localhost:8001/v1")
        try:
            completion = client.chat.completions.create(
                model="Mistral-8*7B-Instruct",
                stream=False,
                messages=[{"role": "user", "content": messages}],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error occurred when calling model Mistral-8*7B-Instruct: {e}")
            return None

class DebateModelPool2:
    def __init__(self):
        self.call_times = 0
        self.models = []
        self.models.append(self.call_model1)
    def call(self, messages, temperature=0.6, top_p=0.95, max_tokens=8192):
        return self.models[0](messages, temperature, top_p, max_tokens)
    def call_model1(self, messages, temperature=0.6, top_p=0.95, max_tokens=8192):
        client = OpenAI(api_key="demo_api_1", base_url="http://localhost:8000/v1")
        try:
            completion = client.chat.completions.create(
                model="Qwen2.5-7B-Instruct",
                stream=False,
                messages=[{"role": "user", "content": messages}],
                temperature=temperature,
                top_p=top_p,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error occurred when calling model Qwen2.5-7B-Instruct: {e}")
            return None

class TopicDebateGenerator:
    def __init__(self, output_dir=None, log_file=None, model='debate1'):
        if model == 'debate1':
            self.model_pool = DebateModelPool1()
        elif model == 'debate2':
            self.model_pool = DebateModelPool2()
        self.output_dir = output_dir
        self.log_file = log_file
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        if output_dir:
            self.topic_dir = os.path.join(output_dir, "topic_generate")
            os.makedirs(self.topic_dir, exist_ok=True)
    
    def generate_topics(self, primary_category, secondary_category, not_value, num_to_generate=5, model='debate2'):
        prompt_template = f"In the field of {primary_category} list {num_to_generate * 2} subtopics in {secondary_category} and provide a brief explanation of each."
        if not_value:
            prompt_template += f"Make sure to avoid topics that could fall under any of these other categories:{not_value}"
        prompt_template += """
Return the output strictly as a JSON array of objects, using the following format:
[
  {{
    "topic": "...",
    "explanation": "...",
  }},
  {{
    "topic": "...",
    "explanation": "...",
  }}
  ...
]
"""
        key = f"{primary_category}_{secondary_category}".replace("/", "_")
        topics_file = os.path.join(self.topic_dir, f"{key}_{model}_topics.json")
        topics = None
        
        if not os.path.exists(topics_file):
            response = self.model_pool.call(prompt_template)
            try:
                topics = self.extract_json_from_text(response)
                if len(topics) > 2:
                    self._write_json_to_file(topics_file, topics)
            except Exception as e:
                print(f"Error parsing JSON from model {model}: {e}")
                print(f"{model} response: {response}")
        
        if not topics and os.path.exists(topics_file):
            topics = self._read_json_from_file(topics_file)
        
        return topics
    
    def _write_json_to_file(self, filename, data):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def extract_json_from_text(self, text, match_char='[]'):
        try:
            if match_char == '[]':
                json_start = text.find('[')
                json_end = text.rfind(']') + 1
            else:
                json_start = -1
                json_end = -1
            
            if json_start == -1 or json_end == -1:
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                
            if json_start != -1 and json_end != -1:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            import re
            json_pattern = r'\[\s*\{.*?\}\s*\]'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            
            topics = []
            topic_pattern = r'"(?:topic|title)":\s*"([^"]*)"'
            expl_pattern = r'"(?:explanation|reason)":\s*"([^"]*)"'
            
            topics_found = re.findall(topic_pattern, text)
            explanations = re.findall(expl_pattern, text)
            
            for i in range(min(len(topics_found), len(explanations))):
                topics.append({
                    "topic": topics_found[i],
                    "explanation": explanations[i]
                })
            
            return topics
    
    def generate_topics_for_category(self, primary_category, secondary_category, not_value, num_to_generate=5, model='debate2'):
        key = f"{primary_category}_{secondary_category}".replace("/", "_")
        topics_file = os.path.join(self.topic_dir, f"{key}_{model}_topics.json")
        
        if os.path.exists(topics_file):
            print(f"Topics already generated, skipping generation phase")
            return True
        
        try:
            self.generate_topics(primary_category, secondary_category, not_value, num_to_generate, model)
            return True
        except Exception as e:
            print(f"Error occurred while generating topics: {e}")
            return False
    
    def process_category_pair(self, primary_category, secondary_category, not_value, num_to_generate=5, model='debate2'):
        self.generate_topics_for_category(primary_category, secondary_category, not_value, num_to_generate, model)
    
    def _read_json_from_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)

def process_single_category(args):
    category_data, output_dir, num_to_generate, model = args
    primary_category = category_data["primary_category"]
    secondary_category = category_data["secondary_category"]
    not_value = category_data["not_value"]
    number = category_data["number"]
    key = f"{primary_category}_{secondary_category}".replace("/", "_")
    result_dir = os.path.join(output_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(result_dir, f"{key}.json")
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
                return None
        except IOError:
            return None
    
    try:
        
        generator = TopicDebateGenerator(output_dir=output_dir, model=model)
        generator.process_category_pair(
            primary_category,
            secondary_category,
            not_value,
            number,
            model,
        )
        return 1
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description="Topic Debate Generator")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_to_generate", type=int, default=10, help="Number of topics to generate for each category")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes, default is half of CPU cores")
    parser.add_argument("--model", type=str, default='claude', help="Model to use for each batch of categories")
    import pandas as pd
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/bisac_expanded_final")['train']
    df = pd.DataFrame(ds['train'])
    import pdb
    # pdb.set_trace()
    # Prepare category data
    categories_data = []
    import numpy as np
    for primary_category in set(df['top_category']):
        primary_df = df[df['top_category'] == primary_category]
        for secondary_category in set(primary_df['subcategory']):
            secondary_df = primary_df[primary_df['subcategory'] == secondary_category]
            existing_topics = list(secondary_df['subtopic'])
            not_value = secondary_df.iloc[0]['not']
            if not_value is np.nan:
                # pdb.set_trace()
                not_value = ""
            number = secondary_df.iloc[0]['expansion_factor']
            # try:
            categories_data.append({
                "primary_category": primary_category,
                "secondary_category": secondary_category,
                "existing_topics": existing_topics,
                "not_value": not_value if len(not_value) > 2 else None,
                "number": number
            })
    
    # Check already generated files
    existing_files = set(os.listdir(args.output_dir))
    
    # Filter out already processed categories
    categories_to_process = []
    for item in categories_data:
        key = f"{item['primary_category']}_{item['secondary_category']}".replace("/", "_") + ".json"
        if key not in existing_files:
            categories_to_process.append(item)
    
    print(f"Total {len(categories_data)} categories, {len(categories_data) - len(categories_to_process)} processed, {len(categories_to_process)} remaining to process")
    
    if not categories_to_process:
        print("All categories have been processed")
        return
    
    # Set number of processes
    if args.processes is None:
        processes = max(1, cpu_count() // 2)  # Default to half of CPU cores
    else:
        processes = args.processes
    
    print(f"Using {processes} processes for processing")
    
    # Prepare multiprocessing arguments
    process_args = [(item, args.output_dir, args.num_to_generate, args.model) for item in categories_to_process]
    import random
    random.shuffle(process_args)
    # Test single processing
    # process_single_category(process_args[0])
    # return
    
    # Use process pool to process multiple categories in parallel
    results = {}
    with Pool(processes=processes) as pool:
        for result in tqdm(pool.imap_unordered(process_single_category, process_args), 
                            total=len(process_args), 
                            desc="Processing categories"):
            ...

if __name__ == "__main__":
    main()
