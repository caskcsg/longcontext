import os
import json
import argparse
import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from openai import OpenAI
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import random
import pdb

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
    def __init__(self, output_dir=None, log_file=None):
        """Initialize model"""
        self.model_pool = DebateModelPool1()

        self.output_dir = output_dir
        self.log_file = log_file
        # Use ThreadPoolExecutor to increase concurrency
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Create model pool for higher concurrency
        self.model_instances = [DebateModelPool1() for _ in range(10)]
        
        # Create subdirectories
        if output_dir:
            self.topic_dir = os.path.join(output_dir, "topic_generate")
            self.critique_dir = os.path.join(output_dir, "critique_generate")
            self.judge_dir = os.path.join(output_dir, "judge_generate")
            self.result_dir = os.path.join(output_dir, "result")
            
            os.makedirs(self.topic_dir, exist_ok=True)
            os.makedirs(self.critique_dir, exist_ok=True)
            os.makedirs(self.judge_dir, exist_ok=True)
            os.makedirs(self.result_dir, exist_ok=True)

    
    def _write_json_to_file(self, filename, data):
        """Helper function: write JSON data to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def extract_json_from_text(self, text, match_char='[]'):
        """Extract JSON part from text"""
        try:
            if match_char == '[]':
                # Try to find the start and end of the JSON array
                json_start = text.find('[') + 1
                json_end = text.rfind(']')
            else:
                json_start = -1
                json_end = -1
            
            if json_start == -1 or json_end == -1:  # If array format not found, try object format
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                
            if json_start != -1 and json_end != -1:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            # If parsing fails, try a more lenient extraction
            import re
            json_pattern = r'\[\s*\{.*?\}\s*\]'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            
            # Last resort: manual extraction
            topics = []
            # Update regex to match different field names
            topic_pattern = r'"(?:topic|title)":\s*"([^"]*)"'
            expl_pattern = r'"(?:explanation|reason)":\s*"([^"]*)"'
            
            topics_found = re.findall(topic_pattern, text)
            explanations = re.findall(expl_pattern, text)
            
            for i in range(min(len(topics_found), len(explanations))):
                topic_dict = {
                    "topic": topics_found[i],
                    "explanation": explanations[i]
                }
                
                topics.append(topic_dict)
            
            return topics
    
    def critique_topics_batch(self, primary_category, secondary_category, existing_topics, generated_topics, model="debate1"):  
        prompt = f"""You are an expert in semantic analysis, topic modeling, and educational content classification.

You are given a list of subtopics generated by another model for the following BISAC topic:

Top Category: {primary_category}  
Subcategory: {secondary_category}

Each subtopic contains:
- Topic Name: name
- Topic Explanation: generate explanation
"""     
        for i, topic in enumerate(generated_topics):
            prompt += f"""
generate topic {i+1}:
Topic Name: {topic['topic']}
Topic Explanation: {topic['explanation']}
"""
        # pdb.set_trace()
        prompt +="""
Your task is to critically evaluate each proposed subtopic for its quality, distinctiveness, and strategic usefulness in semantic clustering and topic seeding for educational datasets. Assess each subtopic using the following rigorous criteria:

1. Topical Relevance — Is the subtopic clearly and directly aligned with its assigned primary or secondary educational category?

2. Semantic Distinctiveness — Does it introduce a meaningfully different angle (e.g., audience, concept, application, cultural lens), or is it redundant with existing entries?

3. Diversity & Complementarity — Does this subtopic expand the conceptual space covered by the full set, providing non-overlapping, complementary perspectives?

4. Anchor Value for Clustering — Can this subtopic act as a strong semantic node or seed in downstream clustering tasks for large-scale educational content?

Reject vague, overly broad, redundant, or weakly anchored subtopics. Prioritize those that enhance overall topical diversity, fill semantic gaps, and support meaningful partitioning in topic models. Be selective and analytical—do not default to acceptance.
---

### Output Format (Strict JSON)

Return your critique in the following JSON structure:

```
[
    {
    "accepted": [
        {
        "topic": "generate name",
        "explanation": "generate explanation",
        "reason": "accept reason"
        },
        ...
    ],
    "rejected": [
        {
        "topic": "generate explanation",
        "explanation": "generate explanation",
        "reason": "reject reason"
        },
        ...
    ]
    }
]
"""

        model_instance = random.choice(self.model_instances)

            
        for attempt in range(5):
            try:
                completion = model_instance.call(prompt)
                if not completion:
                    continue
                import pdb
                # pdb.set_trace()
                json_data = self.extract_json_from_text(completion)
                if json_data:
                    if "accepted" not in json_data and "rejected" not in json_data:
                        return None
                    else:
                        return json_data
                
                print(f"Attempt {attempt+1}/5: Failed to parse critique JSON")
            except Exception as e:
                print(f"Attempt {attempt+1}/5 failed to critique topics: {e}")
                # time.sleep(1)
        
        print("All critique attempts failed, returning default critique")
        return None

    
    def critique_topics_for_category(self, primary_category, secondary_category, not_value, model='debate2'):
        """Stage 2: Critique topics"""
        print(f"Critique topics stage: {primary_category} - {secondary_category}")
        key = f"{primary_category}_{secondary_category}".replace("/", "_")
        
        if model == 'debate1':
            for_critique = 'debate2'
        else:
            for_critique = 'debate1'
        # Check if topic file exists
        debate1_topics_file = os.path.join(self.topic_dir, f"{key}_{for_critique}_topics.json")
        debate2_topics_file = os.path.join(self.topic_dir, f"{key}_{model}_topics.json")
        debate2_critiques_file = os.path.join(self.critique_dir, f"{key}_{model}_critiques.json")
        
        # If both critique files exist, return directly
        if os.path.exists(debate2_critiques_file):
            print(f"Critique already generated, skipping critique stage")
            return True
        
        # Load topic file
        try:
            debate1_topics = self._read_json_from_file(debate1_topics_file)
        except Exception as e:
            print(f"Error loading topic file: {e}")
            return False
        
        # Critique topics, only process if critique file does not exist
        try:

            if not os.path.exists(debate2_critiques_file):
                debate2_critiques = self.critique_topics_batch(
                    primary_category, secondary_category, not_value, debate1_topics, model
                )
                if debate2_critiques is not None:
                    self._write_json_to_file(debate2_critiques_file, debate2_critiques)
            
            return True
        except Exception as e:
            print(f"Error during topic critique: {e}")
            return False
    def process_category_pair(self, primary_category, secondary_category, not_value, num_to_generate=5, model='debate1'):
        """Process a category pair in three stages"""
        print(f"Processing category: {primary_category} - {secondary_category}")
        # Stage 2: Critique topics
        self.critique_topics_for_category(primary_category, secondary_category, not_value, model)
    
    def _read_json_from_file(self, filename):
        """Helper function: Read JSON data from file"""
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)

def process_single_category(args):
    """Single category processing function for multiprocessing"""
    import pdb
    # pdb.set_trace()
    category_data, output_dir, num_to_generate, model = args
    primary_category = category_data["primary_category"]
    secondary_category = category_data["secondary_category"]
    existing_topics = category_data["existing_topics"]
    not_value = category_data["not_value"]
    number = category_data["number"]
    key = f"{primary_category}_{secondary_category}".replace("/", "_")
    result_dir = os.path.join(output_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(result_dir, f"{key}.json")
    
    # Use file lock to check if file exists
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                # Try to acquire shared lock to check file
                fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
                print(f"Category {primary_category} - {secondary_category} already processed, skipping")
                return None
        except IOError:
            # If unable to acquire lock, file is being processed by another process
            print(f"Category {primary_category} - {secondary_category} is being processed by another process, skipping")
            return None
    
    try:
        generator = TopicDebateGenerator(output_dir=output_dir)

        generator.process_category_pair(
            primary_category,
            secondary_category,
            not_value,
            number,
            model,
        )
        
        return 1
    except Exception as e:
        print(f"Error processing category {primary_category} - {secondary_category}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Topic Debate Generator")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_to_generate", type=int, default=10, help="Number of topics to generate for each category")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes, default is half of CPU cores")
    parser.add_argument("--model", type=str, default="gpt", help="Model to use for each batch of categories")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset to get topics
    from datasets import load_dataset
    import pandas as pd
    
    ds = load_dataset("HuggingFaceTB/bisac_expanded_final")
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
    random.shuffle(process_args)  # Shuffle to avoid long processing for some categories
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

