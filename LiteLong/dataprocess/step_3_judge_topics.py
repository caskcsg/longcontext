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

class JudgeModelPool:
    def __init__(self):
        self.call_times = 0
        self.models = []
        self.models.append(self.call_model1)
    def call(self, messages):
        return self.models[0](messages)
    def call_model1(self, messages, temperature=0.6, top_p=0.95, max_tokens=4096):
        client = OpenAI(api_key="demo_api_1", base_url="http://localhost:8000/v1")
        try:
            completion = client.chat.completions.create(
                model="gemma-3-1b-it",
                stream=False,
                messages=[{"role": "user", "content": messages}],
                temperature=temperature,
                top_p=top_p,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling gemma model: {e}")
            return None
    


class TopicDebateGenerator:
    def __init__(self, output_dir=None, log_file=None):
       
        self.output_dir = output_dir
        self.log_file = log_file

        self.executor = ThreadPoolExecutor(max_workers=20)

        self.judge_instances = [JudgeModelPool() for _ in range(10)]
        
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
        """Helper function: Write JSON data to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def extract_json_from_text(self, text, match_char='[]'):
        """Extract JSON part from text"""
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
            
            # Last resort: manual parsing
            topics = []
            # Update regex to match different field names
            topic_pattern = r'"(?:topic|title)":\s*"([^"]*)"'
            desc_pattern = r'"description":\s*"([^"]*)"'
            expl_pattern = r'"(?:explanation|reason)":\s*"([^"]*)"'
            reason_pattern = r'"reason":\s*"([^"]*)"'
            
            topics_found = re.findall(topic_pattern, text)
            descriptions = re.findall(desc_pattern, text)
            explanations = re.findall(expl_pattern, text)
            reasons = re.findall(reason_pattern, text)
            
            for i in range(min(len(topics_found), len(descriptions))):
                topic_dict = {
                    "topic": topics_found[i],
                    "description": descriptions[i]
                }
                
                # Add explanation field if present
                if i < len(explanations):
                    topic_dict["explanation"] = explanations[i]
                
                # Add reason field if present and different from explanation
                if i < len(reasons) and (i >= len(explanations) or reasons[i] != explanations[i]):
                    topic_dict["reason"] = reasons[i]
                
                topics.append(topic_dict)
            
            return topics
    
    def judge_topics(self, primary_category, secondary_category, not_value, 
                    debate1_critiques, debate1_critiques2, number):
        """Use judge model to make final decision on which topics to keep"""
        # Prepare all topics and their critiques
        all_topics_with_critiques = []
        import pdb
        # pdb.set_trace()
        for topic in debate1_critiques['accepted']:
            all_topics_with_critiques.append({
                "title": topic.get("topic", ""),
                "explanation": topic.get("explanation", topic.get("reason", "")),
                "source": "qwen",
                "keep": True,
                "reason": topic.get("reason", ""),
            })
        for topic in debate1_critiques['rejected']:
            all_topics_with_critiques.append({
                "title": topic.get("topic", ""),
                "explanation": topic.get("explanation", topic.get("reason", "")),
                "source": "qwen",
                "keep": False,
                "reason": topic.get("reason", ""),
            })
        import pdb
        # pdb.set_trace()
        for topic in debate1_critiques['accepted']:
            all_topics_with_critiques.append({
                "title": topic.get("topic", ""),
                "explanation": topic.get("explanation", topic.get("reason", "")),
                "source": "mistral",
                "keep": True,
                "reason": topic.get("reason", ""),
            })
        
        for topic in debate1_critiques['rejected']:
            all_topics_with_critiques.append({
                "title": topic.get("topic", ""),
                "explanation": topic.get("explanation", topic.get("reason", "")),
                "source": "mistral",
                "keep": False,
                "reason": topic.get("reason", ""),
            })
        
        # Build judge prompt
        judge_prompt = f"""
You are a fair and insightful topic evaluation judge.

Your task is to assess a set of topic candidates generated by AI models for the following category:

- Primary Category (BISAC): {primary_category}
- Secondary Category: {secondary_category}

Each topic has been reviewed by peer models. Use their evaluations as **reference only** — do not blindly accept or reject based on their suggestions. Apply your own comprehensive judgment to determine which topics are **worth keeping**.

Here are the topic candidates and critiques:
"""
        import pdb
        for i, topic in enumerate(all_topics_with_critiques):
            judge_prompt += f"""Topic {i+1}:
Title: {topic['title']}
Explanation: {topic['explanation']}
Source Model: {topic['source']}
Suggested Action: {"Keep" if topic['keep'] else "Reject"}
Critique Reason: {topic['reason']}
"""
        if not_value is not None:
            judge_prompt += f"Ensure that the retained topics do not fall under any of the following categories:{not_value}"
        import pdb
        # pdb.set_trace()
        judge_prompt += f"""
Evaluation Guidelines:

You are not required to select a fixed number of topics. Instead, remove any topics that meet the criteria below.
Do not fully accept or reject topics solely based on peer critique. Use critique as a source of insight, not as ground truth.
The goal is to retain a set of topics that is diverse, well-balanced, and representative of the overall category.

Removal Criteria:

1. Redundancy / Lack of Diversity – Remove topics that are too similar to others without offering a distinct angle or sub-theme.
2. Insufficient Contribution to Coverage – Remove topics that do not expand the content space meaningfully or are too narrow to add value.
3. Low Relevance – Remove topics that are not clearly aligned with the specified primary and secondary categories.
4. Lower Quality in Similar Groups – If multiple topics are similar, retain the one with clearer wording, stronger conceptual value, or broader applicability, and remove the weaker ones.

Return your decision (remove topic you select) using the following JSON format:"""

        judge_prompt += """{
  "rejected_topics": [
    {
      "title": ...,
      "explanation": ...,
      "source": "Generated Model source",
      "critique reason": "generate Critique reason",
      "judge reason": "Give your reasons for choosing remove this topic"
    },
    ...
  ],
  "summary": "Summarize your overall decision-making approach. Explain how you balanced relevance, diversity, coverage, critique insights, and your own reasoning."
}
tips:The returned title field directly gives the title name. Don't add "Title: " or "Topic 4: "
for example:
Topic 1:
Title: Christian Living for Teens
Explanation: ...
Source Model: ...
Suggested Action: ...
Critique Reason: ...
if you decide remove this topic, please return
"rejected_topics": [
    {   
        "title": Christian Living for Teens,
        ...
    },
    ...
]
"""
        # pdb.set_trace()
        # Use API to generate judge evaluation
        import random
        deepseek_instance = random.choice(self.judge_instances)
        judge_response = deepseek_instance.call(judge_prompt)
        import pdb 
        # pdb.set_trace()
        # Parse judge evaluation result
        try:
            judge_result = self.extract_json_from_text(judge_response, '{}')
            if isinstance(judge_result, list) and len(judge_result) > 0:
                judge_result = judge_result[0]
        except Exception as e:
            print(f"Error parsing judge evaluation JSON: {e}")
            judge_result = {"selected_topics": [], "summary": "Parsing failed"}
        
        return judge_result
    
    def judge_topics_for_category(self, primary_category, secondary_category, not_value, number):
        """Stage 3: Judge decision"""
        print(f"Judge decision stage: {primary_category} - {secondary_category}")
        key = f"{primary_category}_{secondary_category}".replace("/", "_")
        import pdb
        # pdb.set_trace()
        
        # Check if critique file exists
        debate1_critiques_file = os.path.join(self.critique_dir, f"{key}_debate1_critiques.json")
        debate1_critiques_file = os.path.join(self.critique_dir, f"{key}_debate1_critiques.json")
        
        # Check if judge decision file already exists
        judge_file = os.path.join(self.judge_dir, f"{key}_judge.json")
        
        # If judge decision file exists, return directly
        if os.path.exists(judge_file):
            print(f"Judge decision already generated, skipping decision stage")
            return self._read_json_from_file(judge_file)
        
        
        if not os.path.exists(debate1_critiques_file) or not os.path.exists(debate1_critiques_file):
            print(f"Critique file not generated, skipping decision stage")
            return {"selected_topics": [], "summary": "Waiting for critique file timeout"}
        
        # Load critique file
        try:
            debate1_critiques = self._read_json_from_file(debate1_critiques_file)
            debate1_critiques = self._read_json_from_file(debate1_critiques_file)
        except Exception as e:
            print(f"Error loading critique file: {e}")
            return {"selected_topics": [], "summary": f"Error loading critique file: {e}"}
        
        # Judge decision
        try:
            judge_result = self.judge_topics(
                primary_category, secondary_category, not_value,
                debate1_critiques, debate1_critiques, number
            )
            self._write_json_to_file(judge_file, judge_result)
            return judge_result
        except Exception as e:
            print(f"Error during judge decision: {e}")
            return {"selected_topics": [], "summary": f"Judge decision error: {e}"}
    
    def process_category_pair(self, primary_category, secondary_category, not_value, number=5):
        """Process a category pair, execute in three stages"""
        print(f"Processing category: {primary_category} - {secondary_category}")
        
        return self.judge_topics_for_category(primary_category, secondary_category, not_value, number)
    
    def _read_json_from_file(self, filename):
        """Helper function: Read JSON data from file"""
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)

def process_single_category(args):
    """Single category processing function for multiprocessing"""
    import pdb
    # pdb.set_trace()
    category_data, output_dir, num_to_generate = args
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
            # If lock cannot be acquired, file is being processed by another process
            print(f"Category {primary_category} - {secondary_category} is being processed by another process, skipping")
            return None
    
    try:
        generator = TopicDebateGenerator(output_dir=output_dir)
        judge_result = generator.process_category_pair(
            primary_category,
            secondary_category,
            not_value,
            number
        )

        # Save result, use file lock to ensure safe writing
        result = {
            "primary_category": primary_category,
            "secondary_category": secondary_category,
            "existing_topics": existing_topics,
            "selected_topics": judge_result.get("rejected_topics", []),
            "judge_summary": judge_result.get("summary", "")
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Acquire exclusive lock
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(result, f, ensure_ascii=False, indent=2)
            # Release lock
            fcntl.flock(f, fcntl.LOCK_UN)
        
        return result
    except Exception as e:
        print(f"Error processing category {primary_category} - {secondary_category}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Topic Debate Generator")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_to_generate", type=int, default=10, help="Number of topics to generate per category")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes, defaults to half of CPU cores")
    # parser.add_argument("--batch_size", type=int, default=5, help="Batch size for category processing")
    
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
    
    print(f"Total {len(categories_data)} categories, {len(categories_data) - len(categories_to_process)} processed, {len(categories_to_process)} remaining")
    
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
    process_args = [(item, args.output_dir, args.num_to_generate) for item in categories_to_process]
    
    # Test single processing
    # process_single_category(process_args[0])
    # return
    
    # Use process pool to process multiple categories in parallel
    results = {}
    with Pool(processes=processes) as pool:
        for result in tqdm(pool.imap_unordered(process_single_category, process_args), 
                            total=len(process_args), 
                            desc="Processing categories"):
            if result:
                key = f"{result['primary_category']}_{result['secondary_category']}".replace("/", "_")
                results[key] = result
    
    # Save all results, use file lock to ensure safety
    all_results_file = os.path.join(args.output_dir, "all_results.json")
    
    # If result file exists, merge results
    if os.path.exists(all_results_file):
        with open(all_results_file, 'r+', encoding='utf-8') as f:
            # Acquire exclusive lock
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                existing_results = json.load(f)
                existing_results.update(results)
                results = existing_results
                
                # Go to file start and truncate
                f.seek(0)
                f.truncate()
                json.dump(results, f, ensure_ascii=False, indent=2)
            finally:
                # Ensure lock is released
                fcntl.flock(f, fcntl.LOCK_UN)
    else:
        with open(all_results_file, 'w', encoding='utf-8') as f:
            # Acquire exclusive lock
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(results, f, ensure_ascii=False, indent=2)
            finally:
                # Ensure lock is released
                fcntl.flock(f, fcntl.LOCK_UN)
    
    print(f"All results have been saved to {args.output_dir}")

if __name__ == "__main__":
    main()
