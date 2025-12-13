import json
import time
import logging
from typing import Optional, Dict, Any, Callable, Tuple
from transformers import AutoTokenizer
from openai import OpenAI
import requests
from functools import wraps
import os

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def model_query_decorator(func: Callable) -> Callable:
    """
    model query decorator, handle common pre-processing and retry logic
    get max_tries and time_sleep from self instance dynamically
    """
    @wraps(func)
    def wrapper(self, prompt: str) -> str:
        # pre-processing: check empty prompt
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt, return empty string")
            return "", None
        
        # pre-processing: truncate prompt
        processed_prompt = self._truncate_prompt(prompt.strip())
        
        # get retry parameters from self instance
        max_tries = getattr(self, 'max_tries', 5)
        time_sleep = getattr(self, 'time_sleep', 1.0)

        time.sleep(time_sleep)
        
        # retry mechanism
        last_exception = None
        
        for attempt in range(1, max_tries + 1):
            try:
                logger.info(f"Try {attempt} times...")
                
                # call the specific query method
                answer, thinking = func(self, processed_prompt)
                
                logger.info(f"Query successful, try {attempt} times")
                
                return answer, thinking
                
            except KeyboardInterrupt:
                logger.info("User interrupt")
                raise
            except Exception as e:
                last_exception = e
                logger.warning(f"API error (try {attempt}/{max_tries}): {e}")
            
            # exponential backoff strategy
            if attempt < max_tries:
                sleep_time = time_sleep * (2 ** (attempt - 1))
                logger.info(f"Wait {sleep_time:.1f} seconds and retry...")
                time.sleep(sleep_time)
        
        # all tries failed
        logger.error(f"All {max_tries} tries failed, last error: {last_exception}")
        return "", None
    
    return wrapper


class ModelManagerBase:
    """Base model manager"""

    def __init__(
        self,
        tokenizer_path: str, 
        context_max_length: int,
        url: str, 
        api_key: str,
        temperature: float,
        max_new_tokens: int,
        timeout: int,
        max_tries: int,
        time_sleep: float,
    ):
        # parameter validation
        if not os.path.exists(tokenizer_path):
            raise ValueError("tokenizer_path is not found")
        if context_max_length <= 0:
            raise ValueError("context_max_length must be greater than 0")
        if max_tries <= 0:
            raise ValueError("max_tries must be greater than 0")
        
        self.tokenizer_path = tokenizer_path
        self.context_max_length = context_max_length
        self.url = url
        self.api_key = api_key
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.max_tries = max_tries
        self.time_sleep = time_sleep
        self.tokenizer = self._get_tokenizer()
    
    def _get_tokenizer(self) -> AutoTokenizer:
        """Get tokenizer"""
        try:
            return AutoTokenizer.from_pretrained(
                self.tokenizer_path, 
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt, keep important parts"""
        input_ids = self.tokenizer.encode(prompt)
        
        if len(input_ids) <= self.context_max_length:
            return prompt
        
        truncated_input_ids = input_ids[:self.context_max_length//2] + input_ids[-self.context_max_length//2:]

        truncated_prompt = self.tokenizer.decode(
            truncated_input_ids, 
            skip_special_tokens=True
        )

        return truncated_prompt
    
    @model_query_decorator
    def query(self, processed_prompt: str) -> str:
        """Query LLM model"""
        raise NotImplementedError("Subclass must implement this method")

class ModelManagerOpenAI(ModelManagerBase):
    """OpenAI model manager"""

    def __init__(
        self,
        model_name: str,
        tokenizer_path: str = "model/Tokenizers/qwen",
        context_max_length: int = 120000, # 128k - 8k
        url: str = "http://127.0.0.1:8000/v1", 
        api_key: str = "EMPTY",
        temperature: float = 1.0,
        max_new_tokens: int = 8192,
        timeout: int = 1200,
        max_tries: int = 5,
        time_sleep: float = 1.0,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(tokenizer_path, context_max_length, url, api_key, 
                temperature, max_new_tokens, timeout, max_tries, time_sleep)
        self.model_name = model_name
        self.extra_body = extra_body or {}
        self.client = self._create_client()
    
    def _create_client(self) -> OpenAI:
        """Create OpenAI client"""
        try:
            return OpenAI(
                base_url=self.url,
                api_key=self.api_key
            )
        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {e}")
            raise
    
    @model_query_decorator
    def query(self, processed_prompt: str) -> Tuple[str, Optional[str]]:
        """Query LLM model - only handle OpenAI specific logic"""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": processed_prompt}],
            temperature=self.temperature,
            extra_body=self.extra_body,
            max_tokens=self.max_new_tokens,
            timeout=self.timeout,
        )
        answer = completion.choices[0].message.content
        try:
            thinking = completion.choices[0].message.reasoning_content
        except:
            thinking = None
        return answer, thinking