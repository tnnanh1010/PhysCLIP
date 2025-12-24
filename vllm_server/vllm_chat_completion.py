import datetime
import os
import json
import asyncio
import warnings
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Union, Literal
import random
from .base import BaseModelRunner

async def vllm_chat_completion(
    model: str,
    messages: List[Dict[str, Any]],
    model_type: Literal["llm", "vlm"] = "llm",
    temperature: float = 1.0
) -> Dict[str, Any]:
    """
    Make a direct API call to VLLM's chat completions endpoint.
    
    Args:
        model: Model to use (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
        messages: List of conversation messages
        model_type: Type of model - "llm" for text-only, "vlm" for vision-language
        
    Returns:
        Response from VLLM API
    """
    if model_type == "llm":
        urls = [
            # "http://internal-gateway.thedrylab.com:8001"
            "http://0.0.0.0:8001",
            # "http://0.0.0.0:3301",
            # "http://0.0.0.0:3302",
            # "http://0.0.0.0:3303",

        ]
    elif model_type == "vlm":
        urls = [
            "http://0.0.0.0:8001",
            "http://0.0.0.0:8002",
            # "http://0.0.0.0:3300",
            # "http://0.0.0.0:3301",
            # "http://0.0.0.0:3302",
            # "http://0.0.0.0:3303",
            # "http://0.0.0.0:3304",
            # "http://0.0.0.0:3305",
            # "http://0.0.0.0:3306",
            # "http://0.0.0.0:3307"

        ]
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'llm' or 'vlm'")
    
    urls = [url + "/v1/chat/completions" for url in urls]
    url = random.choice(urls)
    
    headers = {
        "Content-Type": "application/json",
    }
    # Prepare request body
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature
        
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"VLLM {model_type.upper()} API error {response.status}: {error_text}")
            
            response_data = await response.json()
            return response_data

import re

def extract_and_strip_actions(text):
    # Pattern to match <action><name>...</name></action>
    pattern = r"<action>\s*<(\w+)>\s*(.*?)\s*</\1>\s*</action>"

    actions = []
    
    def replacer(match):
        name = match.group(1).strip()
        content = match.group(2).strip()
        actions.append({
            "name": name,
            "content": content
        })
        return ""  # Remove the full <action>...</action> block

    # Replace and collect
    processed_text = re.sub(pattern, replacer, text, flags=re.DOTALL)

    # Clean up extra whitespace left after removing blocks
    processed_text = re.sub(r'\n\s*\n', '\n\n', processed_text).strip()

    return processed_text, actions, text

def get_actions(text):
    # Pattern to match <action><name>...</name></action>
    pattern = r"<action>\s*<(\w+)>\s*(.*?)\s*</\1>\s*</action>"

    actions = []
    
    def replacer(match):
        name = match.group(1).strip()
        content = match.group(2).strip()
        if name == "execute":
            actions.append({
                "function": {
                    "name": name,
                    "arguments": json.dumps({
                        "code": content
                    })
                }
            })
        elif name == "list_files":
            actions.append({
                "function": {
                    "name": name,
                    "arguments": json.dumps(content)
                }
            })
        elif name == "submit_answer":
            actions.append({
                "function": {
                    "name": name,
                    "arguments": json.dumps({
                        "answer": content
                    })
                }
            })
        else:
            raise ValueError(f"Unknown action: {name}") 
        return ""  # Remove the full <action>...</action> block
    # Replace and collect
    processed_text = re.sub(pattern, replacer, text, flags=re.DOTALL)
    # Clean up extra whitespace left after removing blocks
    processed_text = re.sub(r'\n\s*\n', '\n\n', processed_text).strip()
    return actions

class VLLMModelRunner(BaseModelRunner):
    """VLLM-specific model runner implementation."""
    
    def __init__(self, model: str, model_type: Literal["llm", "vlm"] = "llm"): 
        super().__init__(model)
        # self.model_type = model_type
    
    async def run_model_completion(self, messages: List[Dict[str, Any]], model_type: Literal["llm", "vlm"], temperature: float=1.0) -> Dict[str, Any]:
        """Run VLLM completion."""
        
        return await vllm_chat_completion(
            model=self.model,
            messages=messages,
            model_type=model_type,
            temperature=temperature
        )
    
    def extract_message_content(self, response: Dict[str, Any], type=None) -> str:
        """Extract message content from VLLM response."""
        message = response["choices"][0]["message"]
        content = message.get("content", "")
        
        # Handle thinking mode - extract only the answer part
        if "<think>" in content and "</think>" in content:
            try:
                # Find the end of thinking content
                thinking_end = content.rfind("</think>")
                if thinking_end != -1:
                    # Extract content after </think>
                    answer_content = content[thinking_end + len("</think>"):].strip()
                    return answer_content
            except Exception as e:
                # Fallback to original content if parsing fails
                print(f"Warning: Failed to parse thinking content: {e}")
                return content
        
        # Legacy support for type parameter
        if type == "thinking":
            thinking_end_marker = "</think>"
            if thinking_end_marker in content:
                end_idx = content.find(thinking_end_marker)
                # Return content after the thinking section (the final answer)
                final_answer = content[end_idx + len(thinking_end_marker):].strip()
                return final_answer
        
        return content

 
    
    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from VLLM response."""
        message = response["choices"][0]["message"]["content"]
        actions = get_actions(message)
        return actions
    
    def create_conversation_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Create conversation message dict from VLLM response."""
        message = response["choices"][0]["message"]["content"]
        if "reasoning" in response["choices"][0]["message"]:
            reasoning = response["choices"][0]["message"]["reasoning"]
            message_dict = {
                "role": "assistant",
                "content": f"<thinking>{reasoning}</thinking>\n{message}",
            }
        else:
            message_dict = {
                "role": "assistant",
                "content": message,
            }
        
        return message_dict
    
    def get_model_specific_system_prompt(self) -> str:
        """Get VLLM-specific system prompt additions."""
        return ""