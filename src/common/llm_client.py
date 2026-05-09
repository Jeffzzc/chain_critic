"""LLM Client Module - Simple interface for calling different LLM providers"""

import os
import random
import time
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
import requests
import json

# Automatically load local .env file if exists
load_dotenv()


def _require_env(key: str, default: Optional[str] = None, required: bool = True) -> str:
    """Get required environment variable, raise exception if missing"""
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


# Default system message for all providers
DEFAULT_SYSTEM_MESSAGE = "You are a professional LLM evaluation expert, skilled in analyzing chain-of-thought quality dimensions."

# Provider configuration mapping
PROVIDER_CONFIGS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "base_url_env": "OPENAI_BASE_URL",
        "default_model": "gpt-4",
        "default_base_url": "https://api.openai.com/v1",
    },
    "deepseek_v31": {
        "api_key_env": "DEEPSEEK_V31_API_KEY",
        "model_env": "DEEPSEEK_V31_MODEL",
        "base_url_env": "DEEPSEEK_V31_BASE_URL",
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
    "deepseek_r1": {
        "api_key_env": "DEEPSEEK_R1_API_KEY",
        "model_env": "DEEPSEEK_R1_MODEL",
        "base_url_env": "DEEPSEEK_R1_BASE_URL",
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
    "doubao_seed_16": {
        "api_key_env": "DOUBAO_SEED_16_API_KEY",
        "model_env": "DOUBAO_SEED_16_MODEL",
        "base_url_env": "DOUBAO_SEED_16_BASE_URL",
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
    "doubao_1.5_lite_32k": {
        "api_key_env": "DOUBAO_1_5_LITE_32K_API_KEY",
        "model_env": "DOUBAO_1_5_LITE_32K_MODEL",
        "base_url_env": "DOUBAO_1_5_LITE_32K_BASE_URL",
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
    "doubao_1.5_pro_256k": {
        "api_key_env": "DOUBAO_1_5_PRO_256K_API_KEY",
        "model_env": "DOUBAO_1_5_PRO_256K_MODEL",
        "base_url_env": "DOUBAO_1_5_PRO_256K_BASE_URL",
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
    # === Modification 1: Register zyuncs here ===
    "zyuncs": {
        "api_key_env": "ZYUNCS_TOKEN",
        "model_env": "ZYUNCS_MODEL", 
        "default_model": "cortex-4"
    },
}


def _get_default_provider() -> Optional[str]:
    """Get default provider by finding the first one with available configuration"""
    # Check LLM_PROVIDER env var first
    provider = os.getenv("LLM_PROVIDER")
    if provider and provider.lower() in PROVIDER_CONFIGS:
        return provider.lower()

    return None

# === Modification 2: Define ask_zyuncs function ===
def ask_zyuncs(prompt: str, model: str = "cortex-4") -> str:
    token = os.getenv("ZYUNCS_TOKEN", "")
    if not token:
        raise ValueError("Please set environment variable ZYUNCS_TOKEN")

    url = "https://llm.api.zyuncs.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": token,  # 你的 curl 示例就是 Authorization: <token>
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # 可选：如果服务端支持再打开
        # "max_tokens": 1024,
        # "temperature": 0.3,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)

        # 先拿到文本，避免 json 解析失败时看不到原始返回
        resp_text = resp.text
        resp.raise_for_status()

        try:
            result_json = resp.json()
        except Exception:
            raise RuntimeError(f"Zyuncs returned non-JSON response: {resp_text[:500]}")

        # --------- 1) OpenAI 兼容格式：顶层 choices ----------
        if isinstance(result_json, dict) and "choices" in result_json:
            choices = result_json.get("choices") or []
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "")
            if choices and "text" in choices[0]:
                return choices[0].get("text", "")
            raise RuntimeError(f"Zyuncs JSON has choices but no content field: {json.dumps(result_json)[:800]}")

        # --------- 2) 你原先假设的格式：ret_code / result ----------
        if isinstance(result_json, dict) and ("ret_code" in result_json or "result" in result_json):
            if result_json.get("ret_code") and result_json.get("ret_code") != "000000":
                raise RuntimeError(f"Zyuncs API business error: {result_json.get('ret_msg', '')}")

            # 常见：result.choices[0].message.content
            if "result" in result_json:
                r = result_json["result"]
                if isinstance(r, dict) and "choices" in r and r["choices"]:
                    msg = r["choices"][0].get("message", {})
                    return msg.get("content", "")

            raise RuntimeError(f"Zyuncs JSON unexpected structure: {json.dumps(result_json)[:800]}")

        # --------- 3) 兜底：如果返回 error ----------
        if isinstance(result_json, dict) and "error" in result_json:
            raise RuntimeError(f"Zyuncs API error: {json.dumps(result_json['error'])[:800]}")

        raise RuntimeError(f"Zyuncs response unrecognized: {json.dumps(result_json)[:800]}")

    except requests.exceptions.HTTPError as e:
        # 把 HTTP 状态码和响应体打出来（截断），方便定位 401/403/429/5xx
        status = getattr(e.response, "status_code", None)
        body = getattr(e.response, "text", "")
        raise RuntimeError(f"Zyuncs HTTPError status={status}, body={body[:800]}") from e
    except Exception as e:
        raise RuntimeError(f"Zyuncs API failed: {e}") from e

def ask_llm(
    prompt: str,
    provider: Optional[str] = None,
    temperature: float = 0.3,
    max_retries: int = 15,
    retry_delay: float = 5.0,
    **kwargs
) -> str:
    """
    Simple function to call LLM API
    """
    # Determine provider
    provider = (provider or _get_default_provider())
    if not provider:
        raise RuntimeError(
            "No provider specified. Please set LLM_PROVIDER or configure one."
        )
    provider = provider.lower()
    
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unsupported provider: {provider}")
    
    config = PROVIDER_CONFIGS[provider]
    
    # # === Modification 3: Special branch logic for zyuncs ===
    # if provider == "zyuncs":
    #     model = kwargs.get("model") or os.getenv(config.get("model_env", ""), config.get("default_model"))
    #     for attempt in range(max_retries):
    #         try:
    #             return ask_zyuncs(prompt, model=model)
    #         except Exception as e:
    #             if attempt < max_retries - 1:
    #                 delay = retry_delay + retry_delay * random.random()
    #                 print(f"Zyuncs API call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {delay:.1f}s...")
    #                 time.sleep(delay)
    #             else:
    #                 raise RuntimeError(f"API call failed after {max_retries} attempts: {e}")
    
    # elif provider == "vllm":
    #     # 本地部署的 vLLM，请求端口
    #     url = f"http://localhost:{port}/v1/completions"
    #     headers = {"Content-Type": "application/json"}
    #     data = {
    #         "model": "Qwen3-Omni-30B-A3B-Instruct",
    #         "prompt": prompt,
    #         "max_tokens": 100,
    #     }
        
    #     try:``
    #         response = requests.post(url, json=data, headers=headers)
    #         response.raise_for_status()  # 如果请求失败会抛出异常
    #         result = response.json()
    #         return result.get("choices", [{}])[0].get("text", "")
    #     except requests.exceptions.RequestException as e:
    #         print(f"Error calling vLLM: {e}")
    #         return "Error in response"
    # else:
    #     raise ValueError(f"Unsupported provider: {provider}")

    # Standard OpenAI Logic below
    api_key = _require_env(config["api_key_env"])
    model = os.getenv(config["model_env"], config.get("default_model"))
    base_url = os.getenv(config.get("base_url_env"), config.get("default_base_url"))
    system_message = DEFAULT_SYSTEM_MESSAGE
    
    if not model:
        raise RuntimeError(f"Model not specified for provider {provider}.")
    
    if not base_url:
        raise RuntimeError(f"Base URL not specified for provider {provider}.")
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                delay = retry_delay + retry_delay * random.random()
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"API call failed after {max_retries} attempts: {e}")
    
    raise RuntimeError("Unexpected error in API call")


def list_providers() -> list:
    """List all supported providers"""
    return list(PROVIDER_CONFIGS.keys())


# Usage example
if __name__ == "__main__":
    answer = ask_llm("你是谁", provider="zyuncs")
    print(answer)
