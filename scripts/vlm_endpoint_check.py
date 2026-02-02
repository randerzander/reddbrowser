#!/usr/bin/env python3
"""Standalone check for the OpenRouter VLM captioning endpoint."""

import argparse
import json
import os
import sys
from openai import OpenAI

# 1x1 transparent PNG
TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def _parse_config(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}

    config = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    config[key] = value
    except Exception:
        return {}

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the OpenRouter VLM endpoint with a tiny image request."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLM_MODEL"),
        help="Model name (default: VLM_MODEL env var if set).",
    )
    parser.add_argument(
        "--config",
        default=os.getenv("REDD_BROWSER_CONFIG", "config.yaml"),
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--try-fallbacks",
        action="store_true",
        help="Try a small fallback list if the primary model fails.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image in detail.",
        help="Prompt for the VLM.",
    )
    return parser.parse_args()


def _attempt_request(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{TINY_PNG_BASE64}"
                        },
                    },
                ],
            }
        ],
        max_tokens=200,
    )

    try:
        message = response.choices[0].message
    except Exception:
        message = None

    content = None
    if message is not None:
        try:
            content = message.content
        except Exception:
            content = None

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(text)
        content = "\n".join(parts) if parts else None

    if (not content or not str(content).strip()) and message is not None:
        # Some providers return text in nonstandard fields.
        fallback_fields = ["output_text", "text", "reasoning"]
        for field in fallback_fields:
            value = getattr(message, field, None)
            if value and str(value).strip():
                content = value
                break

    if not content or not str(content).strip():
        print("Empty response content from VLM endpoint. Raw response:")
        try:
            payload = response.model_dump()
        except Exception:
            payload = repr(response)
        if isinstance(payload, dict):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(payload)
        raise RuntimeError("Empty response content from VLM endpoint")

    return str(content).strip()


def main() -> int:
    args = parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment")
        print("Set it with: export OPENROUTER_API_KEY='your-api-key'")
        return 1

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    config = _parse_config(args.config)

    fallback_models = [
        "qwen/qwen-2.5-vl-7b-instruct:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
    ]
    models_to_try = []
    config_model = config.get("vlm_model") if config else None
    if args.model:
        models_to_try.append(args.model)
    elif config_model:
        models_to_try.append(config_model)
    elif args.try_fallbacks:
        models_to_try.extend(fallback_models)
    else:
        print("ERROR: No model specified.")
        print("Set VLM_MODEL, or set vlm_model in config.yaml, or pass --model, or use --try-fallbacks.")
        return 1

    last_error = None
    for model in models_to_try:
        try:
            content = _attempt_request(client, model, args.prompt)
        except Exception as exc:
            last_error = exc
            print(f"Model failed: {model}")
            print(f"Reason: {exc}")
            continue

        print("VLM endpoint OK")
        print("Model:", model)
        print("Response:")
        print(content)
        return 0

    print("ERROR: All model attempts failed.")
    if last_error is not None:
        print(f"Last error: {last_error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
