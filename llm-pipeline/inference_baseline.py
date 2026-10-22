import os
import re
import json
import time
import psutil
import argparse
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()


def get_current_ram_usage() -> Dict[str, Any]:
    """
    Measures current system RAM and GPU VRAM usage.
    """
    process = psutil.Process(os.getpid())
    process_ram_gb = process.memory_info().rss / (1024 ** 3)
    vm = psutil.virtual_memory()
    
    usage = {
        "process_ram_gb": round(process_ram_gb, 3),
        "system_ram_used_gb": round(vm.used / (1024 ** 3), 2),
        "system_ram_total_gb": round(vm.total / (1024 ** 3), 2),
        "system_ram_percent": vm.percent,
        "gpu_vram_allocated_gb": 0.0,
        "gpu_vram_reserved_gb": 0.0
    }
    
    if torch.cuda.is_available():
        usage["gpu_vram_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024 ** 3), 3)
        usage["gpu_vram_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024 ** 3), 3)
        
    return usage


def extract_json_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses JSON object from model output, handling
    markdown backticks, trailing commas, preamble text, or trailing commentary.
    """
    def try_parse(candidate: str) -> Optional[Dict[str, Any]]:
        # Strip trailing commas before closing braces/brackets
        sanitized = re.sub(r",\s*([}\]])", r"\1", candidate.strip())
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            return None

    block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if block_match:
        res = try_parse(block_match.group(1))
        if res:
            return res

    obj_match = re.search(r"(\{[\s\S]*\})", raw_text)
    if obj_match:
        res = try_parse(obj_match.group(1))
        if res:
            return res

    # Fallback extraction: parse key-value pairs directly if JSON was truncated by token limit
    hype_m = re.search(r'"?hype_score"?\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw_text)
    story_m = re.search(r'"?story_score"?\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw_text)
    nostalgia_m = re.search(r'"?nostalgia_score"?\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw_text)
    if hype_m and story_m and nostalgia_m:
        try:
            return {
                "hype_score": float(hype_m.group(1)),
                "story_score": float(story_m.group(1)),
                "nostalgia_score": float(nostalgia_m.group(1))
            }
        except (ValueError, TypeError):
            pass

    return None


def validate_vibe_scores(data: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, float]]:
    """
    Validates that the parsed dictionary contains hype_score, story_score,
    and nostalgia_score within [1.0, 10.0].
    """
    if not isinstance(data, dict):
        return False, {}

    required_keys = ["hype_score", "story_score", "nostalgia_score"]
    validated = {}

    for key in required_keys:
        if key not in data:
            return False, {}
        try:
            val = float(data[key])
            if 0.0 <= val <= 10.0:
                validated[key] = round(val, 1)
            else:
                return False, {}
        except (ValueError, TypeError):
            return False, {}

    return True, validated


class SteamLLMInferenceBaseline:
    """
    Establishes a working inference baseline with Qwen3.5-0.8B on GPU,
    extracting game vibe scores and monitoring RAM usage after each prediction.
    """
    def __init__(self, model_id: str = "Qwen/Qwen3.5-0.8B", device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = os.getenv("HF_TOKEN") or None

        print(f"Initializing Steam LLM Baseline with model: '{self.model_id}' on device: '{self.device}'")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token
        )
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            token=self.hf_token
        )
        if self.device != "cuda":
            self.model.to(self.device)
            
        self.model.eval()
        print("Model and Tokenizer loaded successfully.")

    def build_prompt(self, game_name: str, genres: str, description: str) -> str:
        system_instruction = (
            "You are an expert video game industry analyst. Your task is to evaluate a game's concept "
            "and description, then output three vibe scores on a scale from 1.0 to 10.0:\n"
            "1. hype_score: anticipated community excitement, viral potential, or multiplayer buzz.\n"
            "2. story_score: narrative depth, lore, worldbuilding, or emotional engagement.\n"
            "3. nostalgia_score: retro aesthetic, classic genre homage, or legacy franchise appeal.\n\n"
            "You MUST respond ONLY with a valid JSON object matching this exact schema:\n"
            "{\n"
            '  "hype_score": float,\n'
            '  "story_score": float,\n'
            '  "nostalgia_score": float\n'
            "}"
        )

        user_content = (
            f"Game Title: {game_name}\n"
            f"Genres: {genres}\n"
            f"Description: {description.strip()[:1000]}\n\n"
            "Output the JSON evaluation now:"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return f"{system_instruction}\n\nUser: {user_content}\n\nAssistant:"

    @torch.no_grad()
    def predict(self, game_name: str, genres: str, description: str, max_new_tokens: int = 150) -> Dict[str, Any]:
        prompt = self.build_prompt(game_name, genres, description)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        start_time = time.time()
        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id
        )
        elapsed = time.time() - start_time
        
        new_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        tokens_per_sec = len(new_tokens) / max(elapsed, 1e-4)

        # Measure RAM & VRAM immediately after output generation
        ram_usage = get_current_ram_usage()

        parsed_json = extract_json_payload(raw_output)
        is_valid, validated_scores = validate_vibe_scores(parsed_json)

        return {
            "game_name": game_name,
            "raw_output": raw_output,
            "parsed_json": parsed_json,
            "is_valid_json": is_valid,
            "scores": validated_scores,
            "latency_seconds": round(elapsed, 3),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "ram_usage": ram_usage
        }


def run_inference_baseline(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    num_samples: int = 3,
    output_path: str = "llm-pipeline/inference_baseline_results.json"
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"   Steam Hit Predictor: LLM Inference Baseline        ")
    print(f"   Model: {model_id}                                   ")
    print(f"=======================================================\n")

    # Load from cleaned train split (which now contains all clean descriptions)
    train_split_path = "cleaned_data/train.csv"
    cleaned_desc_path = "cleaned_data/steam_descriptions_cleaned.csv"

    if os.path.exists(train_split_path):
        df_games = pd.read_csv(train_split_path)
    elif os.path.exists(cleaned_desc_path):
        df_steam = pd.read_csv("raw_data/steam.csv")
        df_desc = pd.read_csv(cleaned_desc_path)
        df_games = pd.merge(df_steam, df_desc, left_on="appid", right_on="steam_appid")
    else:
        # Fallback to raw data
        df_steam = pd.read_csv("raw_data/steam.csv")
        df_desc = pd.read_csv("raw_data/steam_description_data.csv")
        df_games = pd.merge(df_steam, df_desc, left_on="appid", right_on="steam_appid")
        df_games["clean_description"] = df_games.get("short_description", "")

    baseline = SteamLLMInferenceBaseline(model_id=model_id)

    # Pick sample games across genres
    sample_appids = [10, 400, 570]  # Counter-Strike, Portal, Dota 2
    samples = df_games[df_games["appid"].isin(sample_appids)]
    
    if len(samples) < num_samples:
        samples = df_games.head(num_samples)

    runs = []
    print("\n--- Running Baseline Inference with RAM Tracking ---")
    for idx, row in samples.iterrows():
        title = row["name"]
        appid = int(row.get("appid", 0))
        genres = str(row.get("genres", ""))
        desc = str(row.get("clean_description") or row.get("short_description") or "")
        
        print(f"\nEvaluating Game: '{title}' (AppID: {appid}, Genres: {genres})")
        res = baseline.predict(title, genres, desc)
        mem = res["ram_usage"]
        
        print(f"  * Latency:          {res['latency_seconds']}s ({res['tokens_per_sec']} tok/s)")
        print(f"  * Valid JSON:       {res['is_valid_json']}")
        print(f"  * Extracted Vibe:   {res['scores']}")
        print(f"  * Process RAM (RSS):{mem['process_ram_gb']} GB")
        print(f"  * System RAM Used:  {mem['system_ram_used_gb']} GB / {mem['system_ram_total_gb']} GB ({mem['system_ram_percent']}%)")
        print(f"  * GPU VRAM Alloc:   {mem['gpu_vram_allocated_gb']} GB (Reserved: {mem['gpu_vram_reserved_gb']} GB)")
        print(f"  * Raw Generation:   {res['raw_output']}")

        run_entry = {
            "appid": appid,
            "game_name": title,
            "genres": genres,
            "description_snippet": desc[:200] + ("..." if len(desc) > 200 else ""),
            "raw_output": res["raw_output"],
            "parsed_json": res["parsed_json"],
            "is_valid_json": res["is_valid_json"],
            "scores": res["scores"],
            "latency_seconds": res["latency_seconds"],
            "tokens_per_sec": res["tokens_per_sec"],
            "ram_usage": mem
        }
        runs.append(run_entry)

    # Compute aggregate summary
    total = len(runs)
    valid_count = sum(1 for r in runs if r["is_valid_json"])
    avg_latency = sum(r["latency_seconds"] for r in runs) / max(total, 1)
    avg_tok_s = sum(r["tokens_per_sec"] for r in runs) / max(total, 1)
    peak_process_ram = max((r["ram_usage"]["process_ram_gb"] for r in runs), default=0.0)
    peak_gpu_vram = max((r["ram_usage"]["gpu_vram_allocated_gb"] for r in runs), default=0.0)

    summary_results = {
        "model_id": model_id,
        "device": baseline.device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": total,
        "valid_json_count": valid_count,
        "validity_rate": round((valid_count / max(total, 1)) * 100.0, 1),
        "avg_latency_sec": round(avg_latency, 3),
        "avg_tokens_per_sec": round(avg_tok_s, 1),
        "ram_summary": {
            "peak_process_ram_gb": peak_process_ram,
            "peak_gpu_vram_allocated_gb": peak_gpu_vram,
            "latest_system_ram_percent": runs[-1]["ram_usage"]["system_ram_percent"] if runs else 0.0
        },
        "runs": runs
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary_results, f, indent=2)
        print(f"\nSaved inference baseline results JSON to: {output_path}")

    print("\n=======================================================")
    print(f"Inference Baseline completed! Validity Rate: {summary_results['validity_rate']}% | Avg Latency: {summary_results['avg_latency_sec']}s")
    print("=======================================================\n")
    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steam Hit Predictor - LLM Inference Baseline")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-0.8B", help="Hugging Face Model ID")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of qualitative sample games to evaluate")
    parser.add_argument("--output_path", type=str, default="llm-pipeline/baseline_results/inference_baseline_results.json", help="Output JSON path")
    args = parser.parse_args()

    run_inference_baseline(model_id=args.model_id, num_samples=args.num_samples, output_path=args.output_path)
