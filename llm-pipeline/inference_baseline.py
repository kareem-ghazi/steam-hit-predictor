import os
import re
import json
import time
import argparse
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables (such as HF_TOKEN)
load_dotenv()


def extract_json_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses JSON object from model output, handling
    markdown backticks, preamble text, or trailing commentary.
    """
    # 1. Look for ```json ... ``` code blocks
    block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if block_match:
        try:
            return json.loads(block_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Look for any top-level JSON object {...}
    obj_match = re.search(r"(\{[\s\S]*\})", raw_text)
    if obj_match:
        try:
            return json.loads(obj_match.group(1))
        except json.JSONDecodeError:
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
    Establishes a working inference baseline with a local instruction-tuned LLM
    to extract game vibe scores (hype, story, nostalgia) from descriptions.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: Optional[str] = None):
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
            '  "nostalgia_score": float,\n'
            '  "reasoning": "brief 1-2 sentence explanation"\n'
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

        parsed_json = extract_json_payload(raw_output)
        is_valid, validated_scores = validate_vibe_scores(parsed_json)

        return {
            "game_name": game_name,
            "raw_output": raw_output,
            "parsed_json": parsed_json,
            "is_valid_json": is_valid,
            "scores": validated_scores,
            "latency_seconds": round(elapsed, 3),
            "tokens_per_sec": round(tokens_per_sec, 1)
        }


def run_inference_baseline(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_samples: int = 3
):
    """
    Runs qualitative baseline inference on real Steam dataset examples.
    """
    print(f"\n=======================================================")
    print(f"   Steam Hit Predictor: LLM Inference Baseline        ")
    print(f"=======================================================\n")

    # Load Steam games and descriptions
    steam_path = "raw_data/steam.csv"
    desc_path = "raw_data/steam_description_data.csv"

    if not (os.path.exists(steam_path) and os.path.exists(desc_path)):
        raise FileNotFoundError(f"Missing {steam_path} or {desc_path} in raw_data/")

    df_steam = pd.read_csv(steam_path)
    df_desc = pd.read_csv(desc_path)
    df_merged = pd.merge(df_steam, df_desc, left_on="appid", right_on="steam_appid")

    baseline = SteamLLMInferenceBaseline(model_id=model_id)

    # Pick diverse sample games across genres
    sample_appids = [10, 400, 570]  # Counter-Strike, Portal, Dota 2
    samples = df_merged[df_merged["appid"].isin(sample_appids)]
    
    if len(samples) < num_samples:
        samples = df_merged.head(num_samples)

    print("\n--- Running Baseline Inference on Steam Sample Games ---")
    for idx, row in samples.iterrows():
        title = row["name"]
        genres = row["genres"]
        desc = row.get("short_description") or row.get("detailed_description") or ""
        
        print(f"\nEvaluating Game: '{title}' ({genres})")
        res = baseline.predict(title, genres, desc)
        
        print(f"  * Latency:        {res['latency_seconds']}s ({res['tokens_per_sec']} tok/s)")
        print(f"  * Valid JSON:     {res['is_valid_json']}")
        print(f"  * Extracted Vibe: {res['scores']}")
        print(f"  * Raw Generation: {res['raw_output']}")

    print("\n=======================================================")
    print("Inference Baseline completed successfully!")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steam Hit Predictor - LLM Inference Baseline")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face Model ID")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of qualitative sample games to evaluate")
    args = parser.parse_args()

    run_inference_baseline(model_id=args.model_id, num_samples=args.num_samples)
