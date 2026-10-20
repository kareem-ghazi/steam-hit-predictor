import os
import re
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from inference_baseline import extract_json_payload, validate_vibe_scores

load_dotenv()


class PromptStrategy:
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"


def build_zero_shot_prompt(tokenizer, title: str, genres: str, desc: str) -> str:
    system = (
        "You are an AI video game analyst. Evaluate the given game title, genres, and description. "
        "Score the game from 1.0 to 10.0 on three attributes:\n"
        "- hype_score: excitement and virality\n"
        "- story_score: narrative depth\n"
        "- nostalgia_score: classic retro appeal\n\n"
        "Return ONLY a JSON object formatted as:\n"
        '{"hype_score": float, "story_score": float, "nostalgia_score": float}'
    )
    user = f"Title: {title}\nGenres: {genres}\nDescription: {desc[:800]}\nOutput:"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\nUser:\n{user}\n\nAssistant:"


def build_few_shot_prompt(tokenizer, title: str, genres: str, desc: str) -> str:
    system = (
        "You are an AI video game analyst. Evaluate game concepts and return a JSON score object with "
        "hype_score, story_score, and nostalgia_score (each from 1.0 to 10.0)."
    )
    
    # 2 In-context exemplars
    example1_user = (
        "Title: Half-Life\n"
        "Genres: Action;Sci-fi\n"
        "Description: Named Game of the Year by over 50 publications, Valve's debut title blends action and adventure with award-winning technology."
    )
    example1_assistant = '{"hype_score": 8.0, "story_score": 8.5, "nostalgia_score": 9.5}'

    example2_user = (
        "Title: Stardew Valley\n"
        "Genres: Indie;RPG;Simulation\n"
        "Description: You have inherited your grandfather's old farm plot. Armed with hand-me-down tools, can you turn these overgrown fields into a thriving home?"
    )
    example2_assistant = '{"hype_score": 7.5, "story_score": 7.0, "nostalgia_score": 8.5}'

    current_user = f"Title: {title}\nGenres: {genres}\nDescription: {desc[:800]}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": example1_user},
        {"role": "assistant", "content": example1_assistant},
        {"role": "user", "content": example2_user},
        {"role": "assistant", "content": example2_assistant},
        {"role": "user", "content": current_user}
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        f"{system}\n\n"
        f"Example 1:\n{example1_user}\n{example1_assistant}\n\n"
        f"Example 2:\n{example2_user}\n{example2_assistant}\n\n"
        f"Task:\n{current_user}\n\nAssistant:"
    )


def build_cot_prompt(tokenizer, title: str, genres: str, desc: str) -> str:
    system = (
        "You are an AI video game analyst. Analyze the game concept through step-by-step reasoning "
        "before producing the final scores.\n"
        "Reason through:\n"
        "1. Story & Lore (Narrative depth)\n"
        "2. Community & Virality (Hype potential)\n"
        "3. Retro / Heritage (Nostalgia appeal)\n\n"
        "After your reasoning, output the final scores in a ```json code block formatted as:\n"
        "```json\n"
        '{"hype_score": float, "story_score": float, "nostalgia_score": float}\n'
        "```"
    )
    user = f"Title: {title}\nGenres: {genres}\nDescription: {desc[:800]}\nProvide your step-by-step reasoning and final JSON scores:"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\nUser:\n{user}\n\nAssistant:"


class PromptingBaselineEvaluator:
    """
    Evaluates Zero-Shot, Few-Shot, and Chain-of-Thought prompting strategies
    against schema validity, value compliance, and latency.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = os.getenv("HF_TOKEN") or None

        print(f"Loading model '{model_id}' on '{self.device}' for Prompting Baseline Evaluation...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.hf_token)
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            token=self.hf_token
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate_sample(self, prompt: str, max_new_tokens: int = 180) -> Dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        start = time.time()
        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id
        )
        latency = time.time() - start
        
        new_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        parsed = extract_json_payload(raw_output)
        is_valid, scores = validate_vibe_scores(parsed)

        return {
            "raw_output": raw_output,
            "parsed": parsed,
            "is_valid": is_valid,
            "scores": scores,
            "latency": latency
        }

    def run_benchmark(self, benchmark_games: List[Dict[str, str]]) -> Dict[str, Any]:
        strategies = [
            (PromptStrategy.ZERO_SHOT, build_zero_shot_prompt, 100),
            (PromptStrategy.FEW_SHOT, build_few_shot_prompt, 100),
            (PromptStrategy.CHAIN_OF_THOUGHT, build_cot_prompt, 200),
        ]

        results = {}

        for strat_name, prompt_fn, max_tokens in strategies:
            print(f"\nEvaluating Prompt Strategy: [{strat_name.upper()}] ...")
            strat_runs = []

            for g in benchmark_games:
                prompt = prompt_fn(self.tokenizer, g["name"], g["genres"], g["description"])
                eval_res = self.evaluate_sample(prompt, max_new_tokens=max_tokens)
                eval_res["game"] = g["name"]
                strat_runs.append(eval_res)

            total = len(strat_runs)
            valid_count = sum(1 for r in strat_runs if r["is_valid"])
            avg_lat = sum(r["latency"] for r in strat_runs) / max(total, 1)

            results[strat_name] = {
                "total_samples": total,
                "valid_json_count": valid_count,
                "validity_rate": round((valid_count / total) * 100.0, 1),
                "avg_latency_sec": round(avg_lat, 3),
                "runs": strat_runs
            }

        return results


def run_prompting_baseline(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_path: str = "llm-pipeline/prompting_baseline_results.json"
):
    print("Loading benchmark games from Steam dataset...")
    df_steam = pd.read_csv("raw_data/steam.csv")
    df_desc = pd.read_csv("raw_data/steam_description_data.csv")
    merged = pd.merge(df_steam, df_desc, left_on="appid", right_on="steam_appid")

    # Select 5 representative games across genres
    target_appids = [10, 400, 570, 1500, 1250]  # CS, Portal, Dota 2, Darwinia, Killing Floor
    samples = merged[merged["appid"].isin(target_appids)]
    
    benchmark_games = []
    for _, row in samples.iterrows():
        benchmark_games.append({
            "name": row["name"],
            "genres": row["genres"],
            "description": row.get("short_description") or row.get("detailed_description") or ""
        })

    evaluator = PromptingBaselineEvaluator(model_id=model_id)
    benchmark_results = evaluator.run_benchmark(benchmark_games)

    print("\n=======================================================")
    print("       Prompting Baseline Comparison Results           ")
    print("=======================================================")
    print(f"{'Strategy':<20} | {'Validity Rate':<15} | {'Avg Latency (s)':<15}")
    print("-" * 56)
    for strat, data in benchmark_results.items():
        print(f"{strat:<20} | {data['validity_rate']}% ({data['valid_json_count']}/{data['total_samples']}) | {data['avg_latency_sec']}s")
    print("=======================================================\n")

    # Save results to json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Results recorded to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steam Hit Predictor - Prompting Baseline Evaluation")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face Model ID")
    parser.add_argument("--output_path", type=str, default="llm-pipeline/prompting_baseline_results.json", help="Output path")
    args = parser.parse_args()

    run_prompting_baseline(model_id=args.model_id, output_path=args.output_path)
