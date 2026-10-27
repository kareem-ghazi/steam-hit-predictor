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
import logging

from inference_baseline import (
    extract_json_payload,
    validate_vibe_scores,
    get_current_ram_usage,
    load_cleaned_data,
    resolve_data_dir,
    resolve_output_path
)

load_dotenv()
logging.getLogger("transformers").setLevel(logging.ERROR)

class PromptStrategy:
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"



def calculate_accuracy(pred_scores: Dict[str, float], ground_truth: Dict[str, float], tolerance: float = 2.0) -> Tuple[float, float]:
    """
    Computes:
    1. Mean Absolute Error (MAE) across the 3 scores.
    2. Accuracy (% of predicted scores within tolerance, e.g. +/- 2.0 of ground truth).
    """
    if not pred_scores or not ground_truth:
        return 5.0, 0.0

    errors = []
    correct_count = 0
    keys = ["hype_score", "story_score", "nostalgia_score"]

    for k in keys:
        if k in pred_scores and k in ground_truth:
            err = abs(pred_scores[k] - ground_truth[k])
            errors.append(err)
            if err <= tolerance:
                correct_count += 1

    mae = sum(errors) / max(len(errors), 1)
    accuracy_pct = (correct_count / len(keys)) * 100.0 if keys else 0.0
    return round(mae, 2), round(accuracy_pct, 1)


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
    
    example1_user = (
        "Title: Half-Life\n"
        "Genres: action\n"
        "Description: Named Game of the Year by over 50 publications, Valve's debut title blends action and adventure with award-winning technology to create a frighteningly realistic world where players must think to survive."
    )
    example1_assistant = '{"hype_score": 10.0, "story_score": 5.1, "nostalgia_score": 10.0}'

    example2_user = (
        "Title: Stardew Valley\n"
        "Genres: indie;rpg;simulation\n"
        "Description: You've inherited your grandfather's old farm plot in Stardew Valley. Armed with hand-me-down tools and a few coins, you set out to begin your new life. Can you learn to live off the land and turn these overgrown fields into a thriving home?"
    )
    example2_assistant = '{"hype_score": 9.3, "story_score": 7.1, "nostalgia_score": 6.2}'

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
        "You are an AI video game analyst. Analyze the game concept through brief step-by-step reasoning "
        "(1 concise sentence each for story, hype, and nostalgia), followed immediately by the final JSON scores "
        "(each 1.0 to 10.0).\n\n"
        "Output format:\n"
        "Reasoning:\n"
        "- Story: <1 sentence analysis>\n"
        "- Hype: <1 sentence analysis>\n"
        "- Nostalgia: <1 sentence analysis>\n"
        "```json\n"
        '{"hype_score": float, "story_score": float, "nostalgia_score": float}\n'
        "```"
    )
    
    example_user = (
        "Title: Half-Life\n"
        "Genres: action\n"
        "Description: Named Game of the Year by over 50 publications, Valve's debut title blends action and adventure with award-winning technology to create a frighteningly realistic world where players must think to survive."
    )
    example_assistant = (
        "Reasoning:\n"
        "- Story: Immersive environmental storytelling following Gordon Freeman's survival in Black Mesa.\n"
        "- Hype: Historic industry-defining FPS release with massive enduring global acclaim.\n"
        "- Nostalgia: Foundational 1998 classic revered as one of the greatest PC games of all time.\n"
        "```json\n"
        '{"hype_score": 10.0, "story_score": 5.1, "nostalgia_score": 10.0}\n'
        "```"
    )
    
    current_user = f"Title: {title}\nGenres: {genres}\nDescription: {desc[:800]}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": example_assistant},
        {"role": "user", "content": current_user}
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        f"{system}\n\n"
        f"Example:\nUser: {example_user}\nAssistant:\n{example_assistant}\n\n"
        f"User: {current_user}\nAssistant:"
    )


class PromptingBaselineEvaluator:
    """
    Evaluates Zero-Shot, Few-Shot, and Chain-of-Thought prompting strategies
    against schema validity, ground-truth accuracy, and latency.
    """
    def __init__(self, model_id: str = "Qwen/Qwen3.5-0.8B"):
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
            (PromptStrategy.CHAIN_OF_THOUGHT, build_cot_prompt, 220),
        ]

        results = {}

        for strat_name, prompt_fn, max_tokens in strategies:
            print(f"\nEvaluating Prompt Strategy: [{strat_name.upper()}] ...")
            strat_runs = []
            total_accuracy = 0.0
            total_mae = 0.0

            for g in benchmark_games:
                appid = g.get("appid")
                name = g["name"]
                prompt = prompt_fn(self.tokenizer, name, g["genres"], g["description"])
                eval_res = self.evaluate_sample(prompt, max_new_tokens=max_tokens)
                eval_res["appid"] = appid
                eval_res["game"] = name
                
                # Accuracy calculation against ground truth from dataset
                ref_target = g.get("ground_truth", {})
                mae, accuracy_pct = calculate_accuracy(eval_res["scores"], ref_target)
                eval_res["ground_truth"] = ref_target
                eval_res["mae"] = mae
                eval_res["accuracy_pct"] = accuracy_pct
                
                total_accuracy += accuracy_pct
                total_mae += mae
                strat_runs.append(eval_res)

            total = len(strat_runs)
            valid_count = sum(1 for r in strat_runs if r["is_valid"])
            avg_lat = sum(r["latency"] for r in strat_runs) / max(total, 1)
            mean_accuracy = total_accuracy / max(total, 1)
            mean_mae = total_mae / max(total, 1)

            results[strat_name] = {
                "total_samples": total,
                "valid_json_count": valid_count,
                "validity_rate": round((valid_count / total) * 100.0, 1),
                "accuracy_rate": round(mean_accuracy, 1),
                "mean_absolute_error": round(mean_mae, 2),
                "avg_latency_sec": round(avg_lat, 3),
                "runs": strat_runs
            }

        return results


def run_prompting_baseline(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    num_samples: int = 5,
    data_dir: str = "cleaned_data",
    output_path: str = "llm-pipeline/baseline_results/prompting_baseline_results.json"
):
    resolved_data_dir = resolve_data_dir(data_dir)
    resolved_output_path = resolve_output_path(output_path)

    print(f"Loading benchmark games from Steam cleaned dataset at '{resolved_data_dir}'...")
    merged = load_cleaned_data(data_dir=resolved_data_dir)
    print(f"Total games loaded: {len(merged):,}")

    # Pick sample games across genres
    sample_appids = [10, 400, 570]  # Counter-Strike, Portal, Dota 2
    samples = merged[merged["appid"].isin(sample_appids)]
    
    if len(samples) < num_samples:
        samples = merged.head(num_samples)
    else:
        samples = samples.head(num_samples)

    benchmark_games = []
    for _, row in samples.iterrows():
        appid = int(row["appid"])
        gt = {}
        if "hype_score" in row and pd.notnull(row["hype_score"]):
            gt = {
                "hype_score": float(row["hype_score"]),
                "story_score": float(row["story_score"]),
                "nostalgia_score": float(row["nostalgia_score"])
            }

        benchmark_games.append({
            "appid": appid,
            "name": row["name"],
            "genres": str(row.get("genres", "")),
            "description": str(row.get("clean_description") or row.get("short_description") or ""),
            "ground_truth": gt
        })

    evaluator = PromptingBaselineEvaluator(model_id=model_id)
    benchmark_results = evaluator.run_benchmark(benchmark_games)

    print("\n=========================================================================================")
    print("                   Prompting Baseline Comparison Results (Accuracy + Validity)           ")
    print("=========================================================================================")
    print(f"{'Strategy':<18} | {'Validity Rate':<15} | {'Accuracy Rate (±2)':<20} | {'MAE':<8} | {'Avg Latency (s)':<15}")
    print("-" * 89)
    for strat, data in benchmark_results.items():
        print(f"{strat:<18} | {data['validity_rate']}% ({data['valid_json_count']}/{data['total_samples']}) | {data['accuracy_rate']}% | {data['mean_absolute_error']} | {data['avg_latency_sec']}s")
    print("=========================================================================================\n")

    os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)
    with open(resolved_output_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Results recorded to: {resolved_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steam Hit Predictor - Prompting Baseline Evaluation")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-0.8B", help="Hugging Face Model ID")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of qualitative sample games to evaluate")
    parser.add_argument("--data_dir", type=str, default="cleaned_data", help="Path to cleaned_data directory")
    parser.add_argument("--output_path", type=str, default="llm-pipeline/baseline_results/prompting_baseline_results.json", help="Output path")
    args = parser.parse_args()

    run_prompting_baseline(
        model_id=args.model_id,
        num_samples=args.num_samples,
        data_dir=args.data_dir,
        output_path=args.output_path
    )
