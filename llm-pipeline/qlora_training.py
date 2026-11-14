#!/usr/bin/env python3
"""
Steam Hit Predictor - Phase 3: 8-Bit QLoRA Fine-Tuning Pipeline
Fine-tunes Qwen3.5-0.8B to evaluate video game concepts and predict vibe scores
(hype_score, story_score, nostalgia_score) from game descriptions.

Optimized for execution both locally and directly inside Kaggle Notebooks (GPU T4/P100/A100).
Supports direct input from Kaggle dataset paths (e.g. /kaggle/input/<dataset-slug>).
"""

import os
import sys
import re
import json
import time
import argparse
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Load environment variables (.env) if available
for p in [os.path.join(PARENT_DIR, ".env"), os.path.join(CURRENT_DIR, ".env"), ".env"]:
    if os.path.exists(p):
        load_dotenv(p)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)


def is_running_on_kaggle() -> bool:
    """Returns True if the script is running inside a Kaggle notebook or batch environment."""
    return (
        os.path.exists("/kaggle/working")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_URL_BASE" in os.environ
    )


def resolve_data_dir(data_dir: str = "cleaned_data") -> str:
    """
    Resolves data directory path whether executed locally or inside a Kaggle notebook.
    Handles:
    1. Direct paths (e.g. '/kaggle/input/steam-hit-predictor-cleaned-data' or 'cleaned_data')
    2. Kaggle parent input folder ('/kaggle/input')
    3. Kaggle dataset handles (e.g. 'kareemghazi/steam-hit-predictor-cleaned-data')
    4. Auto-discovery across mounted datasets in /kaggle/input/
    """
    data_dir_str = str(data_dir).strip()

    # Case 1: If data_dir is a Kaggle dataset handle ("<owner>/<slug>")
    if "/" in data_dir_str and not os.path.exists(data_dir_str) and not data_dir_str.startswith(("/", "\\", ".")):
        parts = data_dir_str.split("/")
        if len(parts) == 2 and not any(p in parts[0] for p in [":", "\\"]):
            slug = parts[1]
            candidate = f"/kaggle/input/{slug}"
            if os.path.exists(candidate):
                data_dir_str = candidate

    # Case 2: Direct path check (e.g. /kaggle/input/<slug> or local cleaned_data)
    if os.path.exists(data_dir_str):
        # Direct check for train/games.csv
        if os.path.exists(os.path.join(data_dir_str, "train", "games.csv")):
            return data_dir_str if data_dir_str.startswith("/kaggle") else os.path.abspath(data_dir_str)

        # Search inside data_dir for split subdirectories
        for root, dirs, files in os.walk(data_dir_str):
            if "games.csv" in files and os.path.basename(root) in ["train", "valid", "test"]:
                split_dir = os.path.dirname(root)
                return split_dir if split_dir.startswith("/kaggle") else os.path.abspath(split_dir)
            if os.path.basename(root) == "cleaned_data" and any(d in dirs for d in ["train", "test"]):
                return root if root.startswith("/kaggle") else os.path.abspath(root)

    # Case 3: If running on Kaggle, scan /kaggle/input for mounted datasets
    if is_running_on_kaggle():
        kaggle_input = "/kaggle/input"
        if os.path.exists(kaggle_input):
            for root, dirs, files in os.walk(kaggle_input):
                if "games.csv" in files and os.path.basename(root) in ["train", "valid", "test"]:
                    found_dir = os.path.dirname(root)
                    print(f"Auto-detected Kaggle dataset at: '{found_dir}'")
                    return found_dir
                if os.path.basename(root) == "cleaned_data" and any(d in dirs for d in ["train", "test"]):
                    return root

    # Case 4: Local filesystem fallbacks
    candidates = [
        data_dir_str,
        os.path.join(PARENT_DIR, data_dir_str),
        os.path.join(CURRENT_DIR, "..", data_dir_str),
        os.path.join(CURRENT_DIR, data_dir_str),
        "cleaned_data",
        os.path.join(PARENT_DIR, "cleaned_data"),
    ]
    for c in candidates:
        if os.path.exists(c):
            if os.path.exists(os.path.join(c, "train", "games.csv")):
                return os.path.abspath(c)
            for root, dirs, files in os.walk(c):
                if "games.csv" in files and os.path.basename(root) in ["train", "valid", "test"]:
                    return os.path.abspath(os.path.dirname(root))

    return data_dir_str if data_dir_str.startswith("/kaggle") else os.path.abspath(data_dir_str)


def resolve_output_dir(output_dir: str = "saved_models/llm_adapters/qwen_steam_vibe_8bit") -> str:
    """Resolves output directory path. On Kaggle, routes to /kaggle/working to ensure write access."""
    if is_running_on_kaggle():
        if not output_dir.startswith("/kaggle/working"):
            basename = os.path.basename(output_dir.rstrip("/\\"))
            return os.path.join("/kaggle/working", basename)
        return output_dir
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.abspath(os.path.join(PARENT_DIR, output_dir))


def load_cleaned_data_split(
    data_dir: str = "cleaned_data",
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Loads games and descriptions from cleaned_data/<split>/ (games.csv + games_descriptions.csv).
    Merges on appid, validates vibe scores, and returns clean paired DataFrame.
    """
    resolved_dir = resolve_data_dir(data_dir)
    games_path = os.path.join(resolved_dir, split, "games.csv")
    desc_path = os.path.join(resolved_dir, split, "games_descriptions.csv")

    if not os.path.exists(games_path) or not os.path.exists(desc_path):
        raise FileNotFoundError(
            f"Split '{split}' files not found in '{resolved_dir}'. "
            f"Expected '{games_path}' and '{desc_path}'."
        )

    df_g = pd.read_csv(games_path)
    df_d = pd.read_csv(desc_path)
    merged = pd.merge(df_g, df_d, on="appid", how="inner")

    # Fallback to short_description if clean_description is missing
    if "clean_description" not in merged.columns or merged["clean_description"].isna().all():
        merged["clean_description"] = merged.get("short_description", "")
    merged["clean_description"] = merged["clean_description"].fillna("").astype(str).str.strip()

    # Filter out empty descriptions
    merged = merged[merged["clean_description"].str.len() >= 20].copy()

    # Ensure vibe scores are numeric
    for col in ["hype_score", "story_score", "nostalgia_score"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(5.0)

    # Subsample if max_samples is explicitly specified (default: None for full dataset)
    if max_samples and max_samples < len(merged):
        merged = merged.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    print(f"Loaded '{split}' split from '{resolved_dir}': {len(merged):,} games.")
    return merged


def build_system_instruction() -> str:
    return (
        "You are an expert video game industry analyst. Your task is to evaluate a game's concept "
        "and description, then output three vibe scores on a scale from 1.0 to 10.0:\n"
        "1. hype_score: anticipated community excitement, viral potential, or multiplayer buzz.\n"
        "2. story_score: narrative depth, lore, worldbuilding, or emotional engagement.\n"
        "3. nostalgia_score: retro aesthetic, classic genre homage, or legacy franchise appeal.\n\n"
        "You MUST respond ONLY with a valid JSON object matching this exact schema:\n"
        '{\n  "hype_score": float,\n  "story_score": float,\n  "nostalgia_score": float\n}\n'
        "Do NOT include markdown formatting, code fences, or any explanation outside the JSON."
    )


def format_vibe_prompt(
    game_name: str,
    genres: str,
    description: str,
    target_scores: Optional[Dict[str, float]] = None
) -> str:
    """Formats prompt following Qwen chat template."""
    system_inst = build_system_instruction()
    user_content = (
        f"Game: {game_name}\n"
        f"Genres: {genres}\n"
        f"Description:\n{description.strip()}"
    )

    prompt = (
        f"<|im_start|>system\n{system_inst}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if target_scores is not None:
        target_json = json.dumps({
            "hype_score": round(float(target_scores["hype_score"]), 1),
            "story_score": round(float(target_scores["story_score"]), 1),
            "nostalgia_score": round(float(target_scores["nostalgia_score"]), 1)
        })
        prompt += f"{target_json}<|im_end|>"

    return prompt


def parse_vibe_scores_from_text(raw_text: str) -> Optional[Dict[str, float]]:
    """Extracts and validates JSON vibe scores from model generation."""
    clean = re.sub(r"<\|im_end\|>.*", "", raw_text, flags=re.DOTALL).strip()
    match = re.search(r"\{[^{}]*\}", clean)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
        required_keys = ["hype_score", "story_score", "nostalgia_score"]
        if not all(k in data for k in required_keys):
            return None

        result = {}
        for k in required_keys:
            val = float(data[k])
            val = max(1.0, min(10.0, val))
            result[k] = round(val, 2)
        return result
    except Exception:
        return None


def prepare_hf_training_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> Dataset:
    """Vectorized pre-tokenization and formatting using Hugging Face Dataset.map."""
    print(f"Preparing HF Dataset ({len(df)} samples) with parallel .map()...")

    records = {
        "name": df["name"].fillna("Unknown Game").tolist(),
        "genres": df["genres"].fillna("Indie").tolist(),
        "clean_description": df["clean_description"].fillna("").tolist(),
        "hype_score": df["hype_score"].tolist(),
        "story_score": df["story_score"].tolist(),
        "nostalgia_score": df["nostalgia_score"].tolist(),
    }
    raw_dataset = Dataset.from_dict(records)

    def _batch_format_prompts(batch):
        formatted_texts = []
        for name, genres, desc, hype, story, nost in zip(
            batch["name"], batch["genres"], batch["clean_description"],
            batch["hype_score"], batch["story_score"], batch["nostalgia_score"]
        ):
            target = {
                "hype_score": float(hype),
                "story_score": float(story),
                "nostalgia_score": float(nost)
            }
            formatted_texts.append(
                format_vibe_prompt(
                    game_name=str(name),
                    genres=str(genres),
                    description=str(desc),
                    target_scores=target
                )
            )
        return {"text": formatted_texts}

    num_procs = min(4, os.cpu_count() or 1)
    mapped_dataset = raw_dataset.map(
        _batch_format_prompts,
        batched=True,
        batch_size=500,
        num_proc=num_procs,
        remove_columns=raw_dataset.column_names,
        desc="Vectorizing Training Prompts"
    )

    print(f"HF Dataset ready: {len(mapped_dataset)} samples.")
    return mapped_dataset


def evaluate_model_on_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    df_eval: pd.DataFrame,
    device: str = "cuda",
    top_p: float = 0.9,
    temperature: float = 0.7,
    eval_batch_size: int = 4,
    use_compile: bool = False,
    label: str = "Evaluation"
) -> Dict[str, Any]:
    """Batched evaluation on holdout dataset measuring validity, MAE, and latency."""
    print(f"\n--- Running Batched Evaluation: {label} (N={len(df_eval)}, batch_size={eval_batch_size}) ---")

    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    eval_model = model
    if use_compile and hasattr(torch, "compile") and device == "cuda":
        try:
            print("Accelerating evaluation with torch.compile()...")
            eval_model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile note: {e}, falling back to standard execution.")
            eval_model = model

    valid_count = 0
    errors = []
    latencies = []
    evaluated_games = []
    total_samples = len(df_eval)

    eval_model.eval()
    num_batches = int(np.ceil(total_samples / eval_batch_size))

    for b in tqdm(range(num_batches), desc=f"Evaluating ({label})", unit="batch"):
        start_idx = b * eval_batch_size
        end_idx = min(start_idx + eval_batch_size, total_samples)
        batch = df_eval.iloc[start_idx:end_idx]

        prompts = [
            format_vibe_prompt(
                game_name=str(row.get("name", "Unknown Game")),
                genres=str(row.get("genres", "Indie")),
                description=str(row.get("clean_description", "")),
                target_scores=None
            )
            for _, row in batch.iterrows()
        ]

        batch_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        batch_t0 = time.time()
        with torch.no_grad():
            outputs = eval_model.generate(
                **batch_inputs,
                max_new_tokens=80,
                top_p=top_p,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        batch_duration = time.time() - batch_t0
        per_sample_latency = batch_duration / len(batch)

        for idx, (out_ids, in_ids) in enumerate(zip(outputs, batch_inputs.input_ids)):
            gen_tokens = out_ids[len(in_ids):]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            row = batch.iloc[idx]
            ground_truth = {
                "hype_score": float(row["hype_score"]),
                "story_score": float(row["story_score"]),
                "nostalgia_score": float(row["nostalgia_score"])
            }
            parsed = parse_vibe_scores_from_text(gen_text)
            is_valid = parsed is not None

            if is_valid:
                valid_count += 1
                for k in ["hype_score", "story_score", "nostalgia_score"]:
                    errors.append(abs(parsed[k] - ground_truth[k]))

            latencies.append(per_sample_latency)
            evaluated_games.append({
                "appid": int(row.get("appid", 0)),
                "name": str(row.get("name", "Unknown")),
                "is_valid": is_valid,
                "ground_truth": ground_truth,
                "predicted": parsed
            })

    tokenizer.padding_side = orig_pad_side

    mae = float(np.mean(errors)) if errors else 0.0
    acc_pct = float(np.mean([e <= 2.0 for e in errors]) * 100.0) if errors else 0.0
    validity_rate = (valid_count / total_samples) * 100.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    print(f"Results for {label}:")
    print(f"  * JSON Validity Rate: {validity_rate:.1f}% ({valid_count}/{total_samples})")
    print(f"  * Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  * Accuracy (+/- 2.0 tolerance): {acc_pct:.1f}%")
    print(f"  * Avg Latency: {avg_latency:.3f}s / game")

    return {
        "label": label,
        "total_samples": total_samples,
        "valid_count": valid_count,
        "validity_rate_pct": round(validity_rate, 2),
        "mae": round(mae, 2),
        "accuracy_within_tolerance_pct": round(acc_pct, 2),
        "avg_latency_sec": round(avg_latency, 3),
        "predictions": evaluated_games[:5]
    }


def run_qlora_training(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    data_dir: str = "cleaned_data",
    output_dir: str = "saved_models/llm_adapters/qwen_steam_vibe_8bit",
    max_train_samples: Optional[int] = None,
    max_eval_samples: int = 20,
    eval_batch_size: int = 4,
    use_compile: bool = True,
    epochs: int = 5,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    top_p: float = 0.9,
    temperature: float = 0.7,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes the 8-bit QLoRA fine-tuning pipeline on GPU:
    Trains on the entire training set by default across 5 epochs.
    """
    hf_token = os.getenv("HF_TOKEN") or None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_output = resolve_output_dir(output_dir)
    os.makedirs(resolved_output, exist_ok=True)

    use_bf16 = False
    use_fp16 = True if torch.cuda.is_available() else False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not is_running_on_kaggle():
        use_bf16 = True
        use_fp16 = False
        precision_desc = "bf16=True (bfloat16 accelerated)"
    else:
        precision_desc = "fp16=True (float16 accelerated)" if use_fp16 else "CPU"

    sample_desc = f"{max_train_samples} samples" if max_train_samples else "Full Dataset (all samples)"

    print("==================================================================")
    print("      STEAM HIT PREDICTOR - PHASE 3: 8-BIT QLORA FINE-TUNING      ")
    print("==================================================================")
    print(f"Base Model:             {model_id}")
    print(f"Quantization:           BitsAndBytes 8-bit (load_in_8bit=True)")
    print(f"Mixed Precision:        {precision_desc}")
    print(f"Training Epochs:        {epochs}")
    print(f"Training Dataset:       {sample_desc} from {data_dir}")
    print(f"Sampling Parameters:    top_p={top_p}, temperature={temperature}")
    print(f"Target Output Adapter:  {resolved_output}")
    print("==================================================================\n")

    # 1. Load Data Splits
    df_train = load_cleaned_data_split(data_dir=data_dir, split="train", max_samples=max_train_samples, seed=seed)
    df_eval = load_cleaned_data_split(data_dir=data_dir, split="test", max_samples=max_eval_samples, seed=seed)

    # 2. Tokenizer Setup
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. 8-bit Quantization Config (BitsAndBytes)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Loading base model '{model_id}' in 8-bit quantization on {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=model_dtype,
        device_map="auto" if device == "cuda" else None,
        token=hf_token
    )

    if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
        base_model.generation_config.top_p = top_p
        base_model.generation_config.temperature = temperature
        base_model.generation_config.do_sample = True

    # 4. Before Evaluation (Base Model)
    before_metrics = evaluate_model_on_dataset(
        model=base_model,
        tokenizer=tokenizer,
        df_eval=df_eval,
        device=device,
        top_p=top_p,
        temperature=temperature,
        eval_batch_size=eval_batch_size,
        use_compile=use_compile and not is_running_on_kaggle(),
        label="Before Fine-Tuning (Base Model)"
    )

    # 5. Prepare Model for LoRA Training
    print("\nPreparing base model for 8-bit LoRA training...")
    base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(base_model, peft_config)
    peft_model.print_trainable_parameters()

    # 6. Build Vectorized Training Dataset
    train_hf_ds = prepare_hf_training_dataset(df_train, tokenizer)

    # 7. SFTConfig & SFTTrainer
    tmp_ckpt_dir = os.path.join(resolved_output, "_checkpoints")
    os.makedirs(tmp_ckpt_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=tmp_ckpt_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        logging_steps=20,
        save_strategy="no",
        fp16=use_fp16,
        bf16=use_bf16,
        max_length=512,
        dataset_text_field="text",
        report_to="none",
        seed=seed
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=train_hf_ds,
        args=sft_config,
        processing_class=tokenizer,
    )

    print(f"\nStarting QLoRA 8-bit fine-tuning ({epochs} epochs on {len(df_train)} games)...")
    train_start_t = time.time()
    train_result = trainer.train()
    training_duration = time.time() - train_start_t
    train_loss = train_result.training_loss
    print(f"Fine-tuning complete in {training_duration:.2f}s! Final Training Loss: {train_loss:.4f}")

    # 8. Save Adapter & Tokenizer
    print(f"\nSaving fine-tuned LoRA adapter to: '{resolved_output}'...")
    trainer.model.save_pretrained(resolved_output)
    tokenizer.save_pretrained(resolved_output)

    # Also persist to parent adapter folder if running locally
    if not is_running_on_kaggle():
        root_adapter_dir = os.path.dirname(resolved_output)
        trainer.model.save_pretrained(root_adapter_dir)
        print(f"Saved primary adapter artifacts to: '{root_adapter_dir}' and '{resolved_output}'")

    # 9. After Evaluation
    after_metrics = evaluate_model_on_dataset(
        model=trainer.model,
        tokenizer=tokenizer,
        df_eval=df_eval,
        device=device,
        top_p=top_p,
        temperature=temperature,
        eval_batch_size=eval_batch_size,
        use_compile=use_compile and not is_running_on_kaggle(),
        label="After Fine-Tuning (QLoRA 8-Bit Adapter)"
    )

    # 10. Summary Report
    mae_diff = before_metrics["mae"] - after_metrics["mae"]
    acc_diff = after_metrics["accuracy_within_tolerance_pct"] - before_metrics["accuracy_within_tolerance_pct"]

    print("\n==================================================================")
    print("           QLORA 8-BIT BEFORE VS AFTER IMPROVEMENT REPORT         ")
    print("==================================================================")
    print(f"{'Metric':<32} | {'Before (Base)':<15} | {'After (QLoRA)':<15} | {'Improvement'}")
    print("-" * 75)
    print(f"{'JSON Schema Validity':<32} | {before_metrics['validity_rate_pct']:>13.1f}% | {after_metrics['validity_rate_pct']:>13.1f}% | {after_metrics['validity_rate_pct'] - before_metrics['validity_rate_pct']:>+10.1f}%")
    print(f"{'Mean Absolute Error (MAE)':<32} | {before_metrics['mae']:>15.2f} | {after_metrics['mae']:>15.2f} | {-mae_diff:>+11.2f} (lower is better)")
    print(f"{'Accuracy (+/- 2.0 tolerance)':<32} | {before_metrics['accuracy_within_tolerance_pct']:>13.1f}% | {after_metrics['accuracy_within_tolerance_pct']:>13.1f}% | {acc_diff:>+10.1f}%")
    print("==================================================================\n")

    summary = {
        "model_id": model_id,
        "quantization": "8-bit BitsAndBytes",
        "precision": precision_desc,
        "training_epochs": epochs,
        "training_samples": len(df_train),
        "evaluation_samples": len(df_eval),
        "final_train_loss": round(float(train_loss), 4),
        "training_duration_sec": round(training_duration, 2),
        "adapter_save_path": resolved_output,
        "before_finetuning": before_metrics,
        "after_finetuning": after_metrics,
        "improvement": {
            "mae_reduction": round(float(mae_diff), 2),
            "accuracy_gain_pct": round(float(acc_diff), 2)
        }
    }

    # Save summary report to baseline_results
    results_dir = os.path.join(
        "/kaggle/working" if is_running_on_kaggle() else CURRENT_DIR,
        "baseline_results"
    )
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "qlora_training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved complete evaluation report to: '{results_path}'")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Steam Hit Predictor - Phase 3: 8-bit QLoRA Fine-Tuning"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="cleaned_data",
        help="Path to dataset directory (e.g. 'cleaned_data' or '/kaggle/input/<dataset-slug>')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="saved_models/llm_adapters/qwen_steam_vibe_8bit",
        help="Directory where trained adapter weights and tokenizer will be saved"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of fine-tuning epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train batch size (default: 2)"
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Maximum training samples to use (default: None for full training set)"
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=20,
        help="Number of holdout test samples for before/after evaluation (default: 20)"
    )

    args = parser.parse_args()

    run_qlora_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples
    )


if __name__ == "__main__":
    main()
