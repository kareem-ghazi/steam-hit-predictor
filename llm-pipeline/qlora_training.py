import os
import sys
import re
import json
import time
import psutil
import argparse
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

load_dotenv()
logging.getLogger("transformers").setLevel(logging.ERROR)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


def resolve_data_dir(data_dir: str = "cleaned_data") -> str:
    """Resolves data directory path whether executed from root or llm-pipeline/."""
    candidates = [
        data_dir,
        os.path.join(PARENT_DIR, data_dir),
        os.path.join(CURRENT_DIR, "..", data_dir),
    ]
    return next((os.path.abspath(c) for c in candidates if os.path.exists(c)), os.path.abspath(data_dir))


def resolve_output_dir(output_dir: str) -> str:
    """Resolves output directory path relative to project structure."""
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

    # Use clean_description if present, otherwise fallback to short_description
    if "clean_description" not in merged.columns or merged["clean_description"].isna().all():
        merged["clean_description"] = merged.get("short_description", "")
    merged["clean_description"] = merged["clean_description"].fillna("").astype(str).str.strip()

    # Filter out empty descriptions
    merged = merged[merged["clean_description"].str.len() >= 20].copy()

    # Ensure vibe scores are numeric
    for col in ["hype_score", "story_score", "nostalgia_score"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(5.0)

    if max_samples and max_samples < len(merged):
        merged = merged.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    print(f"Loaded '{split}' split: {len(merged):,} games paired with descriptions.")
    return merged


def build_system_instruction() -> str:
    return (
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


def format_instruction_prompt(
    tokenizer: Any,
    game_name: str,
    genres: str,
    description: str,
    target_scores: Optional[Dict[str, float]] = None
) -> str:
    """Formats prompt using the tokenizer chat template with optional ground-truth target JSON."""
    system_instruction = build_system_instruction()
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

    if target_scores is not None:
        assistant_content = json.dumps({
            "hype_score": round(float(target_scores["hype_score"]), 1),
            "story_score": round(float(target_scores["story_score"]), 1),
            "nostalgia_score": round(float(target_scores["nostalgia_score"]), 1)
        }, indent=2)
        messages.append({"role": "assistant", "content": assistant_content})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(target_scores is None)
        )
    else:
        if target_scores is not None:
            return f"{system_instruction}\n\nUser: {user_content}\n\nAssistant: {assistant_content}"
        return f"{system_instruction}\n\nUser: {user_content}\n\nAssistant:"


def prepare_hf_training_dataset(df: pd.DataFrame, tokenizer: Any) -> Dataset:
    """Converts paired DataFrame into Hugging Face Dataset formatted for SFTTrainer."""
    records = []
    for _, row in df.iterrows():
        name = str(row.get("name", "Unknown Game"))
        genres = str(row.get("genres", "Indie"))
        desc = str(row.get("clean_description", ""))
        scores = {
            "hype_score": float(row["hype_score"]),
            "story_score": float(row["story_score"]),
            "nostalgia_score": float(row["nostalgia_score"])
        }
        formatted_text = format_instruction_prompt(
            tokenizer=tokenizer,
            game_name=name,
            genres=genres,
            description=desc,
            target_scores=scores
        )
        records.append({"text": formatted_text})
    return Dataset.from_list(records)


def extract_json_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extracts and parses JSON object from model output."""
    def try_parse(candidate: str) -> Optional[Dict[str, Any]]:
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

    # Fallback key-value extraction
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
    """Validates that parsed JSON contains required keys within [0.0, 10.0]."""
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


def evaluate_model_on_dataset(
    model: Any,
    tokenizer: Any,
    df_eval: pd.DataFrame,
    device: str,
    top_p: float = 0.9,
    temperature: float = 0.7,
    max_new_tokens: int = 60,
    label: str = "Model"
) -> Dict[str, Any]:
    """Evaluates model prediction accuracy and validity against ground-truth vibe scores with top-p sampling."""
    print(f"\n--- Running Evaluation: {label} ({len(df_eval)} samples, top_p={top_p}, temp={temperature}) ---")
    valid_count = 0
    total_samples = len(df_eval)
    all_errors: List[float] = []
    within_tolerance_count = 0
    total_eval_points = 0
    latencies: List[float] = []

    model.eval()

    for idx, row in df_eval.iterrows():
        name = str(row.get("name", "Unknown Game"))
        genres = str(row.get("genres", "Indie"))
        desc = str(row.get("clean_description", ""))
        gt_scores = {
            "hype_score": float(row["hype_score"]),
            "story_score": float(row["story_score"]),
            "nostalgia_score": float(row["nostalgia_score"])
        }

        prompt = format_instruction_prompt(tokenizer, name, genres, desc, target_scores=None)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start_t = time.time()
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - start_t
        latencies.append(elapsed)

        new_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        parsed = extract_json_payload(raw_output)
        is_valid, validated_scores = validate_vibe_scores(parsed)

        if is_valid:
            valid_count += 1
            for k in ["hype_score", "story_score", "nostalgia_score"]:
                err = abs(validated_scores[k] - gt_scores[k])
                all_errors.append(err)
                if err <= 2.0:
                    within_tolerance_count += 1
                total_eval_points += 1
        else:
            # Penalize invalid formatting with max default error (5.0)
            for _ in range(3):
                all_errors.append(5.0)
                total_eval_points += 1

    validity_rate = (valid_count / max(total_samples, 1)) * 100.0
    mae = sum(all_errors) / max(len(all_errors), 1)
    acc_pct = (within_tolerance_count / max(total_eval_points, 1)) * 100.0
    avg_latency = sum(latencies) / max(len(latencies), 1)

    results = {
        "label": label,
        "total_samples": total_samples,
        "valid_json_count": valid_count,
        "validity_rate_pct": round(validity_rate, 2),
        "mae": round(mae, 2),
        "accuracy_within_tolerance_pct": round(acc_pct, 2),
        "avg_latency_sec": round(avg_latency, 3),
        "top_p": top_p,
        "temperature": temperature
    }

    print(f"Results for [{label}]:")
    print(f"  * JSON Validity Rate: {validity_rate:.1f}% ({valid_count}/{total_samples})")
    print(f"  * Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  * Accuracy (+/- 2.0 tolerance): {acc_pct:.1f}%")
    print(f"  * Avg Latency: {avg_latency:.3f}s / game")
    return results


def run_qlora_training(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    data_dir: str = "cleaned_data",
    output_dir: str = "saved_models/llm_adapters/qwen_steam_vibe_8bit",
    max_train_samples: int = 300,
    max_eval_samples: int = 15,
    epochs: int = 1,
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
    Executes the Phase 3 LLM Fine-Tuning pipeline:
    1. Loads cleaned training & holdout evaluation splits from cleaned_data/.
    2. Initializes Qwen3.5-0.8B with 8-bit BitsAndBytes quantization on CUDA.
    3. Modifies top-p parameter for training generation / sampling.
    4. Evaluates baseline model before fine-tuning.
    5. Configures LoRA adapter (r=16, alpha=32) and SFTTrainer on paired Steam descriptions and vibe scores.
    6. Executes fine-tuning loop on GPU.
    7. Saves trained LoRA adapter to output_dir.
    8. Evaluates fine-tuned model after training and records before/after improvement comparison.
    """
    hf_token = os.getenv("HF_TOKEN") or None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_output = resolve_output_dir(output_dir)
    os.makedirs(resolved_output, exist_ok=True)

    print("==================================================================")
    print("      STEAM HIT PREDICTOR - PHASE 3: 8-BIT QLORA FINE-TUNING      ")
    print("==================================================================")
    print(f"Base Model:             {model_id}")
    print(f"Quantization:           BitsAndBytes 8-bit (load_in_8bit=True)")
    print(f"Sampling Parameters:    top_p={top_p}, temperature={temperature}")
    print(f"LoRA Configuration:     rank={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Training Dataset:       {max_train_samples} samples from cleaned_data/train/")
    print(f"Evaluation Dataset:     {max_eval_samples} samples from cleaned_data/test/")
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

    print(f"Loading base model '{model_id}' in 8-bit quantization on {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        token=hf_token
    )

    # Configure top-p parameter in generation config
    if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
        base_model.generation_config.top_p = top_p
        base_model.generation_config.temperature = temperature
        base_model.generation_config.do_sample = True
        print(f"Configured generation config: top_p={top_p}, temperature={temperature}, do_sample=True")

    # 4. Before Evaluation (Base Model)
    before_metrics = evaluate_model_on_dataset(
        model=base_model,
        tokenizer=tokenizer,
        df_eval=df_eval,
        device=device,
        top_p=top_p,
        temperature=temperature,
        label="Before Fine-Tuning (Base Model)"
    )

    # 5. Prepare Model for k-bit LoRA Training
    print("\nPreparing base model for 8-bit LoRA training...")
    base_model = prepare_model_for_kbit_training(base_model)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(base_model, peft_config)
    print("Trainable LoRA Parameter Summary:")
    peft_model.print_trainable_parameters()

    # 6. Build Hugging Face Training Dataset
    print(f"Formatting {len(df_train)} training instruction pairs...")
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
        logging_steps=10,
        save_strategy="no",
        fp16=False,
        bf16=False,
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

    print("\nStarting QLoRA 8-bit fine-tuning loop on GPU...")
    train_start_t = time.time()
    train_result = trainer.train()
    training_duration = time.time() - train_start_t
    train_loss = train_result.training_loss
    print(f"Fine-tuning complete in {training_duration:.2f}s! Final Training Loss: {train_loss:.4f}")

    # 8. Save Fine-Tuned LoRA Adapter & Tokenizer
    print(f"\nSaving fine-tuned LoRA adapter to: '{resolved_output}'...")
    trainer.model.save_pretrained(resolved_output)
    tokenizer.save_pretrained(resolved_output)

    # Also save adapter config copy at root of llm_adapters for convenience
    root_adapter_dir = os.path.dirname(resolved_output)
    trainer.model.save_pretrained(root_adapter_dir)
    print(f"Saved primary adapter artifacts to: '{root_adapter_dir}' and '{resolved_output}'")

    # 9. After Evaluation (Fine-Tuned Model)
    after_metrics = evaluate_model_on_dataset(
        model=trainer.model,
        tokenizer=tokenizer,
        df_eval=df_eval,
        device=device,
        top_p=top_p,
        temperature=temperature,
        label="After Fine-Tuning (QLoRA 8-Bit Adapter)"
    )

    # 10. Summary Comparison & Report
    mae_diff = before_metrics["mae"] - after_metrics["mae"]
    acc_diff = after_metrics["accuracy_within_tolerance_pct"] - before_metrics["accuracy_within_tolerance_pct"]
    valid_diff = after_metrics["validity_rate_pct"] - before_metrics["validity_rate_pct"]

    print("\n==================================================================")
    print("           QLORA 8-BIT BEFORE VS AFTER IMPROVEMENT REPORT         ")
    print("==================================================================")
    print(f"{'Metric':<32} | {'Before (Base)':<15} | {'After (QLoRA)':<15} | {'Improvement'}")
    print("-" * 75)
    print(f"{'JSON Schema Validity':<32} | {before_metrics['validity_rate_pct']:>13.1f}% | {after_metrics['validity_rate_pct']:>13.1f}% | {valid_diff:>+10.1f}%")
    print(f"{'Mean Absolute Error (MAE)':<32} | {before_metrics['mae']:>15.2f} | {after_metrics['mae']:>15.2f} | {-mae_diff:>+11.2f} (lower is better)")
    print(f"{'Accuracy (+/- 2.0 tolerance)':<32} | {before_metrics['accuracy_within_tolerance_pct']:>13.1f}% | {after_metrics['accuracy_within_tolerance_pct']:>13.1f}% | {acc_diff:>+10.1f}%")
    print(f"{'Avg Latency':<32} | {before_metrics['avg_latency_sec']:>14.3f}s | {after_metrics['avg_latency_sec']:>14.3f}s | {after_metrics['avg_latency_sec'] - before_metrics['avg_latency_sec']:>+10.3f}s")
    print("==================================================================\n")

    summary = {
        "model_id": model_id,
        "quantization": "8-bit (BitsAndBytesConfig load_in_8bit=True)",
        "top_p": top_p,
        "temperature": temperature,
        "lora_rank": lora_r,
        "lora_alpha": lora_alpha,
        "training_samples": len(df_train),
        "evaluation_samples": len(df_eval),
        "training_epochs": epochs,
        "final_train_loss": round(float(train_loss), 4),
        "training_duration_sec": round(training_duration, 2),
        "adapter_save_path": resolved_output,
        "before_finetuning": before_metrics,
        "after_finetuning": after_metrics,
        "improvement": {
            "mae_reduction": round(float(mae_diff), 2),
            "accuracy_gain_pct": round(float(acc_diff), 2),
            "validity_gain_pct": round(float(valid_diff), 2)
        }
    }

    results_dir = os.path.join(CURRENT_DIR, "baseline_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "qlora_training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved complete evaluation report to: '{results_path}'")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Steam Hit Predictor - Phase 3: 8-bit QLoRA Fine-Tuning")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-0.8B", help="Base model identifier")
    parser.add_argument("--data-dir", type=str, default="cleaned_data", help="Cleaned dataset directory")
    parser.add_argument("--output-dir", type=str, default="saved_models/llm_adapters/qwen_steam_vibe_8bit", help="Adapter output directory")
    parser.add_argument("--max-train-samples", type=int, default=300, help="Number of training samples to use")
    parser.add_argument("--max-eval-samples", type=int, default=15, help="Number of holdout test samples for evaluation")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter for generation/evaluation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_qlora_training(
        model_id=args.model_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        top_p=args.top_p,
        temperature=args.temperature,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

