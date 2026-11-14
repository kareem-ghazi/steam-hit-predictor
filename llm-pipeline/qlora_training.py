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

# Load environment variables (.env) from root or current directory
for p in [os.path.join(PARENT_DIR, ".env"), os.path.join(CURRENT_DIR, ".env"), ".env"]:
    if os.path.exists(p):
        load_dotenv(p)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)


def setup_kaggle_environment() -> Tuple[bool, Optional[str]]:
    """
    Sets up Kaggle authentication using the modern API Token (KAGGLE_API_TOKEN) from .env.
    Returns (is_authenticated, username).
    """
    api_token = os.getenv("KAGGLE_API_TOKEN", "").strip()
    if not api_token or api_token == "your_kaggle_api_token":
        return False, None

    os.environ["KAGGLE_API_TOKEN"] = api_token

    # Write ~/.kaggle/access_token so both kagglehub and kaggle CLI recognize the token
    try:
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        token_file = os.path.join(kaggle_dir, "access_token")
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(api_token)
    except Exception as e:
        print(f"Note: Could not write ~/.kaggle/access_token: {e}")

    try:
        import kagglehub
        from kagglehub.config import set_kaggle_api_token
        set_kaggle_api_token(api_token)
        user_info = kagglehub.whoami(verbose=False)
        username = user_info.get("username")
        print(f"Kaggle API successfully authenticated for user: '{username}'")
        return True, username
    except Exception as e:
        print(f"Kaggle authentication note: {e}")
        return True, None


def is_running_on_kaggle() -> bool:
    """Returns True if the script is running inside a Kaggle notebook or batch kernel environment."""
    return (
        os.path.exists("/kaggle/working")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_URL_BASE" in os.environ
    )


def resolve_data_dir(data_dir: str = "cleaned_data") -> str:
    """Resolves data directory path whether executed locally or inside a Kaggle kernel."""
    if is_running_on_kaggle():
        kaggle_input = "/kaggle/input"
        if os.path.exists(kaggle_input):
            for root, dirs, files in os.walk(kaggle_input):
                if "games.csv" in files and os.path.basename(root) in ["train", "valid", "test"]:
                    return os.path.dirname(root)
                if os.path.basename(root) == "cleaned_data" and any(d in dirs for d in ["train", "test"]):
                    return root

    candidates = [
        data_dir,
        os.path.join(PARENT_DIR, data_dir),
        os.path.join(CURRENT_DIR, "..", data_dir),
    ]
    return next((os.path.abspath(c) for c in candidates if os.path.exists(c)), os.path.abspath(data_dir))


def resolve_output_dir(output_dir: str = "saved_models/llm_adapters/qwen_steam_vibe_8bit") -> str:
    """Resolves output directory path relative to project structure or Kaggle working directory."""
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
    """Vectorized preparation of the training dataset using .map(batched=True)."""
    keep_cols = ["name", "genres", "clean_description", "hype_score", "story_score", "nostalgia_score"]
    filtered_df = df[[c for c in keep_cols if c in df.columns]].copy()
    dataset = Dataset.from_pandas(filtered_df, preserve_index=False)
    num_proc = max(1, min(4, os.cpu_count() or 1))

    def batch_format_prompts(batch: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        names = batch.get("name", ["Unknown Game"] * len(batch["clean_description"]))
        genres_list = batch.get("genres", ["Indie"] * len(batch["clean_description"]))
        descriptions = batch.get("clean_description", [""] * len(batch["clean_description"]))
        hype_scores = batch.get("hype_score", [5.0] * len(batch["clean_description"]))
        story_scores = batch.get("story_score", [5.0] * len(batch["clean_description"]))
        nostalgia_scores = batch.get("nostalgia_score", [5.0] * len(batch["clean_description"]))

        formatted_texts = []
        for name, genres, desc, hype, story, nostalgia in zip(
            names, genres_list, descriptions, hype_scores, story_scores, nostalgia_scores
        ):
            scores = {
                "hype_score": float(hype),
                "story_score": float(story),
                "nostalgia_score": float(nostalgia)
            }
            text = format_instruction_prompt(
                tokenizer=tokenizer,
                game_name=str(name),
                genres=str(genres),
                description=str(desc),
                target_scores=scores
            )
            formatted_texts.append(text)
        return {"text": formatted_texts}

    try:
        formatted_dataset = dataset.map(
            batch_format_prompts,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="Vectorizing instruction dataset"
        )
    except Exception as e:
        print(f"Fallback to num_proc=1 ({e})")
        formatted_dataset = dataset.map(
            batch_format_prompts,
            batched=True,
            batch_size=1000,
            num_proc=1,
            remove_columns=dataset.column_names,
            desc="Vectorizing instruction dataset"
        )

    return formatted_dataset


def parse_vibe_scores_from_text(text: str) -> Optional[Dict[str, float]]:
    """Extracts and validates JSON vibe scores from model generation text."""
    try:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        required = ["hype_score", "story_score", "nostalgia_score"]
        if not all(k in data for k in required):
            return None
        scores = {}
        for k in required:
            val = float(data[k])
            scores[k] = max(1.0, min(10.0, round(val, 1)))
        return scores
    except Exception:
        return None


def evaluate_model_on_dataset(
    model: Any,
    tokenizer: Any,
    df_eval: pd.DataFrame,
    device: str,
    top_p: float = 0.9,
    temperature: float = 0.7,
    eval_batch_size: int = 4,
    use_compile: bool = True,
    label: str = "Evaluation Benchmark"
) -> Dict[str, Any]:
    """Evaluates the model on holdout test samples with batched generation."""
    print(f"\n--- Running {label} ({len(df_eval)} samples, batch_size={eval_batch_size}) ---")
    model.eval()

    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    eval_model = model
    if use_compile and hasattr(torch, "compile") and not is_running_on_kaggle():
        try:
            eval_model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile notice: {e}. Using uncompiled model.")
            eval_model = model

    total_samples = len(df_eval)
    valid_count = 0
    errors = []
    latencies = []
    evaluated_games = []

    batches = [df_eval.iloc[i : min(i + eval_batch_size, total_samples)] for i in range(0, total_samples, eval_batch_size)]

    for batch in tqdm(batches, desc=f"Evaluating ({label})", unit="batch"):
        prompts = [
            format_instruction_prompt(
                tokenizer=tokenizer,
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
    fp16: Optional[bool] = None,
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
    print(f"Training Dataset:       {sample_desc} from cleaned_data/train/")
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

    results_dir = os.path.join(CURRENT_DIR, "baseline_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "qlora_training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved complete evaluation report to: '{results_path}'")

    return summary


def launch_training_on_kaggle(
    epochs: int = 5,
    batch_size: int = 2,
    max_train_samples: Optional[int] = None,
    max_eval_samples: int = 20,
    detach: bool = False,
    data_dir: str = "cleaned_data",
    output_dir: str = "saved_models/llm_adapters/qwen_steam_vibe_8bit"
) -> bool:
    """
    Submits and manages a full GPU-accelerated QLoRA training job remotely on Kaggle:
    1. Authenticates using KAGGLE_API_TOKEN from .env.
    2. Uploads or checks the cleaned dataset on Kaggle Datasets (<username>/steam-hit-predictor-cleaned-data).
    3. Builds GPU kernel metadata (enable_gpu: true, enable_internet: true).
    4. Pushes training code to Kaggle Kernels (<username>/steam-vibe-qlora-training).
    5. Monitors execution status until completion.
    6. Automatically downloads trained LoRA adapter weights directly into local output_dir.
    """
    is_auth, username = setup_kaggle_environment()
    if not is_auth or not username:
        print("\n" + "=" * 70)
        print(" [!] Kaggle Authentication Required for Remote Training")
        print("=" * 70)
        print("To run training on Kaggle's free GPU cloud:")
        print("1. Go to https://www.kaggle.com/settings/api")
        print("2. Under 'API Tokens', click 'Generate New Token'")
        print("3. Copy the token and paste it into your .env file as:")
        print("   KAGGLE_API_TOKEN=<your_copied_token>")
        print("=" * 70 + "\n")
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Error authenticating with Kaggle API: {e}")
        return False

    resolved_data = resolve_data_dir(data_dir)
    resolved_output = resolve_output_dir(output_dir)
    dataset_ref = f"{username}/steam-hit-predictor-cleaned-data"
    kernel_ref = f"{username}/steam-vibe-qlora-training"

    sample_desc = f"{max_train_samples} samples" if max_train_samples else "Full Dataset (all 22,248 games)"

    print("==================================================================")
    print("      LAUNCHING STEAM VIBE QLORA TRAINING ON KAGGLE GPU          ")
    print("==================================================================")
    print(f"Kaggle User:           {username}")
    print(f"Kaggle Dataset:        {dataset_ref}")
    print(f"Kaggle Kernel:         {kernel_ref}")
    print(f"Training Epochs:       {epochs}")
    print(f"Training Samples:      {sample_desc}")
    print(f"Target Adapter Output: {resolved_output}")
    print("==================================================================\n")

    # Step 1: Ensure Dataset exists on Kaggle
    print(f"Checking if dataset '{dataset_ref}' exists on Kaggle...")
    dataset_exists = False
    try:
        status = str(api.dataset_status(dataset_ref))
        print(f"Dataset '{dataset_ref}' is present on Kaggle (status: {status}).")
        dataset_exists = True
    except Exception:
        dataset_exists = False

    if not dataset_exists:
        print(f"Uploading '{resolved_data}' to Kaggle Datasets as '{dataset_ref}'...")
        try:
            import kagglehub
            kagglehub.dataset_upload(
                handle=dataset_ref,
                local_dataset_dir=resolved_data,
                version_notes="Cleaned Steam dataset splits for QLoRA training"
            )
            print(f"Dataset upload complete: https://www.kaggle.com/datasets/{dataset_ref}")
        except Exception as e:
            print(f"Dataset upload note: {e}")

    # Step 2: Prepare Kaggle Kernel Staging Directory
    staging_dir = os.path.join(PARENT_DIR, "saved_models", "_kaggle_kernel")
    os.makedirs(staging_dir, exist_ok=True)

    current_script_path = os.path.abspath(__file__)
    staged_script_path = os.path.join(staging_dir, "qlora_training.py")
    import shutil
    shutil.copy2(current_script_path, staged_script_path)

    driver_script_path = os.path.join(staging_dir, "kaggle_train_driver.py")
    sample_code = "None" if max_train_samples is None else str(max_train_samples)
    driver_code = f'''import os
import sys
import subprocess

print("==================================================================")
print("     KAGGLE GPU: STEAM VIBE 8-BIT QLORA TRAINING (5 EPOCHS)       ")
print("==================================================================")

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.40.0", "peft>=0.10.0", "trl>=0.8.0",
    "bitsandbytes>=0.43.0", "accelerate>=0.28.0", "datasets>=2.18.0", "tqdm"
])

import qlora_training

qlora_training.run_qlora_training(
    data_dir="/kaggle/input",
    output_dir="/kaggle/working/qwen_steam_vibe_8bit",
    max_train_samples={sample_code},
    max_eval_samples={max_eval_samples},
    epochs={epochs},
    batch_size={batch_size},
    fp16=True,
    use_compile=False
)
print("Kaggle GPU training finished! Output saved to /kaggle/working/.")
'''
    with open(driver_script_path, "w", encoding="utf-8") as f:
        f.write(driver_code)

    metadata = {
        "id": kernel_ref,
        "title": "Steam Vibe QLoRA Training",
        "code_file": "kaggle_train_driver.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    metadata_path = os.path.join(staging_dir, "kernel-metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Step 3: Push Kernel to Kaggle
    print(f"Pushing GPU kernel package to Kaggle: '{kernel_ref}'...")
    try:
        api.kernels_push(staging_dir)
        print(f"Kernel successfully submitted to Kaggle!")
        print(f"Track execution online at: https://www.kaggle.com/code/{kernel_ref}\n")
    except Exception as e:
        print(f"Error pushing kernel to Kaggle: {e}")
        return False

    if detach:
        print("Detach mode: Kernel running in background on Kaggle.")
        print("Download adapter weights anytime when done by running:")
        print("  python llm-pipeline/qlora_training.py --pull")
        return True

    # Step 4: Monitor Execution
    print("Monitoring Kaggle GPU execution (polling status)...")
    last_status = None
    start_poll_t = time.time()
    while True:
        time.sleep(20)
        try:
            status_obj = api.kernels_status(kernel_ref)
            status_str = str(getattr(status_obj, "status", status_obj)).lower()
            msg = getattr(status_obj, "failure_message", None)
            elapsed = time.time() - start_poll_t

            if status_str != last_status:
                print(f"  [{elapsed:6.1f}s] Kernel status: '{status_str}'")
                last_status = status_str

            if status_str == "complete":
                print(f"\nKaggle GPU training completed successfully in {elapsed:.1f}s!")
                break
            elif status_str in ["error", "failed", "cancelacknowledged"]:
                print(f"\n[!] Kaggle kernel failed with status: '{status_str}'.")
                if msg:
                    print(f"Failure message: {msg}")
                return False
        except Exception as e:
            print(f"Notice during status check: {e}")

    # Step 5: Pull Output Artifacts
    return pull_kaggle_kernel_output(output_dir=resolved_output)


def pull_kaggle_kernel_output(
    output_dir: str = "saved_models/llm_adapters/qwen_steam_vibe_8bit"
) -> bool:
    """Downloads trained adapter outputs from the completed Kaggle kernel."""
    is_auth, username = setup_kaggle_environment()
    if not is_auth or not username:
        print("Kaggle authentication required. Please set KAGGLE_API_TOKEN in .env.")
        return False

    kernel_ref = f"{username}/steam-vibe-qlora-training"
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Error authenticating with Kaggle API: {e}")
        return False

    resolved_output = resolve_output_dir(output_dir)
    os.makedirs(resolved_output, exist_ok=True)
    print(f"\nDownloading outputs from Kaggle kernel '{kernel_ref}' into '{resolved_output}'...")

    try:
        outfiles, _ = api.kernels_output(kernel_ref, path=resolved_output, force=True, quiet=False)
        print(f"Downloaded {len(outfiles)} file(s) from Kaggle.")

        root_adapter_dir = os.path.dirname(resolved_output)
        import shutil
        for fname in ["adapter_config.json", "adapter_model.safetensors"]:
            candidates = [
                os.path.join(resolved_output, fname),
                os.path.join(resolved_output, "qwen_steam_vibe_8bit", fname),
            ]
            for c in candidates:
                if os.path.exists(c):
                    target = os.path.join(root_adapter_dir, fname)
                    shutil.copy2(c, target)
                    print(f"Copied primary adapter artifact: {fname} -> {target}")

        print(f"\nSuccessfully synchronized fine-tuned LoRA adapter from Kaggle!")
        print(f"Local adapter location: '{resolved_output}'")
        return True
    except Exception as e:
        print(f"Failed to download outputs from Kaggle kernel: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Steam Hit Predictor - Phase 3: 8-bit QLoRA Fine-Tuning (Default: Kaggle GPU)"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size (default: 2)")
    parser.add_argument("--local", action="store_true", default=False, help="Force training to run locally on PC instead of Kaggle GPU")
    parser.add_argument("--detach", action="store_true", default=False, help="Submit Kaggle GPU job and exit immediately without polling")
    parser.add_argument("--pull", action="store_true", default=False, help="Download output artifacts from a completed Kaggle kernel")

    args = parser.parse_args()

    # 1. Pull output from completed Kaggle job
    if args.pull:
        pull_kaggle_kernel_output()
        return

    # 2. If running inside Kaggle environment OR user explicitly requests local PC training
    if is_running_on_kaggle() or args.local:
        run_qlora_training(
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_train_samples=None
        )
        return

    # 3. DEFAULT: Launch full GPU training job remotely on Kaggle
    launch_training_on_kaggle(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_samples=None,
        detach=args.detach
    )


if __name__ == "__main__":
    main()
