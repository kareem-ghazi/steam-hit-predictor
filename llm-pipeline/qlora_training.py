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
    Sets up Kaggle authentication using credentials from environment (.env).
    Supports:
    1. Modern API Token (KAGGLE_API_TOKEN) for kagglehub (>= 0.4.1) and kaggle CLI (>= 1.8.0).
    2. Legacy credentials (KAGGLE_USERNAME and KAGGLE_KEY) for backwards compatibility.
    
    Returns (is_authenticated, username).
    """
    # 1. Check newer recommended API token
    api_token = os.getenv("KAGGLE_API_TOKEN", "").strip()
    is_token_valid = bool(api_token and api_token != "your_kaggle_api_token")

    # 2. Check legacy username + key credentials
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    key = os.getenv("KAGGLE_KEY", "").strip()
    is_legacy_valid = bool(
        username
        and key
        and username != "your_kaggle_username"
        and key != "your_kaggle_api_key"
    )

    if not is_token_valid and not is_legacy_valid:
        return False, None

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    # Preferred: Modern API Token workflow
    if is_token_valid:
        os.environ["KAGGLE_API_TOKEN"] = api_token
        # Save token to ~/.kaggle/access_token so both kagglehub and kaggle CLI recognize it
        try:
            token_file = os.path.join(kaggle_dir, "access_token")
            with open(token_file, "w", encoding="utf-8") as f:
                f.write(api_token)
        except Exception as e:
            print(f"Note: Could not write ~/.kaggle/access_token: {e}")

        try:
            import kagglehub
            from kagglehub.config import set_kaggle_api_token
            set_kaggle_api_token(api_token)

            # Query authenticated username via whoami()
            auth_username = None
            try:
                user_info = kagglehub.whoami(verbose=False)
                auth_username = user_info.get("username")
            except Exception:
                auth_username = username if is_legacy_valid else None

            user_str = f" for user: '{auth_username}'" if auth_username else ""
            print(f"Kaggle API successfully authenticated via API Token{user_str} (modern token auth).")
            return True, auth_username
        except Exception as e:
            print(f"Kaggle API token authentication notice: {e}")
            return True, username if is_legacy_valid else None

    # Fallback: Legacy credentials workflow
    if is_legacy_valid:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        try:
            kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
            with open(kaggle_json_path, "w", encoding="utf-8") as f:
                json.dump({"username": username, "key": key}, f, indent=2)
        except Exception as e:
            print(f"Note: Could not write ~/.kaggle/kaggle.json: {e}")

        try:
            import kagglehub
            kagglehub.login()
            print(f"Kaggle API successfully authenticated for user: '{username}' (legacy credentials).")
            return True, username
        except Exception as e:
            print(f"Kaggle legacy authentication notice: {e}")
            return False, username

    return False, None


def upload_adapter_to_kaggle(
    local_model_dir: str,
    handle: str,
    version_notes: str = "Trained 8-bit QLoRA Steam Vibe Predictor adapter"
) -> bool:
    """Uploads the trained LoRA adapter to Kaggle Models / Kaggle Hub."""
    try:
        import kagglehub
        print(f"\nUploading fine-tuned adapter to Kaggle Hub model: '{handle}'...")
        kagglehub.model_upload(
            handle=handle,
            local_model_dir=local_model_dir,
            version_notes=version_notes
        )
        print(f"Successfully uploaded adapter to Kaggle: https://www.kaggle.com/models/{handle}")
        return True
    except Exception as e:
        print(f"Kaggle adapter upload failed: {e}")
        return False


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
        # Search Kaggle input mounts: /kaggle/input
        kaggle_input = "/kaggle/input"
        if os.path.exists(kaggle_input):
            for root, dirs, files in os.walk(kaggle_input):
                if "games.csv" in files and os.path.basename(root) in ["train", "valid", "test"]:
                    parent = os.path.dirname(root)
                    print(f"Detected Kaggle input dataset root at: '{parent}'")
                    return parent
                if os.path.basename(root) == "cleaned_data" and any(d in dirs for d in ["train", "test"]):
                    return root

    candidates = [
        data_dir,
        os.path.join(PARENT_DIR, data_dir),
        os.path.join(CURRENT_DIR, "..", data_dir),
    ]
    return next((os.path.abspath(c) for c in candidates if os.path.exists(c)), os.path.abspath(data_dir))


def resolve_output_dir(output_dir: str) -> str:
    """Resolves output directory path relative to project structure or Kaggle working directory."""
    if is_running_on_kaggle():
        if not output_dir.startswith("/kaggle/working"):
            basename = os.path.basename(output_dir.rstrip("/\\"))
            return os.path.join("/kaggle/working", basename)
        return output_dir
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.abspath(os.path.join(PARENT_DIR, output_dir))


def launch_training_on_kaggle(
    data_dir: str = "cleaned_data",
    output_dir: str = "saved_models/llm_adapters/qwen_steam_vibe_8bit",
    dataset_slug: str = "steam-hit-predictor-cleaned-data",
    kernel_slug: str = "steam-vibe-qlora-training",
    model_id: str = "Qwen/Qwen3.5-0.8B",
    max_train_samples: int = 300,
    max_eval_samples: int = 20,
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    top_p: float = 0.9,
    temperature: float = 0.7,
    detach: bool = False,
    skip_dataset_upload: bool = False,
    poll_interval: int = 20
) -> bool:
    """
    Submits and manages a full GPU-accelerated QLoRA training job remotely on Kaggle:
    1. Authenticates using KAGGLE_API_TOKEN from .env.
    2. Uploads or checks the cleaned dataset on Kaggle Datasets (<username>/<dataset_slug>).
    3. Bundles code and prepares kernel metadata enabling GPU and internet.
    4. Pushes the training code to Kaggle Kernels (<username>/<kernel_slug>).
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
    dataset_ref = f"{username}/{dataset_slug}"
    kernel_ref = f"{username}/{kernel_slug}"

    print("==================================================================")
    print("      LAUNCHING STEAM VIBE QLORA TRAINING ON KAGGLE GPU          ")
    print("==================================================================")
    print(f"Kaggle User:           {username}")
    print(f"Kaggle Dataset:        {dataset_ref}")
    print(f"Kaggle Kernel:         {kernel_ref}")
    print(f"Local Data Directory:  {resolved_data}")
    print(f"Target Adapter Output: {resolved_output}")
    print(f"Training Config:       {max_train_samples} samples, {epochs} epochs, top_p={top_p}")
    print("==================================================================\n")

    # Step 1: Ensure Dataset exists on Kaggle
    if not skip_dataset_upload:
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
                    version_notes="Cleaned Steam dataset splits (train, valid, test) for QLoRA training"
                )
                print(f"Dataset upload complete: https://www.kaggle.com/datasets/{dataset_ref}")
            except Exception as e:
                print(f"Note on dataset upload via kagglehub: {e}")
                print("If the dataset already exists or was created manually, the kernel will attempt mounting.")

    # Step 2: Prepare Kaggle Kernel Staging Directory
    staging_dir = os.path.join(PARENT_DIR, "saved_models", "_kaggle_kernel")
    os.makedirs(staging_dir, exist_ok=True)

    # Copy current qlora_training.py into staging
    current_script_path = os.path.abspath(__file__)
    staged_script_path = os.path.join(staging_dir, "qlora_training.py")
    import shutil
    shutil.copy2(current_script_path, staged_script_path)

    # Create driver script that installs dependencies and runs training on Kaggle GPU
    driver_script_path = os.path.join(staging_dir, "kaggle_train_driver.py")
    driver_code = f'''import os
import sys
import subprocess

print("==================================================================")
print("     KAGGLE GPU DRIVER: STEAM VIBE 8-BIT QLORA FINE-TUNING       ")
print("==================================================================")

# Install required packages
print("Installing training dependencies in Kaggle environment...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.40.0", "peft>=0.10.0", "trl>=0.8.0",
    "bitsandbytes>=0.43.0", "accelerate>=0.28.0", "datasets>=2.18.0", "tqdm"
])

import qlora_training

print("\\nStarting 8-bit QLoRA training loop on Kaggle GPU...")
qlora_training.run_qlora_training(
    model_id="{model_id}",
    data_dir="/kaggle/input",
    output_dir="/kaggle/working/qwen_steam_vibe_8bit",
    max_train_samples={max_train_samples},
    max_eval_samples={max_eval_samples},
    epochs={epochs},
    batch_size={batch_size},
    grad_accum={grad_accum},
    learning_rate={learning_rate},
    top_p={top_p},
    temperature={temperature},
    fp16=True,
    use_compile=False
)
print("\\nKaggle GPU fine-tuning completed successfully! Outputs written to /kaggle/working/.")
'''
    with open(driver_script_path, "w", encoding="utf-8") as f:
        f.write(driver_code)

    # Create kernel-metadata.json
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
        print("Detach mode active: Kernel is running remotely on Kaggle in the background.")
        print(f"When finished, download the trained adapter by running:")
        print(f"  python llm-pipeline/qlora_training.py --pull-kaggle-output --kernel-slug {kernel_slug}")
        return True

    # Step 4: Monitor Kernel Execution Status
    print(f"Monitoring Kaggle GPU execution (polling every {poll_interval}s)...")
    last_status = None
    start_poll_t = time.time()
    while True:
        time.sleep(poll_interval)
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

    # Step 5: Pull Trained Adapter Artifacts from Kaggle Output
    return pull_kaggle_kernel_output(kernel_ref=kernel_ref, output_dir=resolved_output)


def pull_kaggle_kernel_output(
    kernel_ref: Optional[str] = None,
    kernel_slug: str = "steam-vibe-qlora-training",
    output_dir: str = "saved_models/llm_adapters/qwen_steam_vibe_8bit"
) -> bool:
    """Downloads the trained LoRA adapter outputs from a completed Kaggle kernel."""
    is_auth, username = setup_kaggle_environment()
    if not is_auth or not username:
        print("Kaggle authentication required. Please set KAGGLE_API_TOKEN in .env.")
        return False

    if not kernel_ref:
        kernel_ref = f"{username}/{kernel_slug}"

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

        # Ensure root adapter directory has copies of config and weights
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
                    if not os.path.exists(target) or os.path.getmtime(c) > os.path.getmtime(target):
                        shutil.copy2(c, target)
                        print(f"Copied primary adapter artifact: {fname} -> {target}")

        print(f"\nSuccessfully synchronized fine-tuned LoRA adapter from Kaggle!")
        print(f"Local adapter location: '{resolved_output}'")
        return True
    except Exception as e:
        print(f"Failed to download outputs from Kaggle kernel: {e}")
        return False


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


def prepare_hf_training_dataset(
    df: pd.DataFrame,
    tokenizer: Any,
    num_proc: Optional[int] = None
) -> Dataset:
    """
    Converts paired DataFrame into Hugging Face Dataset formatted for SFTTrainer
    using vectorized batched .map() across CPU cores instead of slow sequential iteration.
    """
    keep_cols = ["name", "genres", "clean_description", "hype_score", "story_score", "nostalgia_score"]
    filtered_df = df[[c for c in keep_cols if c in df.columns]].copy()
    dataset = Dataset.from_pandas(filtered_df, preserve_index=False)

    if num_proc is None:
        cpu_count = os.cpu_count() or 1
        num_proc = max(1, min(4, cpu_count))

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

    print(f"Vectorizing & pre-formatting instruction dataset using .map(batched=True, num_proc={num_proc})...")
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
        print(f"Multiprocessing map fallback to num_proc=1 ({e})")
        formatted_dataset = dataset.map(
            batch_format_prompts,
            batched=True,
            batch_size=1000,
            num_proc=1,
            remove_columns=dataset.column_names,
            desc="Vectorizing instruction dataset"
        )

    return formatted_dataset


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
    eval_batch_size: int = 4,
    use_compile: bool = True,
    label: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluates model prediction accuracy and validity against ground-truth vibe scores
    using batched evaluation, torch.compile(), and left-padding generation acceleration.
    """
    print(f"\n--- Running Batched Evaluation: {label} ({len(df_eval)} samples, batch_size={eval_batch_size}, top_p={top_p}, temp={temperature}) ---")
    
    # Enable torch.compile() if requested
    eval_model = model
    if use_compile and hasattr(torch, "compile"):
        try:
            print(f"Applying torch.compile() to {label} for kernel execution speedup...")
            eval_model = torch.compile(model)
        except Exception as e:
            print(f"Note: torch.compile() fallback to eager execution: {e}")
            eval_model = model

    eval_model.eval()

    # Left-padding is required for batched autoregressive generation
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = df_eval.to_dict(orient="records")
    total_samples = len(records)
    valid_count = 0
    all_errors: List[float] = []
    within_tolerance_count = 0
    total_eval_points = 0
    latencies: List[float] = []

    num_batches = (total_samples + eval_batch_size - 1) // eval_batch_size
    batch_indices = [
        range(i, min(i + eval_batch_size, total_samples))
        for i in range(0, total_samples, eval_batch_size)
    ]

    for b_idx in tqdm(batch_indices, total=num_batches, desc=f"Evaluating [{label}] (Batched)", unit="batch"):
        batch_records = [records[i] for i in b_idx]
        prompts = [
            format_instruction_prompt(
                tokenizer=tokenizer,
                game_name=str(r.get("name", "Unknown Game")),
                genres=str(r.get("genres", "Indie")),
                description=str(r.get("clean_description", "")),
                target_scores=None
            )
            for r in batch_records
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        start_t = time.time()
        with torch.no_grad():
            output_tokens = eval_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - start_t
        per_item_latency = elapsed / max(len(batch_records), 1)

        prompt_len = inputs["input_ids"].shape[1]
        for idx_in_batch, r in enumerate(batch_records):
            latencies.append(per_item_latency)
            gt_scores = {
                "hype_score": float(r["hype_score"]),
                "story_score": float(r["story_score"]),
                "nostalgia_score": float(r["nostalgia_score"])
            }

            new_tokens = output_tokens[idx_in_batch][prompt_len:]
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
                for _ in range(3):
                    all_errors.append(5.0)
                    total_eval_points += 1

    # Restore tokenizer padding side
    tokenizer.padding_side = orig_padding_side

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
        "eval_batch_size": eval_batch_size,
        "top_p": top_p,
        "temperature": temperature
    }

    print(f"Results for [{label}]:")
    print(f"  * JSON Validity Rate: {validity_rate:.1f}% ({valid_count}/{total_samples})")
    print(f"  * Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  * Accuracy (+/- 2.0 tolerance): {acc_pct:.1f}%")
    print(f"  * Avg Latency: {avg_latency:.3f}s / game (Batched x{eval_batch_size})")
    return results


def run_qlora_training(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    data_dir: str = "cleaned_data",
    output_dir: str = "saved_models/llm_adapters/qwen_steam_vibe_8bit",
    max_train_samples: int = 300,
    max_eval_samples: int = 20,
    eval_batch_size: int = 4,
    use_compile: bool = True,
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    top_p: float = 0.9,
    temperature: float = 0.7,
    bf16: Optional[bool] = None,
    fp16: Optional[bool] = None,
    num_proc: Optional[int] = None,
    push_to_kaggle: bool = False,
    kaggle_model_handle: Optional[str] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes the optimized Phase 3 LLM Fine-Tuning pipeline:
    1. Loads cleaned training & holdout evaluation splits from cleaned_data/.
    2. Initializes base LLM (from HuggingFace or Kaggle Hub) with 8-bit BitsAndBytes quantization on CUDA.
    3. Enables hardware accelerated mixed precision (bf16/fp16).
    4. Evaluates baseline model with batched generation and torch.compile().
    5. Formats training dataset using parallel vectorized .map(batched=True).
    6. Configures LoRA adapter (r=16, alpha=32) and SFTTrainer with bf16/fp16.
    7. Executes fine-tuning loop on GPU.
    8. Saves trained LoRA adapter to output_dir.
    9. Optionally uploads fine-tuned adapter to Kaggle Models via Kaggle Hub.
    10. Evaluates fine-tuned model after training and records before/after improvement comparison.
    """
    hf_token = os.getenv("HF_TOKEN") or None
    is_kaggle_auth, kaggle_user = setup_kaggle_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_output = resolve_output_dir(output_dir)
    os.makedirs(resolved_output, exist_ok=True)

    # Support downloading base model from Kaggle Hub via kaggle:// prefix
    if model_id.startswith("kaggle://"):
        kaggle_handle = model_id[len("kaggle://"):]
        try:
            import kagglehub
            print(f"Downloading base model from Kaggle Hub: '{kaggle_handle}'...")
            model_id = kagglehub.model_download(kaggle_handle)
            print(f"Downloaded model to local path: '{model_id}'")
        except Exception as e:
            print(f"Failed to download model from Kaggle Hub: {e}")
            raise

    # Mixed precision auto-detection & configuration
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        if bf16 is True or (bf16 is None and torch.cuda.is_bf16_supported()):
            use_bf16 = True
            use_fp16 = False
            precision_desc = "bf16=True (Hardware bfloat16 accelerated)"
        elif fp16 is True or (fp16 is None and not torch.cuda.is_bf16_supported()):
            use_fp16 = True
            use_bf16 = False
            precision_desc = "fp16=True (Hardware float16 accelerated)"
        else:
            precision_desc = "fp32 (Standard precision)"
    else:
        precision_desc = "CPU (fp32)"

    kaggle_status = f"Authenticated as '{kaggle_user}'" if is_kaggle_auth else "Not configured / Local mode"
    print("==================================================================")
    print("      STEAM HIT PREDICTOR - PHASE 3: 8-BIT QLORA FINE-TUNING      ")
    print("==================================================================")
    print(f"Base Model:             {model_id}")
    print(f"Quantization:           BitsAndBytes 8-bit (load_in_8bit=True)")
    print(f"Mixed Precision:        {precision_desc}")
    print(f"Kaggle Integration:     {kaggle_status}")
    print(f"Sampling Parameters:    top_p={top_p}, temperature={temperature}")
    print(f"LoRA Configuration:     rank={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Training Dataset:       {max_train_samples} samples from cleaned_data/train/")
    print(f"Evaluation Dataset:     {max_eval_samples} samples (Batched x{eval_batch_size}) from cleaned_data/test/")
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
    print(f"Loading base model '{model_id}' in 8-bit quantization with {model_dtype} on {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=model_dtype,
        device_map="auto" if device == "cuda" else None,
        token=hf_token
    )

    # Configure top-p parameter in generation config
    if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
        base_model.generation_config.top_p = top_p
        base_model.generation_config.temperature = temperature
        base_model.generation_config.do_sample = True
        print(f"Configured generation config: top_p={top_p}, temperature={temperature}, do_sample=True")

    # 4. Before Evaluation (Base Model with Batched Evaluation & torch.compile)
    before_metrics = evaluate_model_on_dataset(
        model=base_model,
        tokenizer=tokenizer,
        df_eval=df_eval,
        device=device,
        top_p=top_p,
        temperature=temperature,
        eval_batch_size=eval_batch_size,
        use_compile=use_compile,
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

    # 6. Build Hugging Face Training Dataset via Vectorized Batched .map()
    train_hf_ds = prepare_hf_training_dataset(df_train, tokenizer, num_proc=num_proc)

    # 7. SFTConfig & SFTTrainer with Hardware-Accelerated Mixed Precision
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

    print(f"\nStarting QLoRA 8-bit fine-tuning loop on GPU (bf16={use_bf16}, fp16={use_fp16})...")
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

    # 9. After Evaluation (Fine-Tuned Model with Batched Evaluation & torch.compile)
    after_metrics = evaluate_model_on_dataset(
        model=trainer.model,
        tokenizer=tokenizer,
        df_eval=df_eval,
        device=device,
        top_p=top_p,
        temperature=temperature,
        eval_batch_size=eval_batch_size,
        use_compile=use_compile,
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

    # 11. Optional Upload to Kaggle Hub Models
    kaggle_pushed = False
    if push_to_kaggle:
        target_handle = kaggle_model_handle
        if not target_handle and kaggle_user:
            target_handle = f"{kaggle_user}/steam-vibe-qlora-adapter/transformers/default"
        if target_handle:
            kaggle_pushed = upload_adapter_to_kaggle(
                local_model_dir=resolved_output,
                handle=target_handle,
                version_notes="Trained 8-bit QLoRA Steam Vibe Predictor adapter"
            )
        else:
            print("Notice: --push-to-kaggle was specified, but neither --kaggle-model-handle was provided nor could a Kaggle username be resolved. Provide --kaggle-model-handle or configure KAGGLE_API_TOKEN in .env.")

    summary = {
        "model_id": model_id,
        "quantization": "8-bit (BitsAndBytesConfig load_in_8bit=True)",
        "precision": precision_desc,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "kaggle": {
            "authenticated": is_kaggle_auth,
            "username": kaggle_user,
            "pushed": kaggle_pushed
        },
        "top_p": top_p,
        "temperature": temperature,
        "eval_batch_size": eval_batch_size,
        "use_compile": use_compile,
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
    parser = argparse.ArgumentParser(description="Steam Hit Predictor - Phase 3: Optimized 8-bit QLoRA Fine-Tuning")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-0.8B", help="Base model identifier")
    parser.add_argument("--data-dir", type=str, default="cleaned_data", help="Cleaned dataset directory")
    parser.add_argument("--output-dir", type=str, default="saved_models/llm_adapters/qwen_steam_vibe_8bit", help="Adapter output directory")
    parser.add_argument("--max-train-samples", type=int, default=10000, help="Number of training samples to use")
    parser.add_argument("--max-eval-samples", type=int, default=100, help="Number of holdout test samples for evaluation (default: 20)")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Batch size for model evaluation generation (default: 4)")
    parser.add_argument("--use-compile", action="store_true", default=True, help="Enable torch.compile() for evaluation acceleration")
    parser.add_argument("--no-compile", dest="use_compile", action="store_false", help="Disable torch.compile()")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter for generation/evaluation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--bf16", action="store_true", default=None, help="Force bfloat16 mixed precision")
    parser.add_argument("--fp16", action="store_true", default=None, help="Force float16 mixed precision")
    parser.add_argument("--num-proc", type=int, default=None, help="CPU workers for dataset .map()")
    parser.add_argument("--push-to-kaggle", action="store_true", default=False, help="Upload trained LoRA adapter to Kaggle Models")
    parser.add_argument("--kaggle-model-handle", type=str, default=None, help="Kaggle Model handle, e.g. <username>/<model-name>/transformers/<variation-slug>")
    parser.add_argument("--train-on-kaggle", "--run-on-kaggle", dest="train_on_kaggle", action="store_true", default=False, help="Submit and execute full GPU training job remotely on Kaggle")
    parser.add_argument("--detach", action="store_true", default=False, help="Submit Kaggle training job and exit immediately without polling")
    parser.add_argument("--pull-kaggle-output", action="store_true", default=False, help="Download output artifacts from a completed Kaggle kernel")
    parser.add_argument("--dataset-slug", type=str, default="steam-hit-predictor-cleaned-data", help="Kaggle Dataset slug (default: steam-hit-predictor-cleaned-data)")
    parser.add_argument("--kernel-slug", type=str, default="steam-vibe-qlora-training", help="Kaggle Kernel slug (default: steam-vibe-qlora-training)")
    parser.add_argument("--skip-dataset-upload", action="store_true", default=False, help="Skip checking/uploading dataset to Kaggle Datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Remote Kaggle execution entry points
    if args.train_on_kaggle:
        launch_training_on_kaggle(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            dataset_slug=args.dataset_slug,
            kernel_slug=args.kernel_slug,
            model_id=args.model_id,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            learning_rate=args.lr,
            top_p=args.top_p,
            temperature=args.temperature,
            detach=args.detach,
            skip_dataset_upload=args.skip_dataset_upload
        )
        return

    if args.pull_kaggle_output:
        pull_kaggle_kernel_output(
            kernel_slug=args.kernel_slug,
            output_dir=args.output_dir
        )
        return

    run_qlora_training(
        model_id=args.model_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        eval_batch_size=args.eval_batch_size,
        use_compile=args.use_compile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        top_p=args.top_p,
        temperature=args.temperature,
        bf16=args.bf16,
        fp16=args.fp16,
        num_proc=args.num_proc,
        push_to_kaggle=args.push_to_kaggle,
        kaggle_model_handle=args.kaggle_model_handle,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
