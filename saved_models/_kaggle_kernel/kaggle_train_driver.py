import os
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

print("\nStarting 8-bit QLoRA training loop on Kaggle GPU...")
qlora_training.run_qlora_training(
    model_id="Qwen/Qwen3.5-0.8B",
    data_dir="/kaggle/input",
    output_dir="/kaggle/working/qwen_steam_vibe_8bit",
    max_train_samples=300,
    max_eval_samples=20,
    epochs=1,
    batch_size=2,
    grad_accum=4,
    learning_rate=0.0002,
    top_p=0.9,
    temperature=0.7,
    fp16=True,
    use_compile=False
)
print("\nKaggle GPU fine-tuning completed successfully! Outputs written to /kaggle/working/.")
