import os
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
    max_train_samples=None,
    max_eval_samples=20,
    epochs=5,
    batch_size=2,
    fp16=True,
    use_compile=False
)
print("Kaggle GPU training finished! Output saved to /kaggle/working/.")
