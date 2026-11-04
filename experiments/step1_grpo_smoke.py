"""Step 1 — single-client GRPO smoke test.

Goal: confirm the GRPO engine actually trains and the mean reward goes UP.
Uses GSM8K (small, fast, clean numeric reward) to validate the pipeline cheaply;
switch `DATASET` to AI-MO/NuminaMath-CoT for the real federated experiments.

Tuned for a 12GB GPU (RTX 3060): Qwen2.5-0.5B + LoRA + bf16 + checkpointing.

Run:  python experiments/step1_grpo_smoke.py
Watch the `reward` column in the logs climb from ~0.1 upward = engine works.
"""
import re
import transformers, trl
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

print(f"transformers {transformers.__version__} | trl {trl.__version__}")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "openai/gsm8k"          # smoke test; later: "AI-MO/NuminaMath-CoT"
N_EXAMPLES = 1000

# ---------------------------------------------------------------- data
ds = load_dataset(DATASET, "main", split=f"train[:{N_EXAMPLES}]")
SYS = "Solve the problem step by step. Put the final numeric answer after '####'."

def gold_answer(a):
    return a.split("####")[-1].strip().replace(",", "")

def to_prompt(ex):
    return {
        "prompt": [{"role": "system", "content": SYS},
                   {"role": "user", "content": ex["question"]}],
        "gold": gold_answer(ex["answer"]),
    }

ds = ds.map(to_prompt, remove_columns=ds.column_names)

# ---------------------------------------------------------------- reward
def last_number(txt):
    nums = re.findall(r"-?\d[\d,]*", txt.replace(",", ""))
    return nums[-1] if nums else ""

def reward_correct(completions, gold, **kwargs):
    """+1 if the model's final number matches the gold answer, else 0."""
    scores = []
    for comp, g in zip(completions, gold):
        ans = last_number(comp[0]["content"])
        scores.append(1.0 if ans == g else 0.0)
    return scores

# ---------------------------------------------------------------- train
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

cfg = GRPOConfig(
    output_dir="runs/step1",
    per_device_train_batch_size=6,      # must be a multiple of num_generations
    gradient_accumulation_steps=4,
    num_generations=6,                  # GRPO group size (answers per prompt)
    max_completion_length=256,          # (newer TRL dropped max_prompt_length)
    learning_rate=1e-5,
    beta=0.04,                          # KL penalty to the reference model
    logging_steps=5,
    max_steps=150,
    save_steps=150,
    bf16=True,                          # Ampere (3060) supports bf16
    gradient_checkpointing=True,
    use_vllm=False,                     # keep generation in-process on 12GB
    report_to="none",
)

trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=reward_correct,
    args=cfg,
    train_dataset=ds,
    peft_config=lora,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("runs/step1/final")
    print("\nDone. If the `reward` in the logs rose over the run, the GRPO engine works.")
