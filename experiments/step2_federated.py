"""Step 2 — federated GRPO baseline (clean + logged to CSV).

4 clients, one GPU, simulated federation. Each client does local GRPO on its own
non-IID shard, reports (mu, sigma); the server averages the LoRA deltas; the
global model is evaluated each round. Results are written to
runs/step2_results.csv and printed as a clean table at the end.

Run:  python experiments/step2_federated.py   then:  cat runs/step2_results.csv
"""
import re, gc, os, csv
import numpy as np, torch
import transformers
transformers.logging.set_verbosity_error()            # silence the per-step spam
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (LoraConfig, get_peft_model,
                  get_peft_model_state_dict, set_peft_model_state_dict)
from trl import GRPOConfig, GRPOTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_CLIENTS, ROUNDS, LOCAL_STEPS, EVAL_N, SEED = 4, 3, 6, 30, 0
os.makedirs("runs", exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)

SYS = "Solve step by step. Put the final numeric answer after '####'."
def gold(a): return a.split("####")[-1].strip().replace(",", "")
def fmt(ex):
    return {"prompt": [{"role": "system", "content": SYS},
                       {"role": "user", "content": ex["question"]}],
            "gold": gold(ex["answer"])}
raw = load_dataset("openai/gsm8k", "main")
train = raw["train"].select(range(2000)).map(fmt, remove_columns=raw["train"].column_names)
test  = raw["test"].select(range(EVAL_N)).map(fmt, remove_columns=raw["test"].column_names)

idx = np.arange(len(train)); np.random.shuffle(idx)
props = np.random.dirichlet([2.0] * N_CLIENTS)
cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
client_ds = [train.select(s.tolist()) for s in np.split(idx, cuts)]
print("shard sizes:", [len(c) for c in client_ds], flush=True)

def last_num(t):
    m = re.findall(r"-?\d[\d,]*", t.replace(",", "")); return m[-1] if m else ""
def reward_correct(completions, gold, **kw):
    return [1.0 if last_num(c[0]["content"]) == g else 0.0
            for c, g in zip(completions, gold)]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = get_peft_model(base, lora_cfg); model.cuda()

def clean_ref():
    if hasattr(model, "peft_config") and "ref" in model.peft_config:
        model.delete_adapter("ref")
    model.set_adapter("default")
def get_lora():
    clean_ref(); return {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
def set_lora(s):
    clean_ref(); set_peft_model_state_dict(model, s)
def grpo_args(o):
    return GRPOConfig(output_dir=o, per_device_train_batch_size=4, gradient_accumulation_steps=2,
        num_generations=4, max_completion_length=256, learning_rate=1e-5, beta=0.04,
        logging_steps=LOCAL_STEPS, max_steps=LOCAL_STEPS, save_strategy="no", bf16=True,
        gradient_checkpointing=True, use_vllm=False, report_to="none", disable_tqdm=True)

@torch.no_grad()
def evaluate(bs=15):
    clean_ref(); model.eval(); c = 0
    for i in range(0, EVAL_N, bs):
        batch = test.select(range(i, min(i + bs, EVAL_N)))
        texts = [tok.apply_chat_template(ex["prompt"], add_generation_prompt=True, tokenize=False)
                 for ex in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        for j, ex in enumerate(batch):
            txt = tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True)
            c += (last_num(txt) == ex["gold"])
    model.train(); return c / EVAL_N

# ---- csv + console ----
csv_path = "runs/step2_results.csv"
cols = ["round", "acc"] + [f"{s}{k}" for k in range(N_CLIENTS) for s in ("mu", "sig")]
f = open(csv_path, "w", newline=""); w = csv.writer(f); w.writerow(cols)

def record(rnd, acc, mus, sigmas):
    row = [rnd, round(acc, 4)] + [round(v, 4) for p in zip(mus, sigmas) for v in p]
    w.writerow(row); f.flush()
    pc = "  ".join(f"{m:.2f}/{s:.2f}" for m, s in zip(mus, sigmas))
    print(f"[round {rnd:>4}]  acc={acc:.3f}   {pc}", flush=True)

print("\n== federated baseline (honest) ==", flush=True)
global_lora = get_lora()
record("init", evaluate(), [0]*N_CLIENTS, [0]*N_CLIENTS)
for r in range(ROUNDS):
    deltas, mus, sigmas = [], [], []
    for k in range(N_CLIENTS):
        clean_ref(); set_lora(global_lora)
        trainer = GRPOTrainer(model=model, reward_funcs=reward_correct,
                              args=grpo_args(f"runs/step2/r{r}c{k}"), train_dataset=client_ds[k])
        trainer.train()
        logs = [l for l in trainer.state.log_history if "reward" in l]
        mus.append(float(logs[-1]["reward"])); sigmas.append(float(logs[-1].get("reward_std", 0.0)))
        new = get_lora()
        deltas.append({kk: (new[kk] - global_lora[kk]) for kk in global_lora})
        del trainer; gc.collect(); torch.cuda.empty_cache()
    agg = {kk: torch.stack([d[kk] for d in deltas]).mean(0) for kk in global_lora}
    global_lora = {kk: global_lora[kk] + agg[kk] for kk in global_lora}
    set_lora(global_lora)
    record(r, evaluate(), mus, sigmas)

f.close()
print("\n== results (runs/step2_results.csv) ==")
print(open(csv_path).read())
print("acc holds/rises across rounds => federated loop works.")
