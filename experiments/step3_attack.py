"""Step 3 — the attack. Vulnerable Fed-GRPO (sigma-weighted aggregation) + a
lying client, compared against the honest baseline under identical settings.

The server now weights each client's update by its reported sigma (w = sig/Sum),
exactly as Fed-GRPO does. The liar (client LIAR) does two things:
  1) trains on a FLIPPED reward (rewards WRONG answers) -> a poisoning update;
  2) reports a hugely inflated sigma  -> grabs almost all the aggregation weight.
So its poison dominates the global model. We expect: attack accuracy < honest.

Runs honest then attack back to back, writes runs/step3_results.csv.
Run:  python experiments/step3_attack.py   then:  cat runs/step3_results.csv
"""
import re, gc, os, csv
import numpy as np, torch
import transformers; transformers.logging.set_verbosity_error()
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (LoraConfig, get_peft_model,
                  get_peft_model_state_dict, set_peft_model_state_dict)
from trl import GRPOConfig, GRPOTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_CLIENTS, ROUNDS, LOCAL_STEPS, EVAL_N, SEED = 4, 3, 6, 50, 0
LIAR, SIGMA_FACTOR, BOOST = 3, 30.0, 8.0   # client 3 lies: sign-flip x8 + sigma x30
os.makedirs("runs", exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)

SYS = "Solve step by step. Put the final numeric answer after '####'."
def gold(a): return a.split("####")[-1].strip().replace(",", "")
def fmt(ex):
    return {"prompt": [{"role": "system", "content": SYS},
                       {"role": "user", "content": ex["question"]}], "gold": gold(ex["answer"])}
raw = load_dataset("openai/gsm8k", "main")
train = raw["train"].select(range(2000)).map(fmt, remove_columns=raw["train"].column_names)
test  = raw["test"].select(range(EVAL_N)).map(fmt, remove_columns=raw["test"].column_names)
idx = np.arange(len(train)); np.random.shuffle(idx)
cuts = (np.cumsum(np.random.dirichlet([2.0]*N_CLIENTS))*len(idx)).astype(int)[:-1]
client_ds = [train.select(s.tolist()) for s in np.split(idx, cuts)]
print("shard sizes:", [len(c) for c in client_ds], flush=True)

def last_num(t):
    m = re.findall(r"-?\d[\d,]*", t.replace(",", "")); return m[-1] if m else ""
def reward_correct(completions, gold, **kw):            # reward RIGHT
    return [1.0 if last_num(c[0]["content"]) == g else 0.0 for c, g in zip(completions, gold)]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = get_peft_model(base, lora_cfg); model.cuda()

def clean_ref():
    if hasattr(model, "peft_config") and "ref" in model.peft_config: model.delete_adapter("ref")
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
def evaluate(bs=20):
    clean_ref(); model.eval(); c = 0
    for i in range(0, EVAL_N, bs):
        b = test.select(range(i, min(i+bs, EVAL_N)))
        texts = [tok.apply_chat_template(ex["prompt"], add_generation_prompt=True, tokenize=False) for ex in b]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        for j, ex in enumerate(b):
            c += (last_num(tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True)) == ex["gold"])
    model.train(); return c / EVAL_N

def sigma_weighted(deltas, sigmas):
    s = np.clip(np.asarray(sigmas, float), 1e-6, None); w = s / s.sum()
    return {kk: sum(float(w[i]) * deltas[i][kk] for i in range(len(deltas))) for kk in deltas[0]}

def run(attack: bool):
    g = {k: v.clone() for k, v in init_lora.items()}      # fresh global model
    accs = []
    for r in range(ROUNDS):
        deltas, sigmas = [], []
        for k in range(N_CLIENTS):
            liar = attack and k == LIAR
            clean_ref(); set_lora(g)
            trainer = GRPOTrainer(model=model, reward_funcs=reward_correct,
                                  args=grpo_args(f"runs/step3/{int(attack)}r{r}c{k}"),
                                  train_dataset=client_ds[k])
            trainer.train()
            logs = [l for l in trainer.state.log_history if "reward" in l]
            sig = float(logs[-1].get("reward_std", 0.1))
            new = get_lora()
            d = {kk: new[kk] - g[kk] for kk in g}
            if liar:                                       # Byzantine: sign-flip + amplify, then lie about sigma
                d = {kk: -BOOST * d[kk] for kk in d}
                sig *= SIGMA_FACTOR
            deltas.append(d); sigmas.append(sig)
            del trainer; gc.collect(); torch.cuda.empty_cache()
        agg = sigma_weighted(deltas, sigmas)
        g = {kk: g[kk] + agg[kk] for kk in g}
        set_lora(g); accs.append(evaluate())
    return accs

init_lora = get_lora()
print("\n== HONEST baseline (sigma-weighted) ==", flush=True)
honest = run(attack=False)
print("honest acc per round:", [round(a, 3) for a in honest], flush=True)
print("\n== ATTACK (lying client: flipped reward + fake sigma x30) ==", flush=True)
atk = run(attack=True)
print("attack acc per round:", [round(a, 3) for a in atk], flush=True)

with open("runs/step3_results.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["round", "honest_acc", "attack_acc", "drop"])
    for r in range(ROUNDS):
        wr.writerow([r, round(honest[r], 4), round(atk[r], 4), round(honest[r]-atk[r], 4)])

print("\n== results (runs/step3_results.csv) ==")
print(f"{'round':>5} {'honest':>8} {'attack':>8} {'drop':>8}")
for r in range(ROUNDS):
    print(f"{r:>5} {honest[r]:>8.3f} {atk[r]:>8.3f} {honest[r]-atk[r]:>8.3f}")
print("\nattack_acc < honest_acc (positive drop) => the lying client works.")
