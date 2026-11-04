"""Step 3+4 — controlled attack + defense (one training pass, cheap to re-tune).

Train 4 honest clients once (15 local GRPO steps each) and cache their LoRA
updates to disk. Then evaluate THREE aggregations of the SAME cached updates:

  honest  : sigma-weighted FedAvg (vulnerable Fed-GRPO, all clients honest)
  attack  : client LIAR sends a sign-flipped x BOOST poison update AND a fake
            sigma x SIGMA_FACTOR  -> it grabs the weight and poisons the model
  defense : VERITAS flags the inconsistent client (reported sigma vs real update
            size) and averages only the trusted updates -> recovers

Because only the aggregation changes, the comparison is clean, and tuning BOOST /
SIGMA_FACTOR needs no retraining (the cache is reused).

Run:  python experiments/step34_controlled.py
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
N_CLIENTS, LOCAL_STEPS, EVAL_N, SEED = 4, 15, 80, 0
LIAR, SIGMA_FACTOR, BOOST = 3, 30.0, 8.0
CACHE = "runs/deltas.pt"
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

def last_num(t):
    m = re.findall(r"-?\d[\d,]*", t.replace(",", "")); return m[-1] if m else ""
def reward_correct(completions, gold, **kw):
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
    clean_ref(); return {k: v.detach().float().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
def set_lora(state):
    clean_ref()
    dev = model.device
    set_peft_model_state_dict(model, {k: v.to(dev).to(torch.bfloat16) for k, v in state.items()})
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

# ---------- train once (or load cache) ----------
init_lora = get_lora()
if os.path.exists(CACHE):
    print("loading cached client updates", flush=True)
    blob = torch.load(CACHE); deltas, sigmas, init_lora = blob["deltas"], blob["sigmas"], blob["init"]
else:
    print(f"training {N_CLIENTS} honest clients x {LOCAL_STEPS} steps (one-time)...", flush=True)
    deltas, sigmas = [], []
    for k in range(N_CLIENTS):
        set_lora(init_lora)
        trainer = GRPOTrainer(model=model, reward_funcs=reward_correct,
                              args=grpo_args(f"runs/step34/c{k}"), train_dataset=client_ds[k])
        trainer.train()
        logs = [l for l in trainer.state.log_history if "reward" in l]
        sigmas.append(float(logs[-1].get("reward_std", 0.1)))
        new = get_lora()
        deltas.append({kk: new[kk] - init_lora[kk] for kk in init_lora})
        del trainer; gc.collect(); torch.cuda.empty_cache()
        print(f"  client {k}: sigma={sigmas[-1]:.3f}", flush=True)
    torch.save({"deltas": deltas, "sigmas": sigmas, "init": init_lora}, CACHE)

# ---------- aggregation helpers ----------
keys = list(init_lora.keys())
def dnorm(d): return float(torch.sqrt(sum((v.float()**2).sum() for v in d.values())))
def weighted(deltas, sigmas):
    s = np.clip(np.asarray(sigmas, float), 1e-6, None); w = s / s.sum()
    return {kk: sum(float(w[i]) * deltas[i][kk] for i in range(len(deltas))) for kk in keys}
def veritas(deltas, sigmas, tau=2.5, fold=2.5):
    g = np.array([dnorm(d) for d in deltas]); s = np.asarray(sigmas, float)
    ratio = s / (g + 1e-9)
    med = np.median(ratio); mad = 1.4826*np.median(np.abs(ratio-med)) + 1e-9
    z = (ratio-med)/mad; fr = ratio/(med+1e-9)
    flags = (np.abs(z) > tau) & ((fr > fold) | (fr < 1.0/fold))
    keep = [i for i in range(len(deltas)) if not flags[i]] or list(range(len(deltas)))
    agg = {kk: torch.stack([deltas[i][kk] for i in keep]).mean(0) for kk in keys}
    return agg, flags
def apply_eval(agg):
    set_lora({kk: init_lora[kk] + agg[kk] for kk in keys}); return evaluate()

# ---------- three scenarios from the SAME cached updates ----------
print("\nclient update norms:", [round(dnorm(d), 3) for d in deltas], "sigmas:", [round(s,3) for s in sigmas], flush=True)

# honest
honest_acc = apply_eval(weighted(deltas, sigmas))

# attack: liar sign-flips+amplifies its update and inflates its sigma
d_atk = [dict(d) for d in deltas]; s_atk = list(sigmas)
d_atk[LIAR] = {kk: -BOOST * deltas[LIAR][kk] for kk in keys}
s_atk[LIAR] = sigmas[LIAR] * SIGMA_FACTOR
attack_acc = apply_eval(weighted(d_atk, s_atk))

# defense: VERITAS on the attacked set
agg_def, flags = veritas(d_atk, s_atk)
defense_acc = apply_eval(agg_def)
detected = bool(flags[LIAR]); false_pos = int(sum(flags[k] for k in range(N_CLIENTS) if k != LIAR))

with open("runs/step34_results.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["scenario", "accuracy"])
    wr.writerow(["honest", round(honest_acc, 4)])
    wr.writerow(["attack", round(attack_acc, 4)])
    wr.writerow(["defense", round(defense_acc, 4)])
    wr.writerow(["liar_detected", int(detected)])
    wr.writerow(["false_positives", false_pos])

print("\n================ RESULTS ================")
print(f"  honest  accuracy : {honest_acc:.3f}")
print(f"  attack  accuracy : {attack_acc:.3f}   (drop {honest_acc-attack_acc:+.3f})")
print(f"  defense accuracy : {defense_acc:.3f}   (recovered {defense_acc-attack_acc:+.3f})")
print(f"  liar flagged     : {detected}   (client {LIAR});  false positives: {false_pos}")
print(f"  flags            : {flags.tolist()}")
print("=========================================")
print("want: attack << honest, defense ~ honest, liar flagged True, false_pos 0")
