"""Step 5 — ablation sweeps from the cached updates (no retraining).

Reuses runs/deltas.pt (from step34) to produce three paper figures cheaply:
  1. attack strength : accuracy vs BOOST   (attack collapses, defense holds)
  2. detection vs lie : as the liar inflates sigma it gains weight but becomes
     easier to flag                        (the attacker's dilemma)
  3. threshold ROC   : TPR / FPR vs tau    (detector operating point)

Saves CSVs and PNGs under runs/. Run after step34 has created runs/deltas.pt.
Run:  pip install matplotlib && python experiments/step5_sweeps.py
"""
import re, os, csv
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import transformers; transformers.logging.set_verbosity_error()
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_CLIENTS, EVAL_N, LIAR, SEED = 4, 60, 3, 0
CACHE = "runs/deltas.pt"
np.random.seed(SEED); torch.manual_seed(SEED)

SYS = "Solve step by step. Put the final numeric answer after '####'."
def gold(a): return a.split("####")[-1].strip().replace(",", "")
def fmt(ex):
    return {"prompt": [{"role": "system", "content": SYS},
                       {"role": "user", "content": ex["question"]}], "gold": gold(ex["answer"])}
test = load_dataset("openai/gsm8k", "main")["test"].select(range(EVAL_N))
test = test.map(fmt, remove_columns=test.column_names)

def last_num(t):
    m = re.findall(r"-?\d[\d,]*", t.replace(",", "")); return m[-1] if m else ""

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
def set_lora(state):
    clean_ref(); dev = model.device
    set_peft_model_state_dict(model, {k: v.to(dev).to(torch.bfloat16) for k, v in state.items()})

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

blob = torch.load(CACHE)
deltas, sigmas, init_lora = blob["deltas"], blob["sigmas"], blob["init"]
keys = list(init_lora.keys())
def dnorm(d): return float(torch.sqrt(sum((v.float()**2).sum() for v in d.values())))
def weighted(ds, ss):
    s = np.clip(np.asarray(ss, float), 1e-6, None); w = s / s.sum()
    return {kk: sum(float(w[i]) * ds[i][kk] for i in range(len(ds))) for kk in keys}
def flags_for(ss, ds, tau=2.5, fold=2.5):
    g = np.array([dnorm(d) for d in ds]); s = np.asarray(ss, float)
    ratio = s / (g + 1e-9); med = np.median(ratio); mad = 1.4826*np.median(np.abs(ratio-med)) + 1e-9
    z = (ratio-med)/mad; fr = ratio/(med+1e-9)
    return (np.abs(z) > tau) & ((fr > fold) | (fr < 1.0/fold))
def veritas_agg(ds, ss):
    fl = flags_for(ss, ds)
    keep = [i for i in range(len(ds)) if not fl[i]] or list(range(len(ds)))
    return {kk: torch.stack([ds[i][kk] for i in keep]).mean(0) for kk in keys}, fl
def apply_eval(agg):
    set_lora({kk: init_lora[kk] + agg[kk] for kk in keys}); return evaluate()

def poison(boost):
    d = [dict(x) for x in deltas]; d[LIAR] = {kk: -boost * deltas[LIAR][kk] for kk in keys}; return d

honest_acc = apply_eval(weighted(deltas, sigmas))
defense_acc = apply_eval(veritas_agg(poison(100.0), [s if i != LIAR else s*30 for i, s in enumerate(sigmas)])[0])
print(f"honest={honest_acc:.3f}  defense={defense_acc:.3f}", flush=True)

# ---- sweep 1: attack strength (needs eval) ----
boosts = [2, 5, 10, 20, 50, 100, 200]
atk_accs = []
s_atk = [s if i != LIAR else s*30 for i, s in enumerate(sigmas)]
for b in boosts:
    a = apply_eval(weighted(poison(b), s_atk)); atk_accs.append(a)
    print(f"  boost {b:>4}: attack_acc={a:.3f}", flush=True)
with open("runs/sweep_boost.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["boost", "attack_acc", "honest_acc", "defense_acc"])
    for b, a in zip(boosts, atk_accs): wr.writerow([b, round(a, 4), round(honest_acc, 4), round(defense_acc, 4)])
plt.figure(figsize=(6, 4))
plt.plot(boosts, atk_accs, "o-", color="#bf3b2a", label="attack (no defense)")
plt.axhline(honest_acc, ls="--", color="#5f7138", label="honest baseline")
plt.axhline(defense_acc, ls="--", color="#bb8622", label="defense (VERITAS)")
plt.xscale("log"); plt.xlabel("attack strength (BOOST)"); plt.ylabel("global accuracy")
plt.title("Attack strength vs accuracy"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("runs/fig1_attack_strength.png", dpi=130); plt.close()

# ---- sweep 2: detection vs sigma inflation (no eval) ----
sfac = [1, 2, 3, 5, 10, 20, 30, 50, 100]
weights, detected = [], []
for sf in sfac:
    ss = [s if i != LIAR else s*sf for i, s in enumerate(sigmas)]
    w = np.asarray(ss, float) / np.sum(ss); weights.append(float(w[LIAR]))
    detected.append(bool(flags_for(ss, poison(100.0))[LIAR]))
with open("runs/sweep_sigma.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["sigma_factor", "liar_weight", "detected"])
    for sf, w, d in zip(sfac, weights, detected): wr.writerow([sf, round(w, 4), int(d)])
plt.figure(figsize=(6, 4))
plt.plot(sfac, weights, "o-", color="#2f4f74", label="liar's aggregation weight")
for sf, w, d in zip(sfac, weights, detected):
    plt.scatter([sf], [w], color=("#bf3b2a" if d else "#5f7138"), zorder=5, s=60)
plt.xscale("log"); plt.xlabel("sigma inflation factor"); plt.ylabel("liar's weight share")
plt.title("Attacker's dilemma: more weight -> easier to flag\n(red = flagged, green = undetected)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("runs/fig2_detection_vs_sigma.png", dpi=130); plt.close()

# ---- sweep 3: threshold ROC (no eval) ----
taus = np.linspace(0.5, 6.0, 23)
s_fixed = [s if i != LIAR else s*30 for i, s in enumerate(sigmas)]; d_fixed = poison(100.0)
tpr, fpr = [], []
for t in taus:
    fl = flags_for(s_fixed, d_fixed, tau=float(t))
    tpr.append(float(fl[LIAR]))
    fpr.append(float(sum(fl[k] for k in range(N_CLIENTS) if k != LIAR)) / (N_CLIENTS - 1))
with open("runs/sweep_tau.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["tau", "tpr", "fpr"])
    for t, a, b in zip(taus, tpr, fpr): wr.writerow([round(float(t), 3), a, round(b, 3)])
plt.figure(figsize=(6, 4))
plt.plot(taus, tpr, "o-", color="#5f7138", label="TPR (liar caught)")
plt.plot(taus, fpr, "s-", color="#bf3b2a", label="FPR (honest flagged)")
plt.xlabel("flag threshold tau"); plt.ylabel("rate")
plt.title("Detector threshold sensitivity"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("runs/fig3_threshold.png", dpi=130); plt.close()

print("\nsaved: runs/fig1_attack_strength.png, fig2_detection_vs_sigma.png, fig3_threshold.png")
print("       runs/sweep_boost.csv, sweep_sigma.csv, sweep_tau.csv")
