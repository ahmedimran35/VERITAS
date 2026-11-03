"""GRPO-on-LLM client (GPU phase — stub).

Wraps a local GRPO step and emits the federated report. Fill in with TRL's
GRPOTrainer (or verl) when moving to the real model. Kept deliberately thin so
the attack/defense logic in src/veritas stays model-agnostic.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ClientConfig:
    client_id: int
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    group_size: int = 8          # answers sampled per prompt (GRPO group)
    local_steps: int = 1
    is_malicious: bool = False


class GRPOClient:
    def __init__(self, cfg: ClientConfig):
        self.cfg = cfg
        # TODO(GPU): load model + tokenizer; build TRL GRPOTrainer with a
        # rule-based reward (math-verify / exact-match) on this client's shard.

    def local_round(self, global_state, global_mu: float) -> dict:
        """Run one local GRPO step; return the federated report.

        Returns {"mu", "sigma", "update"} where:
          mu, sigma  = mean / std of this round's group rewards
          update     = parameter delta (or LoRA delta) vs the broadcast model
        """
        raise NotImplementedError(
            "GPU phase: run GRPOTrainer locally, then compute group-reward "
            "mu/sigma and the parameter delta. The honest report feeds "
            "src.veritas.aggregate; a malicious client applies src.veritas.attacks."
        )

    @staticmethod
    def report_from(mu: float, sigma: float, update: np.ndarray) -> dict:
        return {"mu": float(mu), "sigma": float(sigma),
                "update": np.asarray(update, float)}
