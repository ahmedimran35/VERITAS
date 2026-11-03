"""GRPO federated server (GPU phase — stub).

The aggregation logic is already real in src/veritas/aggregate.py; this server
just drives the rounds and broadcasts the robust global state + global mu.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from veritas.aggregate import veritas_aggregate, plain_aggregate  # noqa: E402
from veritas.robust import Reputation                              # noqa: E402


class GRPOServer:
    def __init__(self, n_clients: int, defense: str = "veritas", cfg: dict | None = None):
        self.defense = defense
        self.cfg = cfg or {}
        self.reputation = Reputation(n_clients) if defense == "veritas" else None

    def aggregate(self, reports: list[dict]) -> dict:
        if self.defense == "veritas":
            return veritas_aggregate(reports, self.cfg, self.reputation)
        return plain_aggregate(reports)

    # TODO(GPU): apply out["update"] to the global model, broadcast the new
    # weights + out["mu"] to clients, log out["flags"] / out["weights"].
