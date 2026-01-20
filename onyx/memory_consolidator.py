#!/usr/bin/env python3
"""
Onyx Memory Consolidation (Memory-Knobs Only)

This module is designed for your specific intent:
- DO NOT train the whole model
- ONLY train memory-related "knobs" (eta/alpha/gate/value_gen by default)
- Optionally compute a Fisher diagonal (EWC-style) *only for those knobs*
- Or use a simpler anchor L2-to-previous (often just as good for tiny param sets)

Key fixes vs your draft:
- Uses model's own out["loss"] (correct shift + ignore_index masking)
- Includes gate_proj (+ optional value_gen) as trainable
- Handles streaming/Iterable dataloaders (requires fisher_sample_size or steps_per_epoch)
- Averages Fisher over actually used batches
- Saves/loads consolidation state safely (CPU tensors)

Usage pattern (typical):
1) Instantiate consolidator(model, dataloader, device, ...)
2) (Optional) call initialize_anchors(...) to set theta_star and fisher
3) call consolidate(...) for a few epochs/steps
4) save_checkpoint(...) if desired
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Callable, Iterable, Tuple

import torch
from torch import nn
from tqdm import tqdm

from onyx.model import Onyx, M3Optimizer


# -----------------------------
# Config
# -----------------------------

@dataclass
class ConsolidationConfig:
    # Regularization knobs
    use_fisher_ewc: bool = True         # If True, compute Fisher and use EWC penalty
    ewc_lambda: float = 500.0           # Strength of EWC penalty
    fisher_sample_size: Optional[int] = 256  # Batches for Fisher estimation (REQUIRED for Iterable/streaming)
    importance_threshold: float = 1e-5  # Zero-out tiny Fisher entries

    # Simple anchor alternative (often enough for memory knobs)
    use_anchor_l2: bool = False         # If True, use anchor L2 penalty (no Fisher)
    anchor_lambda: float = 1.0          # Strength for L2-to-theta_star (mean-squared)

    # Training loop
    lr: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    epochs: int = 3
    steps_per_epoch: Optional[int] = None  # If dataloader is streaming/infinite, set this
    log_interval: int = 25

    # What to train (by param-name substring)
    train_eta_alpha_gate: bool = True
    train_value_gen: bool = True
    train_M_init: bool = False  # usually keep False (can destabilize)


# -----------------------------
# Consolidator
# -----------------------------

class OnyxMemoryConsolidator:
    def __init__(
        self,
        model: Onyx,
        dataloader: Iterable[Dict[str, Any]],
        device: torch.device,
        cfg: Optional[ConsolidationConfig] = None,
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.cfg = cfg or ConsolidationConfig()

        # Anchors + importance
        self.theta_star: Dict[str, torch.Tensor] = {}
        self.fisher_diagonal: Dict[str, torch.Tensor] = {}

        # Freeze/unfreeze memory knobs only
        self._freeze_backbone()

    # -----------------------------
    # Param selection
    # -----------------------------

    def _is_trainable_memory_param(self, name: str) -> bool:
        """
        Decide if a parameter name should be trained in consolidation.
        Adjust this to match your architecture conventions.
        """
        # eta/alpha/gate projections live inside your memory modules
        if self.cfg.train_eta_alpha_gate:
            if ("eta_proj" in name) or ("alpha_proj" in name) or ("gate_proj" in name):
                return True

        # optional: value generator if generate_own_values=True
        if self.cfg.train_value_gen:
            if "value_gen" in name:
                return True

        # optional: allow M_init to train (usually avoid)
        if self.cfg.train_M_init:
            if name.endswith("M_init") or ".M_init" in name:
                return True

        return False

    def _freeze_backbone(self) -> None:
        trainable_params = 0
        frozen_params = 0

        for name, param in self.model.named_parameters():
            if self._is_trainable_memory_param(name):
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()

        print(f"[Consolidator] Trainable params: {trainable_params:,} | Frozen: {frozen_params:,}")

    def _trainable_named_params(self) -> List[Tuple[str, torch.Tensor]]:
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    # -----------------------------
    # Anchors
    # -----------------------------

    @torch.no_grad()
    def snapshot_theta_star(self) -> None:
        """Store current trainable memory-knob parameters as anchors."""
        self.theta_star = {}
        for n, p in self._trainable_named_params():
            self.theta_star[n] = p.detach().clone().to("cpu")
        print(f"[Consolidator] theta_star captured for {len(self.theta_star)} tensors.")

    def initialize_anchors(
        self,
        memory_states: Optional[List[Dict[str, Any]]] = None,
        compute_fisher: Optional[bool] = None,
        verbose: bool = True,
    ) -> None:
        """
        Capture theta_star and (optionally) compute fisher_diagonal for EWC.
        """
        self.snapshot_theta_star()
        do_fisher = self.cfg.use_fisher_ewc if compute_fisher is None else compute_fisher
        if do_fisher:
            self.fisher_diagonal = self.estimate_importance(memory_states=memory_states, verbose=verbose)
            print(f"[Consolidator] fisher_diagonal computed for {len(self.fisher_diagonal)} tensors.")
        else:
            self.fisher_diagonal = {}
            print("[Consolidator] fisher_diagonal skipped.")

    # -----------------------------
    # Fisher estimation
    # -----------------------------

    def _require_fisher_sample_size(self) -> int:
        if self.cfg.fisher_sample_size is None or self.cfg.fisher_sample_size <= 0:
            raise ValueError(
                "fisher_sample_size must be set (>0), especially for streaming/Iterable dataloaders."
            )
        return int(self.cfg.fisher_sample_size)

    def estimate_importance(
        self,
        memory_states: Optional[List[Dict[str, Any]]] = None,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Empirical Fisher diagonal approximation for ONLY trainable memory knobs.

        Uses squared gradients of the model's own training loss:
        - correct shift
        - ignore_index handling
        """
        self.model.eval()

        trainable = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p.data) for n, p in trainable.items()}

        num_batches = self._require_fisher_sample_size()
        it = iter(self.dataloader)
        rng = range(num_batches)
        if verbose:
            print("[Consolidator] Estimating Fisher diagonal...")
            rng = tqdm(rng, desc="Fisher")

        used = 0
        for _ in rng:
            try:
                batch = next(it)
            except StopIteration:
                break

            # Clear grads only for trainable
            for p in trainable.values():
                p.grad = None

            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            out = self.model(
                input_ids=input_ids,
                labels=labels,
                memory_states=memory_states,
                update_memories=False,
            )
            loss = out.get("loss")
            if loss is None:
                continue

            loss.backward()
            used += 1

            for n, p in trainable.items():
                if p.grad is not None:
                    fisher[n].add_(p.grad.detach().pow(2))

        if used == 0:
            # Return zeros (nothing learned)
            self.model.train()
            return fisher

        # Average + threshold
        thr = float(self.cfg.importance_threshold or 0.0)
        for n in fisher:
            fisher[n].div_(used)
            if thr > 0:
                fisher[n] = torch.where(fisher[n] > thr, fisher[n], torch.zeros_like(fisher[n]))

        self.model.train()
        return fisher

    # -----------------------------
    # Penalties
    # -----------------------------

    def compute_ewc_loss(self) -> torch.Tensor:
        """
        EWC penalty on trainable memory knobs: (lambda/2) * sum(F_i * (theta - theta*)^2)
        """
        if not self.theta_star or not self.fisher_diagonal:
            return torch.zeros((), device=self.device)

        penalty = torch.zeros((), device=self.device)
        lam = float(self.cfg.ewc_lambda)

        for n, p in self._trainable_named_params():
            if n not in self.fisher_diagonal or n not in self.theta_star:
                continue
            F = self.fisher_diagonal[n].to(device=self.device, dtype=p.dtype)
            th0 = self.theta_star[n].to(device=self.device, dtype=p.dtype)
            penalty = penalty + (F * (p - th0).pow(2)).sum()

        return (lam / 2.0) * penalty

    def compute_anchor_l2_loss(self) -> torch.Tensor:
        """
        Simple L2-to-anchor (mean-squared) penalty:
          anchor_lambda * mean((theta - theta*)^2)
        Often sufficient for tiny param sets, cheaper than Fisher.
        """
        if not self.theta_star:
            return torch.zeros((), device=self.device)

        lam = float(self.cfg.anchor_lambda)
        penalty = torch.zeros((), device=self.device)
        count = 0

        for n, p in self._trainable_named_params():
            if n not in self.theta_star:
                continue
            th0 = self.theta_star[n].to(device=self.device, dtype=p.dtype)
            penalty = penalty + (p - th0).pow(2).mean()
            count += 1

        if count == 0:
            return torch.zeros((), device=self.device)
        return lam * penalty

    # -----------------------------
    # Consolidation loop
    # -----------------------------

    def consolidate(
        self,
        memory_states: Optional[List[Dict[str, Any]]] = None,
        validate_fn: Optional[Callable[[Onyx, Optional[List[Dict[str, Any]]]], float]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train ONLY the memory knobs on dataloader with optional regularization to anchors.
        """
        trainable = self._trainable_named_params()
        if not trainable:
            raise RuntimeError("No trainable memory parameters found. Check name filters.")

        params_only = [p for _, p in trainable]
        optimizer = M3Optimizer(
            params_only,
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        metrics = {"train_loss": [], "nll_loss": [], "reg_loss": [], "val_loss": []}

        epochs = int(self.cfg.epochs)
        steps_cap = self.cfg.steps_per_epoch

        print("\n" + "=" * 70)
        reg_mode = "EWC" if self.cfg.use_fisher_ewc else ("AnchorL2" if self.cfg.use_anchor_l2 else "None")
        print(f"[Consolidator] Consolidation: epochs={epochs} lr={self.cfg.lr} reg={reg_mode}")
        print("=" * 70)

        for ep in range(epochs):
            self.model.train()

            ep_nll = 0.0
            ep_reg = 0.0
            ep_total = 0.0
            used_steps = 0

            iterator = iter(self.dataloader)
            if steps_cap is None:
                # If dataloader has __len__, use it; otherwise require steps_per_epoch
                try:
                    steps_cap = len(self.dataloader)  # type: ignore
                except Exception:
                    raise ValueError("steps_per_epoch must be set for streaming/Iterable dataloaders.")

            step_range = range(int(steps_cap))
            if verbose:
                step_range = tqdm(step_range, desc=f"Epoch {ep+1}/{epochs}")

            for step_idx in step_range:
                try:
                    batch = next(iterator)
                except StopIteration:
                    break

                optimizer.zero_grad(set_to_none=True)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)

                out = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    memory_states=memory_states,
                    update_memories=False,
                )
                nll = out.get("loss")
                if nll is None:
                    continue

                # Regularization term
                if self.cfg.use_fisher_ewc and self.theta_star and self.fisher_diagonal:
                    reg = self.compute_ewc_loss()
                elif self.cfg.use_anchor_l2 and self.theta_star:
                    reg = self.compute_anchor_l2_loss()
                else:
                    reg = torch.zeros((), device=self.device)

                total = nll + reg
                total.backward()

                if self.cfg.gradient_clip and self.cfg.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params_only, float(self.cfg.gradient_clip))

                optimizer.step()

                ep_nll += float(nll.detach().item())
                ep_reg += float(reg.detach().item())
                ep_total += float(total.detach().item())
                used_steps += 1

                if verbose and self.cfg.log_interval and (step_idx % int(self.cfg.log_interval) == 0):
                    if hasattr(step_range, "set_postfix"):
                        step_range.set_postfix({
                            "nll": f"{float(nll.detach().item()):.4f}",
                            "reg": f"{float(reg.detach().item()):.4f}",
                            "tot": f"{float(total.detach().item()):.4f}",
                        })

            denom = max(1, used_steps)
            avg_nll = ep_nll / denom
            avg_reg = ep_reg / denom
            avg_total = ep_total / denom

            metrics["nll_loss"].append(avg_nll)
            metrics["reg_loss"].append(avg_reg)
            metrics["train_loss"].append(avg_total)

            print(f"\n[Epoch {ep+1}] nll={avg_nll:.4f} reg={avg_reg:.4f} total={avg_total:.4f}")

            if validate_fn is not None:
                try:
                    val = float(validate_fn(self.model, memory_states))
                    metrics["val_loss"].append(val)
                    print(f"[Epoch {ep+1}] val={val:.4f}")
                except Exception as e:
                    print(f"[Epoch {ep+1}] validate_fn failed: {e}")

        # Refresh anchors at the end (optional, but useful for continual phases)
        print("[Consolidator] Updating theta_star and fisher...")
        self.snapshot_theta_star()
        if self.cfg.use_fisher_ewc:
            self.fisher_diagonal = self.estimate_importance(memory_states=memory_states, verbose=False)

        print("[Consolidator] ✓ Consolidation complete\n")
        return metrics

    # Convenience alias
    def distill(self, *args, **kwargs):
        return self.consolidate(*args, **kwargs)

    # -----------------------------
    # Save/load
    # -----------------------------

    def save_checkpoint(self, path: str) -> None:
        """
        Save:
        - model_state (ONLY current model weights)
        - theta_star (CPU tensors)
        - fisher_diagonal (CPU tensors)
        - cfg (as dict)
        """
        def to_cpu(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            return {k: v.detach().to("cpu") for k, v in d.items()}

        ckpt = {
            "model_state": self.model.state_dict(),
            "theta_star": to_cpu(self.theta_star) if self.theta_star else {},
            "fisher_diagonal": to_cpu(self.fisher_diagonal) if self.fisher_diagonal else {},
            "cfg": self.cfg.__dict__,
        }
        torch.save(ckpt, path)
        print(f"[Consolidator] Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Restore cfg if present
        cfg_dict = ckpt.get("cfg")
        if isinstance(cfg_dict, dict):
            self.cfg = ConsolidationConfig(**cfg_dict)

        # Reload model weights
        ms = ckpt.get("model_state")
        if isinstance(ms, dict):
            self.model.load_state_dict(ms, strict=True)

        # Restore anchors
        self.theta_star = {}
        ts = ckpt.get("theta_star", {})
        if isinstance(ts, dict):
            self.theta_star = {k: v.detach().to("cpu") for k, v in ts.items() if torch.is_tensor(v)}

        self.fisher_diagonal = {}
        fd = ckpt.get("fisher_diagonal", {})
        if isinstance(fd, dict):
            self.fisher_diagonal = {k: v.detach().to("cpu") for k, v in fd.items() if torch.is_tensor(v)}

        # Re-freeze with the possibly restored config
        self._freeze_backbone()
        print(f"[Consolidator] Loaded checkpoint: {path}")
