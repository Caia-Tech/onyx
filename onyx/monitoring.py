#!/usr/bin/env python3
"""
Monitoring tools for detecting mode collapse and training issues.

Key metrics:
- Token diversity (top-K mass, entropy, effective vocabulary)
- Memory states (optional - norms, update magnitudes)
"""

import math
from typing import Optional, Dict, Any, List
from collections import deque

import torch
import torch.nn.functional as F


class DiversityMonitor:
    """
    Monitors token diversity and mode collapse indicators.

    Essential metrics:
    - Top-K mass: What % of probability is in top-K tokens?
    - Entropy: How diverse are the predictions?
    - Effective vocabulary: How many tokens are actually used?

    Usage:
        monitor = DiversityMonitor(config, tokenizer)

        # In training loop:
        if monitor.should_monitor(step):
            metrics = monitor.compute_metrics(logits, step)
            if metrics:
                # Log to wandb, console, etc.
                warnings = monitor.check_alerts(metrics, step)
                for warning in warnings:
                    print(warning)
    """

    def __init__(
        self,
        vocab_size: int,
        monitor_every: int = 50,
        alert_top10_mass: float = 0.7,
        alert_entropy_ratio: float = 0.3,
        alert_effective_vocab: int = 100,
        history_size: int = 100,
        tokenizer = None,
    ):
        """
        Initialize diversity monitor.

        Args:
            vocab_size: Size of vocabulary
            monitor_every: Monitor every N steps (default: 50)
            alert_top10_mass: Alert if top-10 tokens exceed this fraction (default: 0.7)
            alert_entropy_ratio: Alert if entropy ratio below this (default: 0.3)
            alert_effective_vocab: Alert if effective vocab below this (default: 100)
            history_size: How many measurements to keep in history (default: 100)
            tokenizer: Optional tokenizer for decoding token IDs
        """
        self.vocab_size = vocab_size
        self.monitor_every = monitor_every
        self.tokenizer = tokenizer

        # Alert thresholds
        self.alert_top10_mass = alert_top10_mass
        self.alert_entropy_ratio = alert_entropy_ratio
        self.alert_effective_vocab = alert_effective_vocab

        # History for trend detection (use deque for efficient memory)
        self.history: Dict[str, deque] = {
            "top10_mass": deque(maxlen=history_size),
            "entropy": deque(maxlen=history_size),
            "effective_vocab": deque(maxlen=history_size),
        }

    def should_monitor(self, step: int) -> bool:
        """Check if we should monitor this step."""
        return step > 0 and step % self.monitor_every == 0

    @torch.no_grad()
    def compute_metrics(self, logits: torch.Tensor, step: int) -> Optional[Dict[str, Any]]:
        """
        Compute diversity metrics from logits.

        Args:
            logits: [B, S, V] tensor of logits
            step: Current training step

        Returns:
            Dictionary of metrics, or None if not monitoring this step
        """
        if not self.should_monitor(step):
            return None

        # Detach and move to float32 for numerical stability
        logits = logits.detach().float()

        # Compute probabilities and average over batch and sequence
        # This gives us the "typical" token distribution
        probs = F.softmax(logits, dim=-1)
        avg_probs = probs.mean(dim=(0, 1))  # [V]

        metrics = {}

        # 1. Top-K mass (what % of probability is in top-K tokens?)
        topk_result = avg_probs.topk(k=min(100, self.vocab_size))
        topk_values = topk_result.values
        topk_indices = topk_result.indices

        metrics["top10_mass"] = float(topk_values[:10].sum().item())
        metrics["top50_mass"] = float(topk_values[:50].sum().item())
        metrics["top100_mass"] = float(topk_values[:100].sum().item())

        # 2. Entropy (H(p) = -sum(p * log(p)))
        # Use torch.where to avoid log(0)
        log_probs = torch.where(
            avg_probs > 1e-10,
            torch.log(avg_probs),
            torch.zeros_like(avg_probs)
        )
        entropy = -(avg_probs * log_probs).sum().item()
        max_entropy = math.log(self.vocab_size)

        metrics["entropy"] = float(entropy)
        metrics["entropy_ratio"] = float(entropy / max_entropy)
        metrics["max_entropy"] = float(max_entropy)

        # 3. Effective vocabulary (how many tokens have non-negligible probability?)
        # Threshold: 0.001 = 0.1% probability
        metrics["effective_vocab"] = int((avg_probs > 0.001).sum().item())

        # 4. Top token IDs and strings (for inspection)
        top10_ids = topk_indices[:10].cpu().tolist()
        metrics["top10_token_ids"] = top10_ids

        if self.tokenizer is not None:
            try:
                # Decode top-5 tokens for quick inspection
                top5_strings = []
                for idx in top10_ids[:5]:
                    token_str = self.tokenizer.decode([idx])
                    # Clean up whitespace for display
                    token_str = repr(token_str)  # Show escapes like \n
                    top5_strings.append(token_str)
                metrics["top5_token_strings"] = top5_strings
            except Exception:
                # If decoding fails, just skip
                pass

        # Track history for trend detection
        self.history["top10_mass"].append(metrics["top10_mass"])
        self.history["entropy"].append(metrics["entropy"])
        self.history["effective_vocab"].append(metrics["effective_vocab"])

        return metrics

    def check_alerts(self, metrics: Optional[Dict[str, Any]], step: int) -> List[str]:
        """
        Check if any metrics cross alert thresholds.

        Args:
            metrics: Metrics dictionary from compute_metrics
            step: Current training step

        Returns:
            List of warning strings (empty if no alerts)
        """
        if not metrics:
            return []

        warnings = []

        # Alert 1: Too much mass in top-10 tokens
        if metrics["top10_mass"] > self.alert_top10_mass:
            warnings.append(
                f"[MODE COLLAPSE WARNING @ step {step}] "
                f"{metrics['top10_mass']*100:.1f}% probability mass in top-10 tokens "
                f"(threshold: {self.alert_top10_mass*100:.1f}%)"
            )

        # Alert 2: Entropy too low
        if metrics["entropy_ratio"] < self.alert_entropy_ratio:
            warnings.append(
                f"[MODE COLLAPSE WARNING @ step {step}] "
                f"Entropy ratio {metrics['entropy_ratio']:.3f} too low "
                f"(threshold: {self.alert_entropy_ratio:.3f})"
            )

        # Alert 3: Effective vocabulary shrinking
        if metrics["effective_vocab"] < self.alert_effective_vocab:
            warnings.append(
                f"[MODE COLLAPSE WARNING @ step {step}] "
                f"Only {metrics['effective_vocab']} tokens actively used "
                f"(threshold: {self.alert_effective_vocab})"
            )

        # Alert 4: Rapid trend toward collapse
        if len(self.history["top10_mass"]) >= 5:
            recent = list(self.history["top10_mass"])[-5:]
            if recent[-1] > recent[0] + 0.15:  # 15% increase over 5 measurements
                warnings.append(
                    f"[TREND WARNING @ step {step}] "
                    f"Top-10 mass increasing rapidly: "
                    f"{recent[0]:.3f} → {recent[-1]:.3f} (Δ={recent[-1]-recent[0]:.3f})"
                )

        return warnings


class MemoryMonitor:
    """
    Optional monitor for memory state diagnostics.

    Tracks:
    - Memory matrix norms per layer
    - Memory update magnitudes per layer

    Usage:
        monitor = MemoryMonitor(num_layers)

        # In training loop (after model forward):
        if monitor.should_monitor(step):
            metrics = monitor.compute_metrics(memory_states, step)
    """

    def __init__(
        self,
        num_layers: int,
        monitor_every: int = 50,
        alert_norm_threshold: float = 100.0,
        history_size: int = 100,
    ):
        """
        Initialize memory monitor.

        Args:
            num_layers: Number of model layers
            monitor_every: Monitor every N steps
            alert_norm_threshold: Alert if any memory norm exceeds this
            history_size: How many measurements to keep
        """
        self.num_layers = num_layers
        self.monitor_every = monitor_every
        self.alert_norm_threshold = alert_norm_threshold

        # Track previous memory states for computing updates
        self.prev_memory_states: Optional[List[Dict[str, Any]]] = None

        # History per layer
        self.history: Dict[str, List[deque]] = {
            "norms": [deque(maxlen=history_size) for _ in range(num_layers)],
            "update_mags": [deque(maxlen=history_size) for _ in range(num_layers)],
        }

    def should_monitor(self, step: int) -> bool:
        """Check if we should monitor this step."""
        return step > 0 and step % self.monitor_every == 0

    @torch.no_grad()
    def compute_metrics(
        self,
        memory_states: List[Dict[str, Any]],
        step: int
    ) -> Optional[Dict[str, Any]]:
        """
        Compute memory state metrics.

        Args:
            memory_states: List of memory state dicts (one per layer)
            step: Current training step

        Returns:
            Dictionary of metrics per layer, or None if not monitoring
        """
        if not self.should_monitor(step):
            return None

        metrics = {
            "norms": [],
            "update_mags": [],
        }

        for layer_idx, layer_state in enumerate(memory_states):
            # Extract memory matrices (could be in 'attention' dict)
            if "attention" in layer_state:
                attn_state = layer_state["attention"]
            else:
                attn_state = layer_state

            # Compute norms for key and value memories (if they exist)
            layer_norms = []
            if "k" in attn_state and torch.is_tensor(attn_state["k"]):
                k_norm = float(torch.norm(attn_state["k"]).item())
                layer_norms.append(k_norm)
            if "v" in attn_state and torch.is_tensor(attn_state["v"]):
                v_norm = float(torch.norm(attn_state["v"]).item())
                layer_norms.append(v_norm)

            # Average norm for this layer
            avg_norm = sum(layer_norms) / len(layer_norms) if layer_norms else 0.0
            metrics["norms"].append(avg_norm)
            self.history["norms"][layer_idx].append(avg_norm)

            # Compute update magnitude (if we have previous state)
            if self.prev_memory_states is not None:
                prev_state = self.prev_memory_states[layer_idx]
                if "attention" in prev_state:
                    prev_attn = prev_state["attention"]
                else:
                    prev_attn = prev_state

                layer_update_mags = []
                if "k" in attn_state and "k" in prev_attn:
                    if torch.is_tensor(attn_state["k"]) and torch.is_tensor(prev_attn["k"]):
                        k_update = float(torch.norm(attn_state["k"] - prev_attn["k"]).item())
                        layer_update_mags.append(k_update)
                if "v" in attn_state and "v" in prev_attn:
                    if torch.is_tensor(attn_state["v"]) and torch.is_tensor(prev_attn["v"]):
                        v_update = float(torch.norm(attn_state["v"] - prev_attn["v"]).item())
                        layer_update_mags.append(v_update)

                avg_update = sum(layer_update_mags) / len(layer_update_mags) if layer_update_mags else 0.0
                metrics["update_mags"].append(avg_update)
                self.history["update_mags"][layer_idx].append(avg_update)
            else:
                metrics["update_mags"].append(0.0)

        # Deep copy memory states for next comparison
        # We need to detach and clone tensors
        self.prev_memory_states = []
        for layer_state in memory_states:
            layer_copy = {}
            for k, v in layer_state.items():
                if isinstance(v, dict):
                    layer_copy[k] = {
                        kk: vv.detach().clone() if torch.is_tensor(vv) else vv
                        for kk, vv in v.items()
                    }
                elif torch.is_tensor(v):
                    layer_copy[k] = v.detach().clone()
                else:
                    layer_copy[k] = v
            self.prev_memory_states.append(layer_copy)

        return metrics

    def check_alerts(self, metrics: Optional[Dict[str, Any]], step: int) -> List[str]:
        """
        Check if any memory metrics are problematic.

        Args:
            metrics: Metrics dictionary from compute_metrics
            step: Current training step

        Returns:
            List of warning strings
        """
        if not metrics:
            return []

        warnings = []

        # Alert if any layer's norm is exploding
        for layer_idx, norm in enumerate(metrics["norms"]):
            if norm > self.alert_norm_threshold:
                warnings.append(
                    f"[MEMORY WARNING @ step {step}] "
                    f"Layer {layer_idx} memory norm {norm:.1f} exceeds threshold "
                    f"{self.alert_norm_threshold:.1f}"
                )

        # Alert if norms are rapidly increasing
        for layer_idx in range(self.num_layers):
            if len(self.history["norms"][layer_idx]) >= 3:
                recent = list(self.history["norms"][layer_idx])[-3:]
                if recent[-1] > recent[0] * 2.0:  # Doubled in 3 measurements
                    warnings.append(
                        f"[MEMORY WARNING @ step {step}] "
                        f"Layer {layer_idx} memory norm doubling: "
                        f"{recent[0]:.1f} → {recent[-1]:.1f}"
                    )

        return warnings
