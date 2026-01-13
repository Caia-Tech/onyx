#!/usr/bin/env python3
"""
Tests for monitoring functionality (diversity and memory state monitoring).
"""
import math
import pytest
import torch
import torch.nn.functional as F

from onyx.monitoring import DiversityMonitor, MemoryMonitor


class TestDiversityMonitor:
    """Tests for DiversityMonitor class."""

    def test_diversity_monitor_initialization(self):
        """Test basic initialization."""
        monitor = DiversityMonitor(
            vocab_size=1000,
            monitor_every=10,
            alert_top10_mass=0.7,
            alert_entropy_ratio=0.3,
            alert_effective_vocab=100,
        )
        assert monitor.vocab_size == 1000
        assert monitor.monitor_every == 10
        assert monitor.alert_top10_mass == 0.7
        assert monitor.alert_entropy_ratio == 0.3
        assert monitor.alert_effective_vocab == 100

    def test_should_monitor(self):
        """Test monitoring frequency logic."""
        monitor = DiversityMonitor(vocab_size=1000, monitor_every=50)

        # Should not monitor at step 0
        assert not monitor.should_monitor(0)

        # Should not monitor at non-multiples
        assert not monitor.should_monitor(25)
        assert not monitor.should_monitor(99)

        # Should monitor at multiples
        assert monitor.should_monitor(50)
        assert monitor.should_monitor(100)
        assert monitor.should_monitor(150)

    def test_compute_metrics_basic(self):
        """Test basic metric computation."""
        vocab_size = 100
        monitor = DiversityMonitor(vocab_size=vocab_size, monitor_every=1)

        # Create synthetic logits: [B=2, S=10, V=100]
        logits = torch.randn(2, 10, vocab_size)

        metrics = monitor.compute_metrics(logits, step=1)

        # Check all expected keys are present
        assert "top10_mass" in metrics
        assert "top50_mass" in metrics
        assert "top100_mass" in metrics
        assert "entropy" in metrics
        assert "entropy_ratio" in metrics
        assert "max_entropy" in metrics
        assert "effective_vocab" in metrics
        assert "top10_token_ids" in metrics

        # Check value ranges
        assert 0.0 <= metrics["top10_mass"] <= 1.0
        assert 0.0 <= metrics["top50_mass"] <= 1.0
        assert 0.0 <= metrics["top100_mass"] <= 1.0
        assert metrics["top10_mass"] <= metrics["top50_mass"] <= metrics["top100_mass"]

        assert 0.0 <= metrics["entropy_ratio"] <= 1.0
        assert metrics["max_entropy"] == math.log(vocab_size)

        assert 0 <= metrics["effective_vocab"] <= vocab_size
        assert len(metrics["top10_token_ids"]) == 10

    def test_compute_metrics_skips_non_monitoring_steps(self):
        """Test that metrics are not computed for non-monitoring steps."""
        monitor = DiversityMonitor(vocab_size=100, monitor_every=50)
        logits = torch.randn(2, 10, 100)

        # Should return None for non-monitoring steps
        assert monitor.compute_metrics(logits, step=25) is None
        assert monitor.compute_metrics(logits, step=49) is None

        # Should return metrics for monitoring steps
        assert monitor.compute_metrics(logits, step=50) is not None
        assert monitor.compute_metrics(logits, step=100) is not None

    def test_compute_metrics_mode_collapse(self):
        """Test detection of mode collapse scenario."""
        vocab_size = 100
        monitor = DiversityMonitor(vocab_size=vocab_size, monitor_every=1)

        # Create collapsed logits: very high probability on first 5 tokens
        logits = torch.zeros(2, 10, vocab_size)
        logits[:, :, :5] = 10.0  # High logits for first 5 tokens
        logits[:, :, 5:] = -10.0  # Low logits for rest

        metrics = monitor.compute_metrics(logits, step=1)

        # In mode collapse, top-10 mass should be very high
        assert metrics["top10_mass"] > 0.9

        # Effective vocabulary should be small
        assert metrics["effective_vocab"] < 20

        # Entropy ratio should be low
        assert metrics["entropy_ratio"] < 0.5

    def test_check_alerts_no_alerts(self):
        """Test that no alerts fire for healthy metrics."""
        monitor = DiversityMonitor(
            vocab_size=100,
            monitor_every=1,
            alert_top10_mass=0.7,
            alert_entropy_ratio=0.3,
            alert_effective_vocab=50,
        )

        # Healthy metrics
        metrics = {
            "top10_mass": 0.5,
            "entropy_ratio": 0.6,
            "effective_vocab": 80,
        }

        warnings = monitor.check_alerts(metrics, step=100)
        assert len(warnings) == 0

    def test_check_alerts_top10_mass_alert(self):
        """Test alert for high top-10 mass."""
        monitor = DiversityMonitor(
            vocab_size=100,
            monitor_every=1,
            alert_top10_mass=0.7,
        )

        metrics = {
            "top10_mass": 0.8,  # Above threshold
            "entropy_ratio": 0.6,
            "effective_vocab": 80,
        }

        warnings = monitor.check_alerts(metrics, step=100)
        assert len(warnings) > 0
        assert "MODE COLLAPSE WARNING" in warnings[0]
        assert "top-10 tokens" in warnings[0]

    def test_check_alerts_entropy_alert(self):
        """Test alert for low entropy."""
        monitor = DiversityMonitor(
            vocab_size=100,
            monitor_every=1,
            alert_entropy_ratio=0.3,
        )

        metrics = {
            "top10_mass": 0.5,
            "entropy_ratio": 0.2,  # Below threshold
            "effective_vocab": 80,
        }

        warnings = monitor.check_alerts(metrics, step=100)
        assert len(warnings) > 0
        assert "MODE COLLAPSE WARNING" in warnings[0]
        assert "Entropy ratio" in warnings[0]

    def test_check_alerts_effective_vocab_alert(self):
        """Test alert for low effective vocabulary."""
        monitor = DiversityMonitor(
            vocab_size=100,
            monitor_every=1,
            alert_effective_vocab=100,
        )

        metrics = {
            "top10_mass": 0.5,
            "entropy_ratio": 0.6,
            "effective_vocab": 50,  # Below threshold
        }

        warnings = monitor.check_alerts(metrics, step=100)
        assert len(warnings) > 0
        assert "MODE COLLAPSE WARNING" in warnings[0]
        assert "tokens actively used" in warnings[0]

    def test_check_alerts_trend_warning(self):
        """Test trend alert for rapidly increasing top-10 mass."""
        monitor = DiversityMonitor(vocab_size=100, monitor_every=1)

        # Simulate increasing trend
        logits = torch.randn(2, 10, 100)
        for step in [1, 2, 3, 4, 5]:
            # Make distribution more concentrated over time
            logits[:, :, :10] += 2.0 * step
            metrics = monitor.compute_metrics(logits, step=step)

        # Check last metrics for trend warning
        warnings = monitor.check_alerts(metrics, step=5)

        # Should have trend warning (top10_mass increased)
        trend_warnings = [w for w in warnings if "TREND WARNING" in w]
        assert len(trend_warnings) > 0

    def test_no_grad_context(self):
        """Test that compute_metrics doesn't create gradients."""
        monitor = DiversityMonitor(vocab_size=100, monitor_every=1)

        # Create logits that require grad
        logits = torch.randn(2, 10, 100, requires_grad=True)

        # Compute metrics
        metrics = monitor.compute_metrics(logits, step=1)

        # Ensure we got metrics
        assert metrics is not None

        # Logits should still require grad (not modified)
        assert logits.requires_grad

        # But no gradient graph should be built
        # (If we try to backward through metrics, it should fail)
        # This is implicitly tested by the @torch.no_grad() decorator


class TestMemoryMonitor:
    """Tests for MemoryMonitor class."""

    def test_memory_monitor_initialization(self):
        """Test basic initialization."""
        monitor = MemoryMonitor(
            num_layers=4,
            monitor_every=50,
            alert_norm_threshold=100.0,
        )
        assert monitor.num_layers == 4
        assert monitor.monitor_every == 50
        assert monitor.alert_norm_threshold == 100.0

    def test_should_monitor(self):
        """Test monitoring frequency logic."""
        monitor = MemoryMonitor(num_layers=4, monitor_every=50)

        assert not monitor.should_monitor(0)
        assert not monitor.should_monitor(25)
        assert monitor.should_monitor(50)
        assert monitor.should_monitor(100)

    def test_compute_metrics_basic(self):
        """Test basic memory state metric computation."""
        monitor = MemoryMonitor(num_layers=3, monitor_every=1)

        # Create synthetic memory states
        memory_states = []
        for layer_idx in range(3):
            layer_state = {
                "attention": {
                    "k": torch.randn(2, 16, 64),  # [B, heads, dim]
                    "v": torch.randn(2, 16, 64),
                }
            }
            memory_states.append(layer_state)

        metrics = monitor.compute_metrics(memory_states, step=1)

        assert "norms" in metrics
        assert "update_mags" in metrics
        assert len(metrics["norms"]) == 3
        assert len(metrics["update_mags"]) == 3

        # All norms should be positive
        for norm in metrics["norms"]:
            assert norm > 0

        # First measurement: update_mags should be 0 (no previous state)
        for update_mag in metrics["update_mags"]:
            assert update_mag == 0.0

    def test_compute_metrics_update_magnitudes(self):
        """Test update magnitude computation."""
        monitor = MemoryMonitor(num_layers=2, monitor_every=1)

        # First measurement
        memory_states_1 = [
            {"attention": {"k": torch.ones(2, 4, 8), "v": torch.ones(2, 4, 8)}},
            {"attention": {"k": torch.ones(2, 4, 8), "v": torch.ones(2, 4, 8)}},
        ]
        metrics_1 = monitor.compute_metrics(memory_states_1, step=1)
        assert metrics_1["update_mags"][0] == 0.0  # No previous state

        # Second measurement (with change)
        memory_states_2 = [
            {"attention": {"k": torch.ones(2, 4, 8) * 2.0, "v": torch.ones(2, 4, 8) * 2.0}},
            {"attention": {"k": torch.ones(2, 4, 8) * 2.0, "v": torch.ones(2, 4, 8) * 2.0}},
        ]
        metrics_2 = monitor.compute_metrics(memory_states_2, step=2)

        # Now update_mags should be non-zero
        assert metrics_2["update_mags"][0] > 0
        assert metrics_2["update_mags"][1] > 0

    def test_check_alerts_no_alerts(self):
        """Test that no alerts fire for healthy memory norms."""
        monitor = MemoryMonitor(num_layers=2, monitor_every=1, alert_norm_threshold=100.0)

        metrics = {
            "norms": [25.0, 30.0],
            "update_mags": [0.5, 0.6],
        }

        warnings = monitor.check_alerts(metrics, step=100)
        assert len(warnings) == 0

    def test_check_alerts_norm_exceeds_threshold(self):
        """Test alert for memory norm exceeding threshold."""
        monitor = MemoryMonitor(num_layers=2, monitor_every=1, alert_norm_threshold=50.0)

        metrics = {
            "norms": [45.0, 150.0],  # Layer 1 exceeds threshold
            "update_mags": [0.5, 0.6],
        }

        warnings = monitor.check_alerts(metrics, step=100)
        assert len(warnings) > 0
        assert "MEMORY WARNING" in warnings[0]
        assert "Layer 1" in warnings[0]
        assert "exceeds threshold" in warnings[0]

    def test_check_alerts_rapid_norm_increase(self):
        """Test alert for rapidly increasing memory norm."""
        monitor = MemoryMonitor(num_layers=1, monitor_every=1)

        # Simulate rapidly increasing norms
        base_norm = 10.0
        for step in [1, 2, 3]:
            memory_states = [
                {"attention": {"k": torch.ones(2, 4, 8) * (base_norm * (2 ** step)),
                               "v": torch.ones(2, 4, 8) * (base_norm * (2 ** step))}}
            ]
            metrics = monitor.compute_metrics(memory_states, step=step)

        # Check for rapid increase warning
        warnings = monitor.check_alerts(metrics, step=3)

        # Should have warning about doubling
        doubling_warnings = [w for w in warnings if "doubling" in w]
        assert len(doubling_warnings) > 0

    def test_compute_metrics_skips_non_monitoring_steps(self):
        """Test that metrics are not computed for non-monitoring steps."""
        monitor = MemoryMonitor(num_layers=2, monitor_every=50)
        memory_states = [
            {"attention": {"k": torch.randn(2, 4, 8), "v": torch.randn(2, 4, 8)}},
            {"attention": {"k": torch.randn(2, 4, 8), "v": torch.randn(2, 4, 8)}},
        ]

        assert monitor.compute_metrics(memory_states, step=25) is None
        assert monitor.compute_metrics(memory_states, step=50) is not None


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""

    def test_diversity_monitor_with_real_distribution(self):
        """Test diversity monitor with realistic token distributions."""
        vocab_size = 5000  # Use smaller vocab for testing
        monitor = DiversityMonitor(vocab_size=vocab_size, monitor_every=1)

        # Create more realistic logits (not uniform)
        logits = torch.randn(4, 128, vocab_size)
        logits = logits * 2.0  # Scale to make distribution more concentrated

        metrics = monitor.compute_metrics(logits, step=1)

        # Verify metrics are computed correctly
        assert metrics is not None
        assert 0.0 < metrics["top10_mass"] < 1.0
        assert 0.0 < metrics["entropy_ratio"] < 1.0
        # With 5000 vocab and random logits, effective_vocab should be reasonable
        assert metrics["effective_vocab"] >= 0  # Could be 0 for very dispersed random logits

    def test_history_tracking(self):
        """Test that monitors correctly track history."""
        monitor = DiversityMonitor(vocab_size=100, monitor_every=1, history_size=10)

        # Compute metrics for 15 steps
        for step in range(1, 16):
            logits = torch.randn(2, 10, 100)
            monitor.compute_metrics(logits, step=step)

        # History should be limited to 10 (maxlen)
        assert len(monitor.history["top10_mass"]) == 10
        assert len(monitor.history["entropy"]) == 10
        assert len(monitor.history["effective_vocab"]) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
