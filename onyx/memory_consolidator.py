import torch
from typing import Dict, Optional, List
from tqdm import tqdm
from onyx.model import Onyx, M3Optimizer

class OnyxMemoryConsolidator:
    def __init__(
        self, 
        model: Onyx, 
        dataloader, 
        device: torch.device, 
        ewc_lambda: float = 500,
        fisher_sample_size: Optional[int] = None,
        importance_threshold: float = 1e-5
    ):
        """
        Memory consolidation with EWC for Onyx.
        
        Args:
            model: Onyx model instance
            dataloader: Training data loader
            device: Compute device
            ewc_lambda: EWC regularization strength
            fisher_sample_size: Number of batches for Fisher estimation (None = all)
            importance_threshold: Minimum Fisher value to consider (memory optimization)
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.fisher_sample_size = fisher_sample_size
        self.importance_threshold = importance_threshold
        
        # Knowledge anchors
        self.theta_star: Dict[str, torch.Tensor] = {}
        self.fisher_diagonal: Dict[str, torch.Tensor] = {}
        
        # Track trainable parameters
        self._freeze_backbone()
        
    def _freeze_backbone(self):
        """Freeze all parameters except memory-related ones."""
        trainable_params = 0
        frozen_params = 0
        
        for name, param in self.model.named_parameters():
            # Memory tensors and learning rates are trainable
            if any(key in name for key in ["memory", "eta_", "alpha_"]):
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"Trainable Parameters: {trainable_params:,} | Frozen: {frozen_params:,}")
    
    def estimate_importance(
        self, 
        memory_states: Optional[Dict] = None,
        verbose: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate Fisher Information Matrix (empirical approximation).
        
        Uses the gradient of log-likelihood to identify parameters critical
        for preserving current knowledge.
        
        Args:
            memory_states: Optional memory state dict
            verbose: Show progress bar
            
        Returns:
            Dictionary mapping parameter names to Fisher diagonal values
        """
        self.model.eval()
        fisher = {}
        
        # Only track gradients for trainable parameters
        trainable_params = {
            n: p for n, p in self.model.named_parameters() 
            if p.requires_grad
        }
        
        # Initialize Fisher accumulators
        for n, p in trainable_params.items():
            fisher[n] = torch.zeros_like(p.data)
        
        if verbose:
            print(f"Estimating Fisher Information Matrix...")
        
        # Determine number of batches to use
        num_batches = (
            min(self.fisher_sample_size, len(self.dataloader))
            if self.fisher_sample_size 
            else len(self.dataloader)
        )
        
        dataloader_iter = iter(self.dataloader)
        iterator = tqdm(range(num_batches), desc="Fisher Estimation") if verbose else range(num_batches)
        
        for _ in iterator:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
                
            self.model.zero_grad()
            
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)
            
            # Forward pass WITHOUT memory updates (pure inference)
            with torch.set_grad_enabled(True):
                out = self.model(
                    input_ids, 
                    memory_states=memory_states, 
                    update_memories=False
                )
                
                # Compute log-likelihood loss
                log_probs = torch.nn.functional.log_softmax(out["logits"], dim=-1)
                loss = torch.nn.functional.nll_loss(
                    log_probs.view(-1, log_probs.size(-1)), 
                    labels.view(-1), 
                    ignore_index=-100
                )
                
                loss.backward()
            
            # Accumulate squared gradients (Fisher diagonal approximation)
            for n, p in trainable_params.items():
                if p.grad is not None:
                    fisher[n] += (p.grad.data ** 2) / num_batches
        
        # Apply importance threshold (memory optimization)
        for n in fisher:
            fisher[n] = torch.where(
                fisher[n] > self.importance_threshold,
                fisher[n],
                torch.zeros_like(fisher[n])
            )
        
        self.model.train()
        return fisher
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization penalty.
        
        Acts as a "spring" pulling parameters back toward their
        optimal values (theta_star) weighted by importance (Fisher).
        
        Returns:
            Scalar penalty term
        """
        if not self.theta_star or not self.fisher_diagonal:
            return torch.tensor(0.0, device=self.device)
        
        penalty = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher_diagonal and p.requires_grad:
                fisher = self.fisher_diagonal[n]
                theta_star = self.theta_star[n]
                
                # Quadratic penalty weighted by importance
                penalty += (fisher * (p - theta_star) ** 2).sum()
        
        return (self.ewc_lambda / 2) * penalty
    
    def consolidate(
        self, 
        memory_states: Optional[Dict] = None,
        epochs: int = 5,
        lr: float = 1e-3,
        gradient_clip: float = 1.0,
        log_interval: int = 10,
        validate_fn: Optional[callable] = None
    ) -> Dict[str, List[float]]:
        """
        Main consolidation loop with global optimization.
        
        Args:
            memory_states: Optional memory state dict
            epochs: Number of training epochs
            lr: Learning rate
            gradient_clip: Max gradient norm (0 = no clipping)
            log_interval: Batches between progress updates
            validate_fn: Optional validation callback
            
        Returns:
            Dictionary of training metrics
        """
        # Collect trainable parameters
        trainable_params = [
            p for p in self.model.named_parameters() 
            if p[1].requires_grad
        ]
        
        optimizer = M3Optimizer([p for _, p in trainable_params], lr=lr)
        
        metrics = {
            "train_loss": [],
            "nll_loss": [],
            "ewc_loss": [],
            "val_loss": []
        }
        
        print(f"\n{'='*60}")
        print(f"Memory Consolidation (EWC λ={self.ewc_lambda})")
        print(f"{'='*60}")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_nll = 0.0
            epoch_ewc = 0.0
            epoch_total = 0.0
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                
                # Forward pass
                out = self.model(
                    batch["input_ids"].to(self.device),
                    labels=batch.get("labels", batch["input_ids"]).to(self.device),
                    memory_states=memory_states,
                    update_memories=False  # Optimizer-driven only
                )
                
                # Compute combined loss
                nll_loss = out["loss"]
                ewc_loss = self.compute_ewc_loss()
                total_loss = nll_loss + ewc_loss
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for _, p in trainable_params], 
                        gradient_clip
                    )
                
                optimizer.step()
                
                # Track metrics
                epoch_nll += nll_loss.item()
                epoch_ewc += ewc_loss.item()
                epoch_total += total_loss.item()
                
                # Update progress bar
                if batch_idx % log_interval == 0:
                    pbar.set_postfix({
                        'NLL': f'{nll_loss.item():.4f}',
                        'EWC': f'{ewc_loss.item():.4f}',
                        'Total': f'{total_loss.item():.4f}'
                    })
            
            # Epoch summary
            avg_nll = epoch_nll / len(self.dataloader)
            avg_ewc = epoch_ewc / len(self.dataloader)
            avg_total = epoch_total / len(self.dataloader)
            
            metrics["train_loss"].append(avg_total)
            metrics["nll_loss"].append(avg_nll)
            metrics["ewc_loss"].append(avg_ewc)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  NLL Loss:   {avg_nll:.4f}")
            print(f"  EWC Loss:   {avg_ewc:.4f}")
            print(f"  Total Loss: {avg_total:.4f}")
            
            # Optional validation
            if validate_fn:
                val_loss = validate_fn(self.model, memory_states)
                metrics["val_loss"].append(val_loss)
                print(f"  Val Loss:   {val_loss:.4f}")
            
            print()
        
        # Update knowledge anchors for continual learning
        print("Updating knowledge anchors...")
        self.fisher_diagonal = self.estimate_importance(memory_states, verbose=False)
        self.theta_star = {
            n: p.data.clone() 
            for n, p in self.model.named_parameters() 
            if p.requires_grad
        }
        
        print(f"✓ Consolidation complete\n")
        return metrics

    def distill(self, *args, **kwargs) -> Dict[str, List[float]]:
        return self.consolidate(*args, **kwargs)
    
    def save_checkpoint(self, path: str):
        """Save consolidator state including Fisher and theta_star."""
        checkpoint = {
            'theta_star': self.theta_star,
            'fisher_diagonal': self.fisher_diagonal,
            'ewc_lambda': self.ewc_lambda,
            'model_state': self.model.state_dict()
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load consolidator state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.theta_star = checkpoint['theta_star']
        self.fisher_diagonal = checkpoint['fisher_diagonal']
        self.ewc_lambda = checkpoint['ewc_lambda']
        self.model.load_state_dict(checkpoint['model_state'])
        print(f"Checkpoint loaded from {path}")
