"""
Training Configuration Classes

Contains dataclasses for LoRA and training configurations.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) training.

    Attributes:
        r: LoRA rank (dimension of low-rank matrices)
        alpha: LoRA scaling factor (alpha/r determines the scaling)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Whether to train bias parameters ("none", "all", or "lora_only")
    """
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"

    def to_dict(self):
        """Convert to dictionary for PEFT config."""
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
        }


@dataclass
class LoKRConfig:
    """Configuration for LoKr (Low-Rank Kronecker) training."""

    linear_dim: int = 64
    linear_alpha: int = 128
    factor: int = -1
    decompose_both: bool = False
    use_tucker: bool = False
    use_scalar: bool = False
    weight_decompose: bool = False
    # Crucible patch (2026-06-07): LyCORIS regularization dropouts for LoKr
    # (METAL_LORA_PLAN §13g Tier 2). These are standard LyCORIS network kwargs:
    #   dropout        - output dropout on the adapter delta
    #   rank_dropout   - drops rank components during training
    #   module_dropout - randomly skips whole adapter modules per step
    # ALL default 0.0 = byte-identical to every prior run (no-regression,
    # [[optional-additions]]). Opt in (e.g. dropout 0.1) to regularize an
    # over-geared / "fried" adapter that breaks up at usable strength.
    dropout: float = 0.0
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    full_matrix: bool = False
    bypass_mode: bool = False
    rs_lora: bool = False
    unbalanced_factorization: bool = False

    def to_dict(self):
        """Convert to dictionary for LyCORIS config."""
        return {
            "linear_dim": self.linear_dim,
            "linear_alpha": self.linear_alpha,
            "factor": self.factor,
            "decompose_both": self.decompose_both,
            "use_tucker": self.use_tucker,
            "use_scalar": self.use_scalar,
            "weight_decompose": self.weight_decompose,
            "dropout": self.dropout,
            "rank_dropout": self.rank_dropout,
            "module_dropout": self.module_dropout,
            "target_modules": self.target_modules,
            "full_matrix": self.full_matrix,
            "bypass_mode": self.bypass_mode,
            "rs_lora": self.rs_lora,
            "unbalanced_factorization": self.unbalanced_factorization,
        }


@dataclass
class TrainingConfig:
    """Configuration for LoRA training process.

    Training uses:
    - Device-aware mixed precision (bf16 on CUDA/XPU, fp16 on MPS, fp32 on CPU)
    - Discrete timesteps from turbo shift=3.0 schedule (8 steps)
    - Randomly samples one of 8 timesteps per training step:
      [1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3]

    Attributes:
        shift: Timestep shift factor (fixed at 3.0 for turbo model)
        num_inference_steps: Number of inference steps (fixed at 8 for turbo)
        learning_rate: Initial learning rate
        batch_size: Training batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_epochs: Maximum number of training epochs
        save_every_n_epochs: Save checkpoint every N epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for clipping
        mixed_precision: Preferred precision mode for logging/config tracking
        seed: Random seed for reproducibility
        output_dir: Directory to save checkpoints and logs
    """
    # Fixed for turbo model
    shift: float = 3.0  # Fixed: turbo uses shift=3.0
    num_inference_steps: int = 8  # Fixed: turbo uses 8 steps
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_epochs: int = 100
    save_every_n_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    use_fp8: bool = False
    gradient_checkpointing: bool = False
    seed: int = 42
    output_dir: str = "./lora_output"

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory_device: str = ""

    # Logging
    log_every_n_steps: int = 10

    # Validation (for loss curve and best-checkpoint tracking)
    val_split: float = 0.0

    # Crucible patch (2026-05-30): timestep sampling mode (METAL_LORA_PLAN §12).
    # "discrete"   = randomly pick from the 8-element turbo grid (current behavior,
    #                preserves all prior runs; the default so opt-in is required).
    # "continuous" = logit-normal sampling per the model's original sample_t_r() —
    #                matches what the model was trained against. Hypothesis (per §13a.3):
    #                this fixes the discrete-vs-continuous distribution mismatch that
    #                may contribute to guitar smear / val-loss plateau on small data.
    timestep_sampling_mode: str = "discrete"
    timestep_mu: float = -0.4
    """Mean for logit-normal sampling (used when mode=continuous). From the
    model's own training config (acestep/training_v2/configs.py:155)."""
    timestep_sigma: float = 1.0
    """Std for logit-normal sampling (used when mode=continuous). Same source."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError("val_split must be in [0.0, 1.0).")
        if self.timestep_sampling_mode not in ("discrete", "continuous"):
            raise ValueError(
                "timestep_sampling_mode must be 'discrete' or 'continuous'"
            )

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "shift": self.shift,
            "num_inference_steps": self.num_inference_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_epochs": self.max_epochs,
            "save_every_n_epochs": self.save_every_n_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "mixed_precision": self.mixed_precision,
            "use_fp8": self.use_fp8,
            "gradient_checkpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "pin_memory_device": self.pin_memory_device,
            "log_every_n_steps": self.log_every_n_steps,
            "val_split": self.val_split,
            "timestep_sampling_mode": self.timestep_sampling_mode,
            "timestep_mu": self.timestep_mu,
            "timestep_sigma": self.timestep_sigma,
        }
