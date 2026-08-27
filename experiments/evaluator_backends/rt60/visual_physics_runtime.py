from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


RT60_500_PROMPT = (
    "You are an expert in architectural acoustics. The first image is RGB, "
    "the second is Depth, and the third is an Acoustic Alpha Map showing "
    "500 Hz absorption distribution. Analyze the room geometry and absorption "
    "distribution to estimate the reverberation time at 500 Hz."
)


class PhysicsRT60Head(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-4,
        rt60_min: float = 0.05,
        rt60_max: float = 5.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.eps = float(eps)
        self.rt60_min = float(rt60_min)
        self.rt60_max = float(rt60_max)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.geometry_head = nn.Linear(hidden_size, 1)
        self.absorption_head = nn.Linear(hidden_size, 1)
        self.residual_head = nn.Linear(hidden_size, 1)
        self.class_head = nn.Linear(hidden_size, 6)
        self.extreme_long_head = nn.Linear(hidden_size, 1)

    def forward(self, scene_representation: torch.Tensor) -> Dict[str, torch.Tensor]:
        scene_representation = scene_representation.to(self.norm.weight.dtype)
        features = self.dropout(self.norm(scene_representation))
        volume_proxy = F.softplus(self.geometry_head(features).squeeze(-1)) + self.eps
        absorption_proxy = F.softplus(self.absorption_head(features).squeeze(-1)) + self.eps
        physical_prior = (0.161 * volume_proxy / absorption_proxy).clamp(
            min=self.rt60_min, max=self.rt60_max
        )
        residual = self.residual_head(features).squeeze(-1)
        log_prediction = torch.log(physical_prior) + residual
        prediction = torch.exp(log_prediction).clamp(min=self.rt60_min, max=self.rt60_max)
        class_logits = self.class_head(features)
        long_logit = self.extreme_long_head(features).squeeze(-1)
        long_probability = torch.sigmoid(long_logit)
        return {
            "V_hat": volume_proxy,
            "A_hat_500": absorption_proxy,
            "rt60_500Hz_phys_prior": physical_prior,
            "residual_500Hz": residual,
            "log_rt60_500Hz_pred": log_prediction,
            "rt60_500Hz_pred": prediction,
            "rt60_500Hz_class_logits": class_logits,
            "rt60_500Hz_class_pred": class_logits.argmax(dim=-1),
            "is_extreme_long_logit": long_logit,
            "is_extreme_long_prob": long_probability,
            "is_extreme_long_pred": long_probability.ge(0.5),
        }


class PhysicsRT60Model(nn.Module):
    """Attach a continuous RT60 head to a multimodal VLM scene representation."""

    def __init__(
        self,
        vlm: nn.Module,
        hidden_size: int,
        head_config: Dict,
    ):
        super().__init__()
        self.vlm = vlm
        self.physics_head = PhysicsRT60Head(hidden_size=hidden_size, **head_config)

    @staticmethod
    def pool_hidden_state(
        hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_state.mean(dim=1)
        mask = attention_mask.to(hidden_state.dtype).unsqueeze(-1)
        return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def encode_scene(self, **model_inputs) -> torch.Tensor:
        forward_kwargs = dict(
            model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        backbone_owner = (
            self.vlm.get_base_model()
            if hasattr(self.vlm, "get_base_model")
            else self.vlm
        )
        backbone = getattr(backbone_owner, "model", backbone_owner)
        outputs = backbone(**forward_kwargs)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs.hidden_states[-1]
        return self.pool_hidden_state(hidden_state, model_inputs.get("attention_mask"))

    def forward(self, **model_inputs) -> Dict[str, torch.Tensor]:
        scene_representation = self.encode_scene(**model_inputs)
        outputs = self.physics_head(scene_representation)
        outputs["scene_representation"] = scene_representation
        return outputs


def build_vlm_and_processor(config: Dict[str, Any]):
    """Construct the frozen Qwen3-VL backbone and released physics head."""

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_config = config["model"]
    dtype = torch.bfloat16 if model_config.get("torch_dtype") == "bfloat16" else None
    processor = AutoProcessor.from_pretrained(model_config["model_name_or_path"])
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_config["model_name_or_path"],
        dtype=dtype,
        attn_implementation=model_config.get("attn_implementation"),
    )
    if model_config.get("gradient_checkpointing"):
        vlm.gradient_checkpointing_enable()
    if model_config.get("freeze_vlm", True):
        for parameter in vlm.parameters():
            parameter.requires_grad = False

    adapter_path = model_config.get("init_lora_path")
    if not adapter_path:
        raise ValueError("RT60 inference requires the released LoRA adapter")
    from peft import PeftModel

    vlm = PeftModel.from_pretrained(vlm, adapter_path, is_trainable=False)
    model = PhysicsRT60Model(
        vlm=vlm,
        hidden_size=int(model_config["hidden_size"]),
        head_config=config["head"],
    )
    return model, processor
