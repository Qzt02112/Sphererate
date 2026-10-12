from __future__ import annotations

from typing import Any

from torch import nn

from sphererate.models.dytsal import (
    CurrentTransitionNextFixationNet,
    DyTSalDMM360,
    DirectStartConditionedSaliencyNet,
    FiLMStartConditionedResidualNet,
    ImageOnlyLatentMarginalResidualNet,
    LightweightStartConditionedResidualNet,
    LearnedStartOnlySaliencyNet,
    ParallelTemporalNet,
    ResidualMixerDyTSalDMM360,
    SemanticDyTSalDMM360,
    SemanticStaticSaliencyNet,
    SemanticStaticAuxSaliencyNet,
    SemanticTeacherResidualDyTSalDMM360,
    SemanticTempSALFusionNet,
    SemanticTempSALTemporalNet,
    StartConditionedTeacherResidualDyTSalDMM360,
    StaticSaliencyNet,
    TempSALStyleMixerNet,
)


def build_model(name: str, config: dict[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    name = name.lower()
    kwargs = {
        "in_channels": int(model_cfg.get("in_channels", 8)),
        "base_channels": int(model_cfg.get("base_channels", 32)),
        "hidden_channels": int(model_cfg.get("hidden_channels", 64)),
        "num_bins": int(model_cfg.get("num_bins", config.get("data", {}).get("num_bins", 6))),
    }
    semantic_kwargs = {
        "semantic_backbone": str(model_cfg.get("semantic_backbone", "resnet18")),
        "semantic_layer": str(model_cfg.get("semantic_layer", "layer3")),
        "semantic_pretrained": bool(model_cfg.get("semantic_pretrained", True)),
    }
    if name == "static":
        return StaticSaliencyNet(**kwargs)
    if name in {"current_transition", "nextfix_transition", "transition_nextfix", "current_transition_renderer", "motion_transition_renderer"}:
        return CurrentTransitionNextFixationNet(
            **kwargs,
            prefix_length=int(model_cfg.get("prefix_length", config.get("train", {}).get("scanpath_prefix_length", 3))),
            motion_bins_y=int(model_cfg.get("motion_bins_y", 9)),
            motion_bins_x=int(model_cfg.get("motion_bins_x", 9)),
            use_motion_renderer=bool(
                model_cfg.get("use_motion_renderer", name in {"current_transition_renderer", "motion_transition_renderer"})
            ),
            motion_render_scale=float(model_cfg.get("motion_render_scale", 1.0)),
            motion_render_learnable_scale=bool(model_cfg.get("motion_render_learnable_scale", False)),
            motion_render_temperature=float(model_cfg.get("motion_render_temperature", 1.0)),
            use_start_prior=bool(model_cfg.get("use_start_prior", False)),
            start_prior_init_scale=float(model_cfg.get("start_prior_init_scale", 0.1)),
            start_prior_temperature=float(model_cfg.get("start_prior_temperature", 1.0)),
            start_prior_fusion=str(model_cfg.get("start_prior_fusion", "residual")),
            start_prior_hybrid_residual_scale=float(model_cfg.get("start_prior_hybrid_residual_scale", 0.0)),
        )
    if name in {"semantic_static", "sem_static", "static_semantic"}:
        return SemanticStaticSaliencyNet(**kwargs, **semantic_kwargs)
    if name in {"semantic_static_aux", "static_temporal_aux", "p2_temporal_aux"}:
        return SemanticStaticAuxSaliencyNet(**kwargs, **semantic_kwargs)
    if name in {"parallel", "directsum", "direct_sum"}:
        return ParallelTemporalNet(**kwargs, learned_accumulation=bool(model_cfg.get("learned_accumulation", True)))
    if name in {"semantic_dytsaldmm", "sem_dytsaldmm", "dytsaldmm_semantic", "semantic_temporal"}:
        return SemanticDyTSalDMM360(
            **kwargs,
            learned_accumulation=bool(model_cfg.get("learned_accumulation", True)),
            **semantic_kwargs,
        )
    if name in {
        "semantic_teacher_residual",
        "sem_teacher_residual",
        "teacher_residual",
        "teacher_residual_dytsaldmm",
    }:
        return SemanticTeacherResidualDyTSalDMM360(
            **kwargs,
            learned_accumulation=bool(model_cfg.get("learned_accumulation", True)),
            teacher_checkpoint=model_cfg.get("teacher_checkpoint"),
            residual_scale=float(model_cfg.get("teacher_residual_scale", 1.0)),
            residual_init_gate=float(model_cfg.get("teacher_residual_init_gate", -1.5)),
            **semantic_kwargs,
        )
    if name in {
        "start_conditioned_teacher_residual",
        "start_teacher_residual",
        "jufe_start_teacher_residual",
        "conditioned_teacher_residual",
    }:
        return StartConditionedTeacherResidualDyTSalDMM360(
            **kwargs,
            learned_accumulation=bool(model_cfg.get("learned_accumulation", True)),
            teacher_checkpoint=model_cfg.get("teacher_checkpoint"),
            residual_scale=float(model_cfg.get("teacher_residual_scale", 1.0)),
            residual_init_gate=float(model_cfg.get("teacher_residual_init_gate", -1.5)),
            static_in_channels=int(model_cfg.get("static_in_channels", 8)),
            **semantic_kwargs,
        )
    if name in {
        "start_conditioned_light_residual",
        "lightweight_start_conditioned_residual",
        "start_light_residual",
    }:
        return LightweightStartConditionedResidualNet(
            **kwargs,
            teacher_checkpoint=model_cfg.get("teacher_checkpoint"),
            residual_scale=float(model_cfg.get("teacher_residual_scale", 1.0)),
            residual_init_gate=float(model_cfg.get("teacher_residual_init_gate", -1.0)),
            static_in_channels=int(model_cfg.get("static_in_channels", 8)),
            residual_hidden_channels=int(model_cfg.get("residual_hidden_channels", 16)),
            **semantic_kwargs,
        )
    if name in {"learned_start_only", "start_only", "condition_only"}:
        return LearnedStartOnlySaliencyNet(
            **kwargs,
            static_in_channels=int(model_cfg.get("static_in_channels", 8)),
            residual_hidden_channels=int(model_cfg.get("residual_hidden_channels", 16)),
        )
    if name in {"direct_start_conditioned", "start_direct", "direct_condition"}:
        return DirectStartConditionedSaliencyNet(
            **kwargs,
            teacher_checkpoint=model_cfg.get("teacher_checkpoint"),
            static_in_channels=int(model_cfg.get("static_in_channels", 8)),
            residual_hidden_channels=int(model_cfg.get("residual_hidden_channels", 16)),
            **semantic_kwargs,
        )
    if name in {"film_start_conditioned", "start_film", "film_condition"}:
        return FiLMStartConditionedResidualNet(
            **kwargs,
            teacher_checkpoint=model_cfg.get("teacher_checkpoint"),
            residual_scale=float(model_cfg.get("teacher_residual_scale", 1.0)),
            residual_init_gate=float(model_cfg.get("teacher_residual_init_gate", -1.0)),
            static_in_channels=int(model_cfg.get("static_in_channels", 8)),
            residual_hidden_channels=int(model_cfg.get("residual_hidden_channels", 16)),
            **semantic_kwargs,
        )
    if name in {
        "image_only_latent_marginal",
        "latent_condition_marginal",
        "start_marginalized_residual",
        "image_only_temporal_aux",
        "latent_condition_logit_pool",
    }:
        default_aggregation = "logit_pool" if name in {"image_only_temporal_aux", "latent_condition_logit_pool"} else "density_marginal"
        return ImageOnlyLatentMarginalResidualNet(
            **kwargs,
            teacher_checkpoint=model_cfg.get("teacher_checkpoint"),
            residual_scale=float(model_cfg.get("teacher_residual_scale", 1.0)),
            residual_init_gate=float(model_cfg.get("teacher_residual_init_gate", -1.0)),
            residual_hidden_channels=int(model_cfg.get("residual_hidden_channels", 16)),
            latent_num_conditions=int(model_cfg.get("latent_num_conditions", 3)),
            latent_code_channels=int(model_cfg.get("latent_code_channels", 4)),
            conditioning=str(model_cfg.get("latent_conditioning", "stage_code")),
            aggregation=str(model_cfg.get("latent_aggregation", default_aggregation)),
            **semantic_kwargs,
        )
    if name in {"tempsal_mixer", "tempsal-style", "tempsal_style", "temporal_mixer"}:
        return TempSALStyleMixerNet(**kwargs, learned_accumulation=bool(model_cfg.get("learned_accumulation", True)))
    if name in {"semantic_tempsal_temporal", "sem_tempsal_temporal", "tempsal_temporal"}:
        return SemanticTempSALTemporalNet(
            **kwargs,
            learned_accumulation=bool(model_cfg.get("learned_accumulation", True)),
            **semantic_kwargs,
        )
    if name in {"semantic_tempsal_fusion", "sem_tempsal_fusion", "tempsal_fusion"}:
        return SemanticTempSALFusionNet(
            **kwargs,
            learned_accumulation=bool(model_cfg.get("learned_accumulation", True)),
            temporal_checkpoint=model_cfg.get("temporal_checkpoint"),
            static_checkpoint=model_cfg.get("static_checkpoint"),
            freeze_temporal=bool(model_cfg.get("freeze_temporal", True)),
            freeze_static=bool(model_cfg.get("freeze_static", True)),
            shuffle_temporal_inputs=bool(model_cfg.get("shuffle_temporal_inputs", False)),
            **semantic_kwargs,
        )
    if name in {"residual_mixer", "dytsaldmm_residual", "dytsaldmm_residual_mixer"}:
        return ResidualMixerDyTSalDMM360(
            **kwargs,
            learned_accumulation=bool(model_cfg.get("learned_accumulation", True)),
            mixer_use_temporal=bool(model_cfg.get("mixer_use_temporal", True)),
        )
    if name in {"dytsaldmm", "dytsaldmm360", "ours"}:
        return DyTSalDMM360(**kwargs, learned_accumulation=bool(model_cfg.get("learned_accumulation", True)))
    raise ValueError(f"Unknown model: {name}")
