# p2_comfy_nodes.py
# P2 ComfyUI 节点层（返回值对齐修正版）

import torch
from typing import Dict, Any
from .p2_evolution_orchestrator import P2EvolutionOrchestrator

class P2OrchestratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "p1_package": ("P1_PACKAGE",),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 50}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 60.0}),
                "sampler_type": (["ddim", "euler", "dpmpp"],),
                "seed": ("INT", {"default": 12345678, "min": 10000000, "max": 99999999}),
            }
        }

    RETURN_TYPES = ("MODEL", "P2_PACKAGE",)
    RETURN_NAMES = ("model", "p2_package",)
    FUNCTION = "process"
    CATEGORY = "P2/Orchestrator"

    def process(self, model, p1_package: Dict[str, Any], cfg_scale, steps, fps, sampler_type, seed):
        # ===================== DEBUG 维度检测 =====================
        print("\n【P2 维度检测】")
        try:
            if "start_frame" in p1_package:
                print(f"Start帧: {p1_package['start_frame'].shape}")
            if "end_frame" in p1_package:
                print(f"End帧: {p1_package['end_frame'].shape}")
        except Exception as e:
            print(f"[P2 Debug] 打印维度失败: {e}")

        # 1. 实例化总调度器并运行管线
        orchestrator = P2EvolutionOrchestrator(
            global_cfg_scale=cfg_scale,
            seed=seed,
            sampler_type=sampler_type,
            num_inference_steps=steps,
            debug=True
        )
        
        # 2. 得到带有标准物理资产和对齐参数的深度数据包
        cleaned_package = orchestrator.run_p2_pipeline(p1_package, fps)

        # ===================== 🔌 核心修复：透传清洗并解包后的正确包 =====================
        return (model, cleaned_package,)