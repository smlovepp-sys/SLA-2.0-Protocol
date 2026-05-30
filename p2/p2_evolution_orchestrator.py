# p2_evolution_orchestrator.py
from typing import Dict, Any, Optional
import copy
from .p2_vfx_adapter import P2VFXAdapter

class P2EvolutionOrchestrator:
    """
    P2 转发层：特征清洗 + 资产骨架标准化 + 扩散参数盖章（完美对齐 P3 应力诊断版）
    """
    def __init__(self,
                 global_cfg_scale: float = 7.5,
                 seed: int = 42,
                 sampler_type: str = "ddim",
                 num_inference_steps: int = 50,
                 debug: bool = True,
                 normalize_normals: bool = False,
                 device: Optional[str] = None):
        self.vfx_adapter = P2VFXAdapter(debug=debug, normalize_normals=normalize_normals)
        self.debug = debug
        self.device = device
        self.global_cfg_scale = global_cfg_scale
        self.seed = seed
        self.sampler_type = sampler_type
        self.num_inference_steps = num_inference_steps

    def run_p2_pipeline(self,
                        p1_package: Optional[Dict[str, Any]] = None,
                        fps: float = 30.0) -> Dict[str, Any]:
        if p1_package is None:
            return {}

        # 1. 智能安全清理并构建实体资产（由 VFXAdapter 内部完成）
        cleaned_package = self.vfx_adapter.adapt(p1_package)

        # 2. 确保 assets 存在（VFXAdapter 应该已经构建了实体资产，这里仅做兜底）
        if "assets" not in cleaned_package or not cleaned_package["assets"]:
            cleaned_package["assets"] = {}

        # 3. 🛡️ 兼容老格式兜底（完美对齐 P3 核心物理键名契约）
        manifest = cleaned_package.get("manifest", [])
        if isinstance(manifest, list) and len(manifest) > 0 and len(cleaned_package["assets"]) == 0:
            assets_dict = cleaned_package["assets"]
            for item in manifest:
                if isinstance(item, dict) and "component_id" in item:
                    cid = item["component_id"]
                    assets_dict[cid] = {
                        "sovereignty": item.get("factory", "unknown_factory"),
                        "slots": {
                            "slot_1_status": [1, 1, 0.5],
                            "slot_6_physics_layer": {
                                "stiffness": 0.5,
                                "friction": 0.5,          # ⚡ 修复：由 friction_coeff 变更为 friction，与 P3 契约完美对齐
                                "elasticity": 0.5,        # ⚡ 补全：对齐 P3 核心时域演化参数
                                "wetness": 0.0,           # ⚡ 补全：对齐 P3 材质流体参数
                                "tear_intensity": 0.0,    # ⚡ 补全：对齐 P3 微观撕裂参数
                                "tension": 0.0,
                                "body_type": 0.5
                            },
                        }
                    }
            if self.debug:
                print(f"[P2Orchestrator] 从 manifest 构建了 {len(assets_dict)} 个实体（旧格式标准化对齐完成）")

        # 4. 添加扩散参数字段
        cleaned_package["p2_diffusion_params"] = {
            "cfg_scale": self.global_cfg_scale,
            "seed": self.seed,
            "sampler_type": self.sampler_type,
            "num_inference_steps": self.num_inference_steps,
            "fps": fps,
        }

        # ⚡ 5. 双向解耦外层兼容补全（确保 P3 各种获取姿势都能命中）
        cleaned_package["seed"] = self.seed
        cleaned_package["cfg"] = self.global_cfg_scale
        cleaned_package["steps"] = self.num_inference_steps
        cleaned_package["sampler_type"] = self.sampler_type
        cleaned_package["sampler_name"] = self.sampler_type  # 满足 P3 资产包组装兜底块的 get("sampler_name")

        return cleaned_package


P2Orchestrator = P2EvolutionOrchestrator