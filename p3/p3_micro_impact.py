# p3_micro_impact.py
# P3 微碰撞模块：处理雨滴等微小质量体撞击织物的物理响应
# 溅落动力学、动态重力补偿、材质抗性

import math
from typing import Dict, Any, List, Tuple, Optional

class P3MicroImpact:
    """
    微碰撞处理器
    接收微小质量体的碰撞信息，产生溅落视觉效果和动态重力补偿
    """

    def __init__(self):
        self.splash_energy_threshold = 0.3
        self.absorption_weight_limit = 0.5
        self.micro_jitter_strength = 0.02
        self.absorption_decay_time = 2.0
        self.material_props = {
            "leather": {"bouncing_rate": 0.9, "diffusion_rate": 0.1, "energy_loss": 0.2},
            "vinyl":   {"bouncing_rate": 0.85, "diffusion_rate": 0.15, "energy_loss": 0.25},
            "silk":    {"bouncing_rate": 0.4,  "diffusion_rate": 0.7,  "energy_loss": 0.6},
            "cotton":  {"bouncing_rate": 0.2,  "diffusion_rate": 0.9,  "energy_loss": 0.8},
            "denim":   {"bouncing_rate": 0.5,  "diffusion_rate": 0.5,  "energy_loss": 0.5},
            "default": {"bouncing_rate": 0.5,  "diffusion_rate": 0.5,  "energy_loss": 0.5},
        }
        self.absorption_weight: Dict[str, float] = {}

    def _safe_float(self, val: Any, default=0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _clamp(self, val: float, lo=0.0, hi=1.0) -> float:
        if val != val:
            return lo
        return max(lo, min(hi, val))

    def _get_material_type(self, asset_data: Dict) -> str:
        slots = asset_data.get("slots", {})
        tex = slots.get("slot_5_texture", {})
        mat = tex.get("material_type", "default").lower()
        if mat not in self.material_props:
            mat = "default"
        return mat

    def _get_stiffness(self, asset_data: Dict) -> float:
        slots = asset_data.get("slots", {})
        layer = slots.get("slot_6_physics_layer", {})
        return self._safe_float(layer.get("stiffness", 0.5))

    def _compute_splash_energy(self, impact_velocity: float, stiffness: float, material: str, dt: float) -> float:
        props = self.material_props.get(material, self.material_props["default"])
        impulse = impact_velocity * dt
        base_energy = (impulse ** 2) * (1.0 - stiffness)
        energy = base_energy * (1.0 - props["energy_loss"])
        return self._clamp(energy, 0.0, 1.0)

    def process(self, p3_package: Dict[str, Any],
                impacts: Optional[List[Tuple[str, Tuple[float, float, float], float]]] = None,
                dt: float = 1.0 / 30.0) -> Dict[str, Any]:
        assets = p3_package.get("assets", {})
        if not assets:
            return p3_package

        # 衰减所有资产的吸收重量
        decay_factor = math.exp(-dt / self.absorption_decay_time)
        for aid in list(self.absorption_weight.keys()):
            if aid not in assets:
                del self.absorption_weight[aid]
            else:
                self.absorption_weight[aid] *= decay_factor
                if self.absorption_weight[aid] < 0.001:
                    del self.absorption_weight[aid]

        if impacts:
            for asset_id, velocity_vec, speed in impacts:
                if asset_id not in assets:
                    continue
                asset_data = assets[asset_id]
                sovereignty = asset_data.get("sovereignty", "")
                if sovereignty != "fabric_factory":
                    continue

                material = self._get_material_type(asset_data)
                stiffness = self._get_stiffness(asset_data)
                splash_energy = self._compute_splash_energy(speed, stiffness, material, dt)

                if splash_energy > self.splash_energy_threshold:
                    slots = asset_data.get("slots", {})
                    if "slot_5_texture" not in slots:
                        slots["slot_5_texture"] = {}
                    tex_slot = slots["slot_5_texture"]
                    highlight_intensity = self._clamp(splash_energy * 1.5, 0.0, 1.0)
                    tex_slot["micro_highlighter"] = {
                        "active": True,
                        "intensity": highlight_intensity,
                        "position": velocity_vec[:2] if len(velocity_vec) >= 2 else (0.5, 0.5),
                        "decay_rate": 0.8
                    }
                    props = self.material_props[material]
                    tex_slot["impact_response"] = {
                        "bouncing_rate": props["bouncing_rate"],
                        "diffusion_rate": props["diffusion_rate"],
                        "splash_energy": splash_energy
                    }
                    # 吸收重量累加
                    current_weight = self.absorption_weight.get(asset_id, 0.0)
                    incoming_energy = splash_energy * 0.1
                    self.absorption_weight[asset_id] = current_weight + incoming_energy * dt
                    asset_data["slots"] = slots

        # 动态重力补偿：根据吸收重量产生微颤动
        for asset_id, weight in self.absorption_weight.items():
            if weight > self.absorption_weight_limit and asset_id in assets:
                asset_data = assets[asset_id]
                slots = asset_data.get("slots", {})
                if "slot_6_physics_layer" not in slots:
                    slots["slot_6_physics_layer"] = {}
                layer_slot = slots["slot_6_physics_layer"]
                jitter = (weight - self.absorption_weight_limit) * self.micro_jitter_strength
                jitter = self._clamp(jitter, 0.0, 0.05)
                if "gravity_multiplier" in layer_slot:
                    layer_slot["gravity_multiplier"] += jitter
                else:
                    layer_slot["gravity_multiplier"] = 1.0 + jitter
                layer_slot["micro_jitter"] = jitter
                asset_data["slots"] = slots

        p3_package["assets"] = assets
        p3_package["absorption_weights"] = self.absorption_weight.copy()
        return p3_package


# 使用示例
if __name__ == "__main__":
    mock_p3 = {
        "assets": {
            "cotton_shirt": {
                "sovereignty": "fabric_factory",
                "slots": {
                    "slot_6_physics_layer": {"stiffness": 0.3},
                    "slot_5_texture": {"material_type": "cotton"}
                }
            }
        }
    }
    impacts = [("cotton_shirt", (0.1, 0.2, 0.5), 2.5)]
    micro = P3MicroImpact()
    result = micro.process(mock_p3, impacts, dt=1/30)
    print(result["assets"]["cotton_shirt"]["slots"]["slot_5_texture"]["micro_highlighter"]["intensity"])