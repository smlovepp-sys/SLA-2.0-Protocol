# p3_micro_detail_tear.py
# P3 微观细节与撕裂模块：各向异性断裂、纤维拉丝侵蚀、边缘应力卷曲
# 修复：修正纤维主轴混合方向计算，确保各向异性撕裂方向正确

import math
import random
from typing import Dict, Any, List, Tuple, Optional

class P3MicroDetailTear:
    """
    微观细节撕裂模块
    - 各向异性断裂：根据材质经纬线和受力矢量生成方向性撕裂掩码
    - 纤维拉丝与侵蚀：在边缘注入高频噪声，随时间加剧脱落
    - 边缘卷曲：通过法线偏移模拟布料断裂后的卷曲效果
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.material_props = {
            "silk": {
                "tensile_strength": 0.8,
                "anisotropy": 0.9,
                "fraying_intensity": 0.2,
                "curl_intensity": 0.1,
                "edge_roughness": 0.3,
                "fiber_orientation": (1.0, 0.0),
                "fiber_bond_strength": 0.7,
            },
            "denim": {
                "tensile_strength": 0.5,
                "anisotropy": 0.7,
                "fraying_intensity": 0.9,
                "curl_intensity": 0.8,
                "edge_roughness": 0.6,
                "fiber_orientation": (0.0, 1.0),
                "fiber_bond_strength": 0.5,
            },
            "leather": {
                "tensile_strength": 0.95,
                "anisotropy": 0.3,
                "fraying_intensity": 0.1,
                "curl_intensity": 0.4,
                "edge_roughness": 0.9,
                "fiber_orientation": (1.0, 0.0),
                "fiber_bond_strength": 0.2,
            },
            "cotton": {
                "tensile_strength": 0.4,
                "anisotropy": 0.5,
                "fraying_intensity": 0.7,
                "curl_intensity": 0.5,
                "edge_roughness": 0.5,
                "fiber_orientation": (1.0, 0.0),
                "fiber_bond_strength": 0.4,
            },
            "default": {
                "tensile_strength": 0.5,
                "anisotropy": 0.5,
                "fraying_intensity": 0.5,
                "curl_intensity": 0.5,
                "edge_roughness": 0.5,
                "fiber_orientation": (1.0, 0.0),
                "fiber_bond_strength": 0.5,
            }
        }
        self.erosion_buffer: Dict[str, float] = {}

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

    def _get_stress_magnitude(self, asset_data: Dict) -> float:
        slots = asset_data.get("slots", {})
        layer = slots.get("slot_6_physics_layer", {})
        stress = self._safe_float(layer.get("stress_magnitude", 0.0))
        return self._clamp(stress)

    def _get_stress_vector(self, asset_data: Dict) -> Tuple[float, float]:
        slots = asset_data.get("slots", {})
        tex = slots.get("slot_5_texture", {})
        stretch = tex.get("anisotropic_stretch", {})
        if "direction" in stretch:
            d = stretch["direction"]
            if len(d) >= 2:
                return (self._safe_float(d[0]), self._safe_float(d[1]))
        return (1.0, 0.0)

    def _generate_anisotropic_tear_mask(self, stress_magnitude: float, stress_dir: Tuple[float, float],
                                        material: str, width: int = 64, height: int = 64) -> Dict[str, Any]:
        props = self.material_props[material]
        if stress_magnitude <= props["tensile_strength"]:
            return {"active": False, "strength": 0.0}

        excess = (stress_magnitude - props["tensile_strength"]) / (1.0 - props["tensile_strength"])
        tear_intensity = self._clamp(excess, 0.0, 1.0)

        fiber_axis = props.get("fiber_orientation", (1.0, 0.0))
        fiber_bond = props.get("fiber_bond_strength", 0.5)

        # 混合方向
        mix_x = (1 - fiber_bond) * stress_dir[0] + fiber_bond * fiber_axis[0]
        mix_y = (1 - fiber_bond) * stress_dir[1] + fiber_bond * fiber_axis[1]
        mag = math.hypot(mix_x, mix_y)
        if mag > 1e-6:
            tearing_dir = (mix_x / mag, mix_y / mag)
        else:
            tearing_dir = (1.0, 0.0)

        # 微观噪声
        noise_angle = (random.random() - 0.5) * 0.2
        cos_noise = math.cos(noise_angle)
        sin_noise = math.sin(noise_angle)
        noisy_x = tearing_dir[0] * cos_noise - tearing_dir[1] * sin_noise
        noisy_y = tearing_dir[0] * sin_noise + tearing_dir[1] * cos_noise
        mag = math.hypot(noisy_x, noisy_y)
        if mag > 1e-6:
            tearing_dir = (noisy_x / mag, noisy_y / mag)

        anisotropy = props["anisotropy"]
        length = 0.2 + tear_intensity * 0.6
        width_ratio = 0.1 + (1.0 - anisotropy) * 0.3

        return {
            "active": True,
            "strength": tear_intensity,
            "direction": tearing_dir,
            "length": length,
            "width": length * width_ratio,
            "edge_roughness": props["edge_roughness"],
        }

    def _add_fiber_fraying(self, tear_info: Dict, material: str, erosion_level: float) -> Dict:
        props = self.material_props[material]
        fray_base = props["fraying_intensity"] * tear_info["strength"]
        fray_intensity = self._clamp(fray_base + erosion_level * 0.5, 0.0, 1.0)
        noise_scale = 0.5 + fray_intensity * 2.0
        return {
            "fraying_intensity": fray_intensity,
            "noise_scale": noise_scale,
            "erosion_decay": erosion_level,
        }

    def _compute_edge_curl(self, tear_info: Dict, material: str, stress_magnitude: float) -> Dict:
        props = self.material_props[material]
        curl_base = props["curl_intensity"] * tear_info["strength"]
        dir_x, dir_y = tear_info["direction"]
        perp = (-dir_y, dir_x)
        curl_offset = curl_base * 0.05
        return {
            "active": True,
            "offset": curl_offset,
            "direction": perp,
            "curl_intensity": curl_base,
        }

    def process_frame(self, p3_package: Dict[str, Any], dt: float = 1.0/30.0) -> Dict[str, Any]:
        assets = p3_package.get("assets", {})
        if not assets:
            return p3_package

        decay_factor = math.exp(-dt / 2.0)
        for aid in list(self.erosion_buffer.keys()):
            if aid not in assets:
                del self.erosion_buffer[aid]
            else:
                self.erosion_buffer[aid] *= decay_factor
                if self.erosion_buffer[aid] < 0.01:
                    del self.erosion_buffer[aid]

        for asset_id, asset_data in assets.items():
            sovereignty = asset_data.get("sovereignty", "")
            if sovereignty != "fabric_factory":
                continue

            material = self._get_material_type(asset_data)
            stress_mag = self._get_stress_magnitude(asset_data)
            stress_dir = self._get_stress_vector(asset_data)

            tear_info = self._generate_anisotropic_tear_mask(stress_mag, stress_dir, material)
            slots = asset_data.get("slots", {})

            if tear_info["active"]:
                erosion_delta = tear_info["strength"] * dt * 0.5
                current_erosion = self.erosion_buffer.get(asset_id, 0.0)
                new_erosion = self._clamp(current_erosion + erosion_delta, 0.0, 1.0)
                self.erosion_buffer[asset_id] = new_erosion

                fray = self._add_fiber_fraying(tear_info, material, new_erosion)
                curl = self._compute_edge_curl(tear_info, material, stress_mag)

                if "slot_5_texture" not in slots:
                    slots["slot_5_texture"] = {}
                tex_slot = slots["slot_5_texture"]
                tex_slot["tear_mask"] = {
                    "active": True,
                    "strength": tear_info["strength"],
                    "direction": tear_info["direction"],
                    "length": tear_info["length"],
                    "width": tear_info["width"],
                    "edge_roughness": tear_info["edge_roughness"],
                }
                tex_slot["fiber_fraying"] = fray

                if "slot_6_physics_layer" not in slots:
                    slots["slot_6_physics_layer"] = {}
                layer_slot = slots["slot_6_physics_layer"]
                if curl["active"]:
                    layer_slot["edge_curl"] = {
                        "offset": curl["offset"],
                        "direction": curl["direction"],
                        "curl_intensity": curl["curl_intensity"]
                    }
                else:
                    layer_slot.pop("edge_curl", None)
            else:
                # 可选的清理
                pass

            asset_data["slots"] = slots

        p3_package["assets"] = assets
        p3_package["erosion_buffer"] = self.erosion_buffer.copy()
        return p3_package


# 使用示例
if __name__ == "__main__":
    mock_p3 = {
        "assets": {
            "denim_jeans": {
                "sovereignty": "fabric_factory",
                "slots": {
                    "slot_6_physics_layer": {"stress_magnitude": 0.9},
                    "slot_5_texture": {"material_type": "denim", "anisotropic_stretch": {"direction": [1.0, 0.2]}}
                }
            }
        }
    }
    tear_module = P3MicroDetailTear()
    result = tear_module.process_frame(mock_p3, dt=1/30)
    print(result["assets"]["denim_jeans"]["slots"]["slot_5_texture"]["tear_mask"]["strength"])