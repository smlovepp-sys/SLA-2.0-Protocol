# p3_kinetic_coupling.py
# 纯净版 | 无固定种子 | 无随机锁定 | 完全由 P3 主控随机
from typing import Dict, Any, Tuple
import math

class P3KineticCoupling:
    def __init__(self):
        # 物理参数（无种子、无随机）
        self.stiffness_boost = 3.0
        self.max_resistance_viscosity = 0.8
        self.grip_offset_max = -0.02
        self.grip_decay_rate = 0.05
        self.stack_thickness_factor = 0.1

        # 状态（无随机）
        self.grip_duration: Dict[str, float] = {}
        self.deformation_memory: Dict[str, Dict[str, float]] = {}

    # 材质预设
    FABRIC_MATERIALS = {
        "silk": {"stiffness_base": 0.2, "viscosity_base": 0.1, "rebound": 0.8, "memory_factor": 0.1},
        "denim": {"stiffness_base": 0.7, "viscosity_base": 0.7, "rebound": 0.2, "memory_factor": 0.4},
        "wool": {"stiffness_base": 0.4, "viscosity_base": 0.4, "rebound": 0.5, "memory_factor": 0.2},
        "default": {"stiffness_base": 0.5, "viscosity_base": 0.5, "rebound": 0.5, "memory_factor": 0.2}
    }

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            val = float(value)
            return val if not math.isnan(val) else default
        except (TypeError, ValueError):
            return default

    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        return max(min_val, min(max_val, value))

    def _get_fabric_material(self, asset_data: Dict[str, Any]):
        material = asset_data.get("material", "default").lower()
        return self.FABRIC_MATERIALS.get(material, self.FABRIC_MATERIALS["default"])

    def _is_fabric_asset(self, asset_data: Dict[str, Any]) -> bool:
        return asset_data.get("sovereignty") == "fabric_factory"

    def _calculate_dynamic_resistance(self, asset_data: Dict[str, Any]) -> float:
        stack_thickness = self._safe_float(asset_data.get("stack_thickness", 1.0), 1.0)
        base_viscosity = self._get_fabric_material(asset_data)["viscosity_base"]
        dynamic_visc = base_viscosity * (1 + self.stack_thickness_factor * math.log(stack_thickness + 1))
        return self._clamp(dynamic_visc, 0.0, self.max_resistance_viscosity)

    def _update_grip_duration(self, fabric_id: str, delta_time: float = 0.1) -> float:
        self.grip_duration[fabric_id] = self.grip_duration.get(fabric_id, 0.0) + delta_time
        decay_coeff = math.exp(-self.grip_decay_rate * self.grip_duration[fabric_id])
        return self._clamp(decay_coeff, 0.1, 1.0)

    def _apply_deformation_memory(self, fabric_id: str, param_name: str, current_val: float) -> float:
        if fabric_id not in self.deformation_memory:
            self.deformation_memory[fabric_id] = {}
        
        memory_val = self.deformation_memory[fabric_id].get(param_name, current_val)
        material = self.FABRIC_MATERIALS.get(
            self.deformation_memory[fabric_id].get("material", "default"),
            self.FABRIC_MATERIALS["default"]
        )
        
        new_memory = memory_val * (1 - material["memory_factor"]) + current_val * material["memory_factor"]
        self.deformation_memory[fabric_id][param_name] = new_memory
        return new_memory

    def reset_grip_state(self, fabric_id=None):
        if fabric_id:
            self.grip_duration.pop(fabric_id, None)
        else:
            self.grip_duration.clear()

    def process(self, p3_package: Dict[str, Any],
                hand_grab_info: Dict[str, Tuple[str, Tuple[float, float, float]]],
                delta_time: float = 0.1):

        assets = p3_package.get("assets", {})
        if not assets or not hand_grab_info:
            return p3_package

        for fabric_id, (joint_id, grab_point) in hand_grab_info.items():
            if fabric_id not in assets:
                continue
            asset_data = assets[fabric_id]
            if not self._is_fabric_asset(asset_data):
                continue

            slots = asset_data.get("slots", {})
            joint_data = assets.get(joint_id, {})
            motion_vec = [self._safe_float(v) for v in joint_data.get("motion_vector", [0.0,0.0,0.0])]
            motion_vec = motion_vec[:3]
            motion_mag = math.hypot(*motion_vec)
            material = self._get_fabric_material(asset_data)

            if fabric_id in self.deformation_memory:
                self.deformation_memory[fabric_id]["material"] = asset_data.get("material", "default")
            else:
                self.deformation_memory[fabric_id] = {"material": asset_data.get("material", "default")}

            decay_coeff = self._update_grip_duration(fabric_id, delta_time)

            # 物理层
            if "slot_6_physics_layer" not in slots:
                slots["slot_6_physics_layer"] = {}
            layer_slot = slots["slot_6_physics_layer"]
            
            original_stiff = self._safe_float(layer_slot.get("stiffness", material["stiffness_base"]))
            new_stiff = original_stiff * self.stiffness_boost * decay_coeff
            new_stiff = new_stiff - (motion_mag * material["rebound"])
            layer_slot["stiffness"] = self._clamp(new_stiff, 0.0, 1.0)

            layer_slot["tension_center"] = {
                "active": True,
                "position": grab_point,
                "stiffness_multiplier": self.stiffness_boost * decay_coeff
            }

            # 拉伸
            if motion_mag > 0.01:
                if "slot_5_texture" not in slots:
                    slots["slot_5_texture"] = {}
                tex_slot = slots["slot_5_texture"]
                dx, dy, dz = [v/motion_mag for v in motion_vec] if motion_mag !=0 else (0,0,0)
                stretch_intensity = self._clamp(motion_mag * 2 * decay_coeff, 0,1)
                tex_slot["anisotropic_stretch"] = {
                    "direction": [dx, dy, dz],
                    "intensity": stretch_intensity
                }
                current_ns = self._safe_float(tex_slot.get("normal_strength", 0.5))
                new_ns = current_ns + motion_mag * 0.5 * decay_coeff
                new_ns = self._apply_deformation_memory(fabric_id, "normal_strength", new_ns)
                tex_slot["normal_strength"] = self._clamp(new_ns, 0.0, 1.5)

            # 阻力
            if asset_data.get("stacking_flag", False):
                dynamic_visc = self._calculate_dynamic_resistance(asset_data)
                layer_slot["viscosity"] = self._clamp(
                    layer_slot.get("viscosity", 0.0) + dynamic_visc * decay_coeff,
                    0.0, self.max_resistance_viscosity
                )
                layer_slot["resistance_feedback"] = True

            # 抓握偏移
            layer_slot["collision_offset"] = self.grip_offset_max * decay_coeff
            asset_data["slots"] = slots
            assets[fabric_id] = asset_data

        # 松开后的回弹
        all_fab = [k for k,v in assets.items() if self._is_fabric_asset(v)]
        grabbed = list(hand_grab_info.keys())
        released = [f for f in all_fab if f not in grabbed]
        
        for fabric_id in released:
            if fabric_id in self.deformation_memory:
                asset_data = assets[fabric_id]
                slots = asset_data.get("slots", {})
                if "slot_6_physics_layer" in slots:
                    layer_slot = slots["slot_6_physics_layer"]
                    current_stiff = self._safe_float(layer_slot.get("stiffness", 0.5))
                    material = self._get_fabric_material(asset_data)
                    layer_slot["stiffness"] = self._clamp(
                        current_stiff * (1 - material["rebound"] * delta_time),
                        material["stiffness_base"], 1.0
                    )
                asset_data["slots"] = slots
                assets[fabric_id] = asset_data

        p3_package["assets"] = assets
        return p3_package