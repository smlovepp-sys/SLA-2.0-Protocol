# p3_dynamic_feedback.py
# P3 动态反馈模块：微观扰动、粘滞摩擦、潜空间噪声
# 独立于主物理引擎，为物理槽位增加丰富性

import random
import math
from typing import Dict, Any

class P3DynamicFeedback:
    """
    动态反馈增强器
    输入：经过 p3_physics_engine 处理后的 JSON 包
    输出：添加了微观扰动、粘滞摩擦、潜空间噪声的数据包
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        if value != value:
            return min_val
        return max(min_val, min(max_val, value))

    def _nonlinear_turbulence(self, evolution: float) -> float:
        # 演化值 0->0, 1->1.2，中间呈指数增长
        return math.pow(evolution, 1.5) * 1.2

    def _add_random_offset(self, value: float, percent: float = 0.02) -> float:
        if value == 0.0:
            return 0.0
        offset = (random.random() * 2 - 1) * percent
        new_val = value * (1.0 + offset)
        # 限制绝对偏移不超过 percent
        if abs(new_val - value) > percent:
            new_val = value + (percent if offset > 0 else -percent)
        return new_val

    def _get_evolution(self, asset_data: Dict[str, Any]) -> float:
        slots = asset_data.get("slots", {})
        status = slots.get("slot_1_status", [1, 1, 0.0])
        evo = self._safe_float(status[2] if len(status) > 2 else 0.0)
        return self._clamp(evo, 0.0, 1.0)

    def process(self, p3_package: Dict[str, Any]) -> Dict[str, Any]:
        assets = p3_package.get("assets", {})
        if not assets:
            return p3_package

        # 为保证每帧随机偏移一致，重置种子
        random.seed(self.seed)

        for asset_id, asset_data in assets.items():
            sovereignty = asset_data.get("sovereignty", "unknown")
            # 主权金科玉律：严禁干涉 joint_factory
            if sovereignty == "joint_factory":
                continue

            slots = asset_data.get("slots", {})
            evolution = self._get_evolution(asset_data)

            # 1. 边缘紊乱：tear_factory 添加 turbulence_factor
            if sovereignty == "tear_factory":
                if "slot_6_edge_physics" not in slots:
                    slots["slot_6_edge_physics"] = {}
                edge_slot = slots["slot_6_edge_physics"]
                turbulence = self._nonlinear_turbulence(evolution)
                turbulence = self._clamp(turbulence, 0.0, 1.2)
                edge_slot["turbulence_factor"] = turbulence

            # 2. 粘滞对冲：fabric_factory 内层且演化>0.6 时增加摩擦系数
            if sovereignty == "fabric_factory":
                layer_slot = slots.get("slot_6_physics_layer", {})
                depth_rank = layer_slot.get("depth_rank", 2)
                if depth_rank == 1 and evolution > 0.6:
                    if "slot_6_physics_layer" not in slots:
                        slots["slot_6_physics_layer"] = {}
                    phys_slot = slots["slot_6_physics_layer"]
                    friction_increase = min(0.5, (evolution - 0.6) / 0.4 * 0.5)
                    base_friction = self._safe_float(phys_slot.get("friction_coeff", 0.3))
                    new_friction = self._clamp(base_friction + friction_increase, 0.0, 1.0)
                    phys_slot["friction_coeff"] = new_friction

            # 3. 潜空间噪声：对所有 slot_6_* 槽位中的数值属性添加 ±2% 随机偏移
            for slot_name, slot_dict in slots.items():
                if not slot_name.startswith("slot_6_"):
                    continue
                for key, val in list(slot_dict.items()):
                    if isinstance(val, (int, float)):
                        if key in ["instance_id", "factory", "material_type"]:
                            continue
                        new_val = self._add_random_offset(val, percent=0.02)
                        if key != "gravity_multiplier":
                            new_val = self._clamp(new_val, 0.0, 1.0)
                        slot_dict[key] = new_val

            asset_data["slots"] = slots

        p3_package["assets"] = assets
        return p3_package


# 使用示例
if __name__ == "__main__":
    mock_p3 = {
        "assets": {
            "tear_wound": {
                "sovereignty": "tear_factory",
                "slots": {
                    "slot_1_status": [1, 1, 0.9],
                    "slot_6_edge_physics": {"edge_stiffness": 0.5}
                }
            },
            "inner_shirt": {
                "sovereignty": "fabric_factory",
                "slots": {
                    "slot_1_status": [1, 1, 0.8],
                    "slot_6_physics_layer": {"depth_rank": 1, "collision_offset": 0.01, "friction_coeff": 0.3}
                }
            },
            "face_joint": {
                "sovereignty": "joint_factory",
                "slots": {"slot_1_status": [1, 1, 0.5]}
            }
        }
    }
    feedback = P3DynamicFeedback(seed=123)
    result = feedback.process(mock_p3)
    for aid, data in result["assets"].items():
        print(aid)
        if "slot_6_edge_physics" in data["slots"]:
            print("  turbulence_factor:", data["slots"]["slot_6_edge_physics"].get("turbulence_factor"))
        if "slot_6_physics_layer" in data["slots"]:
            print("  friction_coeff:", data["slots"]["slot_6_physics_layer"].get("friction_coeff"))