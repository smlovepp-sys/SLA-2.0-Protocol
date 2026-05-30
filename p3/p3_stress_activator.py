# p3_stress_activator.py
# P3 视觉增强子模块：根据演化位激活法线强度与湿润粗糙度调整
# 独立于主物理引擎，仅修改视觉槽位（slot_4 / slot_5）

from typing import Dict, Any

class P3StressActivator:
    """
    应力视觉激活器
    输入：经过 p3_physics_engine 处理后的 JSON 包
    输出：法线增强 & 湿润粗糙度降低后的数据包
    """

    def __init__(self):
        pass

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        if value != value:
            return min_val
        return max(min_val, min(max_val, value))

    def _get_evolution(self, asset_data: Dict[str, Any]) -> float:
        slots = asset_data.get("slots", {})
        status = slots.get("slot_1_status", [1, 1, 0.0])
        evo = self._safe_float(status[2] if len(status) > 2 else 0.0)
        return self._clamp(evo, 0.0, 1.0)

    def activate(self, p3_package: Dict[str, Any]) -> Dict[str, Any]:
        assets = p3_package.get("assets", {})
        if not assets:
            return p3_package

        for asset_id, asset_data in assets.items():
            sovereignty = asset_data.get("sovereignty", "unknown")
            if sovereignty == "joint_factory":
                continue

            evolution = self._get_evolution(asset_data)
            slots = asset_data.get("slots", {})

            # 1. 法线激活（slot_5_texture）
            if evolution > 0.8:
                if "slot_5_texture" not in slots:
                    slots["slot_5_texture"] = {}
                tex_slot = slots["slot_5_texture"]
                current = self._safe_float(tex_slot.get("normal_strength", 0.5))
                delta = (evolution - 0.8) * 1.5
                new_strength = current * (1.0 + delta)
                new_strength = self._clamp(new_strength, 0.0, 2.0)
                tex_slot["normal_strength"] = new_strength
                tex_slot["stress_activated"] = True
                slots["slot_5_texture"] = tex_slot

            # 2. 湿润粗糙度对冲（slot_4_material）
            if evolution > 0.5 and sovereignty in ["skin_factory", "fabric_factory"]:
                if "slot_4_material" not in slots:
                    slots["slot_4_material"] = {}
                mat_slot = slots["slot_4_material"]
                current_rough = self._safe_float(mat_slot.get("roughness", 0.5))
                reduction = min(0.4, evolution * 0.5)
                new_rough = current_rough - reduction
                new_rough = self._clamp(new_rough, 0.1, 1.0)
                mat_slot["roughness"] = new_rough
                mat_slot["wetness_enhanced"] = True
                slots["slot_4_material"] = mat_slot

            asset_data["slots"] = slots

        p3_package["assets"] = assets
        return p3_package


# 使用示例
if __name__ == "__main__":
    mock_p3 = {
        "assets": {
            "skin_arm": {
                "sovereignty": "skin_factory",
                "slots": {
                    "slot_1_status": [1, 1, 0.85],
                    "slot_5_texture": {"normal_strength": 0.6},
                    "slot_4_material": {"roughness": 0.5}
                }
            },
            "face_joint": {
                "sovereignty": "joint_factory",
                "slots": {"slot_1_status": [1, 1, 0.9]}
            }
        }
    }
    activator = P3StressActivator()
    result = activator.activate(mock_p3)
    for aid, data in result["assets"].items():
        print(aid)
        if "slot_5_texture" in data["slots"]:
            print("  normal_strength:", data["slots"]["slot_5_texture"].get("normal_strength"))