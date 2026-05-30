# p3_topology_adapter.py
# P3 穿脱特化模块：处理服装穿脱过程中的开口应力、剥离粘滞、堆叠修复和关节平滑

from typing import Dict, Any, Optional, List

class P3TopologyAdapter:
    """
    穿脱拓扑适配器
    输入：经过 P3 主引擎处理后的 JSON 包，可选动作标志
    输出：针对穿脱场景优化后的数据包
    """

    def __init__(self):
        self.cuff_expansion_factor = 4.0
        self.peeling_factor = 2.0
        self.layer_threshold = 3
        self.joint_stiffness_lock = 1.0

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        if value != value:
            return min_val
        return max(min_val, min(max_val, value))

    def _is_cuff_collar_hem(self, asset_id: str) -> bool:
        keywords = ["cuff", "collar", "hem", "sleeve_end", "neckline"]
        aid_lower = asset_id.lower()
        return any(kw in aid_lower for kw in keywords)

    def _is_inner_layer(self, asset_data: Dict[str, Any]) -> bool:
        sovereignty = asset_data.get("sovereignty", "")
        if sovereignty != "fabric_factory":
            return False
        slots = asset_data.get("slots", {})
        layer_slot = slots.get("slot_6_physics_layer", {})
        depth_rank = layer_slot.get("depth_rank", 2)
        return depth_rank == 1

    def _is_joint_region(self, asset_id: str) -> bool:
        joint_keywords = ["wrist", "elbow", "knee", "ankle", "shoulder", "hip", "neck"]
        aid_lower = asset_id.lower()
        return any(kw in aid_lower for kw in joint_keywords)

    def _count_fabric_layers(self, assets: Dict[str, Any]) -> Dict[str, int]:
        zone_layers = {}
        for aid, data in assets.items():
            sovereignty = data.get("sovereignty", "")
            if sovereignty != "fabric_factory":
                continue
            parts = aid.split('_')
            zone = parts[0] if parts else "default"
            zone_layers[zone] = zone_layers.get(zone, 0) + 1
        return zone_layers

    def process(self, p3_package: Dict[str, Any], action: Optional[str] = None) -> Dict[str, Any]:
        assets = p3_package.get("assets", {})
        if not assets or action not in ["dressing", "undressing"]:
            return p3_package

        zone_layers = self._count_fabric_layers(assets)

        for asset_id, asset_data in assets.items():
            sovereignty = asset_data.get("sovereignty", "unknown")
            slots = asset_data.get("slots", {})

            # 1. 开口应力补偿
            if self._is_cuff_collar_hem(asset_id):
                if "slot_6_physics_layer" not in slots:
                    slots["slot_6_physics_layer"] = {}
                layer_slot = slots["slot_6_physics_layer"]
                if "collision_offset" in layer_slot:
                    original = self._safe_float(layer_slot["collision_offset"], 0.01)
                    expanded = original * self.cuff_expansion_factor
                    expanded = min(expanded, 0.2)
                    layer_slot["collision_offset"] = expanded
                    layer_slot["cuff_expanded"] = True

            # 2. 内层剥离粘滞
            if self._is_inner_layer(asset_data):
                if "slot_6_physics_layer" not in slots:
                    slots["slot_6_physics_layer"] = {}
                layer_slot = slots["slot_6_physics_layer"]
                if "friction_coeff" in layer_slot:
                    new_friction = layer_slot["friction_coeff"] * 0.2
                    layer_slot["friction_coeff"] = self._clamp(new_friction, 0.0, 0.1)
                layer_slot["elastic_recovery"] = self.peeling_factor
                if "stiffness" in layer_slot:
                    layer_slot["stiffness"] = self._clamp(layer_slot["stiffness"] * 0.5, 0.0, 0.5)

            # 3. 堆叠自修复
            zone = asset_id.split('_')[0] if '_' in asset_id else "default"
            layer_count = zone_layers.get(zone, 1)
            if layer_count > self.layer_threshold:
                if "slot_6_physics_layer" not in slots:
                    slots["slot_6_physics_layer"] = {}
                layer_slot = slots["slot_6_physics_layer"]
                extra = (layer_count - self.layer_threshold) * 0.01
                repulsion = min(0.05, extra)
                layer_slot["repulsion_bias"] = repulsion
                if "collision_offset" in layer_slot:
                    layer_slot["collision_offset"] += repulsion * 0.5

            # 4. 关节平滑引导
            if self._is_joint_region(asset_id) and sovereignty == "fabric_factory":
                if "slot_6_physics_layer" not in slots:
                    slots["slot_6_physics_layer"] = {}
                layer_slot = slots["slot_6_physics_layer"]
                layer_slot["joint_smoothing"] = True
                if "damping" not in layer_slot:
                    layer_slot["damping"] = 0.9
                else:
                    layer_slot["damping"] = max(layer_slot["damping"], 0.8)

            # 主权尊重：joint_factory 不修改
            if sovereignty == "joint_factory":
                continue

            asset_data["slots"] = slots

        p3_package["assets"] = assets
        return p3_package


# 使用示例
if __name__ == "__main__":
    mock_p3 = {
        "assets": {
            "torso_outer_jacket": {
                "sovereignty": "fabric_factory",
                "slots": {
                    "slot_1_status": [1, 1, 0.7],
                    "slot_6_physics_layer": {"collision_offset": 0.01, "depth_rank": 3, "friction_coeff": 0.5}
                }
            },
            "torso_inner_shirt": {
                "sovereignty": "fabric_factory",
                "slots": {
                    "slot_1_status": [1, 1, 0.4],
                    "slot_6_physics_layer": {"collision_offset": 0.005, "depth_rank": 1, "friction_coeff": 0.6}
                }
            },
            "cuff_left": {
                "sovereignty": "fabric_factory",
                "slots": {
                    "slot_6_physics_layer": {"collision_offset": 0.008, "depth_rank": 2}
                }
            },
            "wrist_joint": {
                "sovereignty": "joint_factory",
                "slots": {}
            }
        }
    }
    adapter = P3TopologyAdapter()
    result = adapter.process(mock_p3, action="dressing")
    for aid, data in result["assets"].items():
        print(aid)
        if "slot_6_physics_layer" in data["slots"]:
            layer = data["slots"]["slot_6_physics_layer"]
            print("  collision_offset:", layer.get("collision_offset"))
            print("  friction_coeff:", layer.get("friction_coeff"))
            print("  repulsion_bias:", layer.get("repulsion_bias"))
            print("  joint_smoothing:", layer.get("joint_smoothing"))
            print("  elastic_recovery:", layer.get("elastic_recovery"))