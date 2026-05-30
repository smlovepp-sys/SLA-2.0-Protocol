# p3_physics_engine.py (加固兼容版)
import math
from typing import Dict, Any, List, Optional

class P3PhysicsEngine:
    """
    P3 物理对冲引擎
    输入：P2 输出的 JSON 包
    输出：经过全槽位对冲、跨层压力、动态重力调整后的数据包
    """

    def __init__(self):
        self.tension_coeff = 0.5
        self.cross_layer_extra = 0.2

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        if value != value:  # NaN 检测
            return min_val
        return max(min_val, min(max_val, value))

    def _get_evolution(self, asset_data: Dict[str, Any]) -> float:
        slots = asset_data.get("slots", {})
        status = slots.get("slot_1_status", [1, 1, 0.0])
        # 安全获取第三个元素，若不存在则返回 0.0
        evo = 0.0
        if isinstance(status, (list, tuple)) and len(status) > 2:
            evo = self._safe_float(status[2], 0.0)
        return self._clamp(evo, 0.0, 1.0)

    def _is_outer_layer(self, asset_data: Dict[str, Any]) -> bool:
        sovereignty = asset_data.get("sovereignty", "")
        slots = asset_data.get("slots", {})
        layer_slot = slots.get("slot_6_physics_layer", {})
        depth_rank = self._safe_float(layer_slot.get("depth_rank", 2), 2)
        # 显式深度等级为 3 或更高，且主权为 fabric_factory 或任何资产
        if depth_rank >= 3:
            return True
        # 兼容旧逻辑：fabric_factory 且 rank=3
        if sovereignty == "fabric_factory" and depth_rank == 3:
            return True
        return False

    def _is_inner_layer(self, asset_data: Dict[str, Any]) -> bool:
        sovereignty = asset_data.get("sovereignty", "")
        slots = asset_data.get("slots", {})
        layer_slot = slots.get("slot_6_physics_layer", {})
        depth_rank = self._safe_float(layer_slot.get("depth_rank", 2), 2)
        # 内层：depth_rank == 1 或 sovereignty == fabric_factory 且 rank=1
        if depth_rank == 1:
            return True
        if sovereignty == "fabric_factory" and depth_rank == 1:
            return True
        return False

    def _find_inner_assets(self, outer_id: str, assets: Dict[str, Any]) -> List[str]:
        """基于深度等级寻找内层资产，不再依赖名称猜测"""
        inner_ids = []
        for aid, adata in assets.items():
            if aid == outer_id:
                continue
            if self._is_inner_layer(adata):
                inner_ids.append(aid)
        return inner_ids

    def process_physics(self, p2_package: Dict[str, Any]) -> Dict[str, Any]:
        assets = p2_package.get("assets", {})
        if not assets:
            return p2_package

        outer_assets = []

        for asset_id, asset_data in assets.items():
            # 安全地确保 asset_data 为字典（已由主控制器保证，但防御）
            if not isinstance(asset_data, dict):
                continue

            sovereignty = asset_data.get("sovereignty", "unknown")
            evolution = self._get_evolution(asset_data)
            slots = asset_data.get("slots", {})
            if not isinstance(slots, dict):
                slots = {}
                asset_data["slots"] = slots

            # --- 关节工厂特殊处理 ---
            if sovereignty == "joint_factory":
                for slot_name, slot_dict in slots.items():
                    if slot_name.startswith("slot_6_"):
                        if isinstance(slot_dict, dict):
                            slot_dict["collision_offset"] = 0.0
                            slot_dict["stiffness"] = 1.0
                            slot_dict["edge_stiffness"] = 1.0
                            slot_dict["gravity_bias"] = 0.0
                            slot_dict["gravity_multiplier"] = 0.0
                # 添加刚性锚点
                slots["slot_6_rigid_anchor"] = {"stiffness": 1.0}
                asset_data["slots"] = slots
                continue

            # --- 标准槽位对冲 ---
            for slot_name, slot_dict in slots.items():
                if not slot_name.startswith("slot_6_"):
                    continue
                if not isinstance(slot_dict, dict):
                    continue

                # 补齐缺失的核心字段
                if "collision_offset" not in slot_dict:
                    slot_dict["collision_offset"] = 0.02
                if "edge_stiffness" not in slot_dict:
                    slot_dict["edge_stiffness"] = 0.7

                # 碰撞偏移
                base_offset = self._safe_float(slot_dict.get("collision_offset", 0.01), 0.01)
                new_offset = base_offset * (1.0 - evolution * self.tension_coeff)
                new_offset = max(0.0015, new_offset)
                slot_dict["collision_offset"] = new_offset

                # 刚性
                base_stiff = self._safe_float(slot_dict.get("stiffness", 0.8), 0.8)
                new_stiff = base_stiff + (1.0 - base_stiff) * math.tanh(evolution * 3.0) * 0.5
                new_stiff = self._clamp(new_stiff, 0.0, 1.0)
                slot_dict["stiffness"] = new_stiff

                # 边缘刚性
                base_edge = self._safe_float(slot_dict.get("edge_stiffness", 0.7), 0.7)
                new_edge = base_edge + (1.0 - base_edge) * math.tanh(evolution * 3.0) * 0.5
                new_edge = self._clamp(new_edge, 0.0, 1.0)
                slot_dict["edge_stiffness"] = new_edge

                # 重力偏置
                if evolution > 0.8:
                    slot_dict["gravity_bias"] = 0.2
                else:
                    slot_dict["gravity_bias"] = 0.0

            # --- 动态重力倍数 ---
            mass_bias = evolution * 0.15
            mass_bias = self._clamp(mass_bias, 0.0, 0.15)
            # 确保 slot_6_physics_layer 存在
            if "slot_6_physics_layer" not in slots:
                slots["slot_6_physics_layer"] = {}
            slots["slot_6_physics_layer"]["gravity_multiplier"] = mass_bias

            # 记录外层资产
            if self._is_outer_layer(asset_data):
                outer_assets.append(asset_id)

            asset_data["slots"] = slots

        # 跨层对冲：外层资产影响内层
        for outer_id in outer_assets:
            outer_data = assets.get(outer_id)
            if not outer_data:
                continue
            outer_evo = self._get_evolution(outer_data)
            if outer_evo > 0.5:
                inner_ids = self._find_inner_assets(outer_id, assets)
                for inner_id in inner_ids:
                    inner_data = assets.get(inner_id)
                    if not inner_data or not self._is_inner_layer(inner_data):
                        continue
                    inner_slots = inner_data.get("slots", {})
                    layer_slot = inner_slots.get("slot_6_physics_layer", {})
                    if "collision_offset" in layer_slot:
                        current_offset = self._safe_float(layer_slot["collision_offset"], 0.01)
                        new_offset = current_offset * (1.0 - self.cross_layer_extra)
                        new_offset = max(0.0015, new_offset)
                        layer_slot["collision_offset"] = new_offset
                        inner_slots["slot_6_physics_layer"] = layer_slot
                        inner_data["slots"] = inner_slots
                        assets[inner_id] = inner_data

        p2_package["assets"] = assets
        return p2_package