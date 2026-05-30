# p3_material_memory.py (加固版)
import math
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class MemoryTile:
    peak_stress: float = 0.0
    peak_stress_dir: Tuple[float, float] = (0.0, 0.0)
    plastic_displacement: float = 0.0
    crease_depth: float = 0.0
    last_update_frame: int = 0
    hysteresis: float = 0.0


class P3MaterialMemory:
    MATERIAL_PROPS = {
        "silk": {
            "elastic_limit": 0.3,
            "plastic_persistence": 0.1,
            "crease_retention": 0.05,
            "decay_rate": 0.9,
            "relaxation_time": 0.5,
            "yield_threshold": 0.4,
            "plastic_ratio": 0.1,
        },
        "denim": {
            "elastic_limit": 0.5,
            "plastic_persistence": 0.6,
            "crease_retention": 0.2,
            "decay_rate": 0.3,
            "relaxation_time": 2.0,
            "yield_threshold": 0.6,
            "plastic_ratio": 0.5,
        },
        "leather": {
            "elastic_limit": 0.8,
            "plastic_persistence": 0.9,
            "crease_retention": 0.8,
            "decay_rate": 0.05,
            "relaxation_time": 5.0,
            "yield_threshold": 0.9,
            "plastic_ratio": 0.8,
        },
        "cotton": {
            "elastic_limit": 0.4,
            "plastic_persistence": 0.4,
            "crease_retention": 0.15,
            "decay_rate": 0.5,
            "relaxation_time": 1.0,
            "yield_threshold": 0.5,
            "plastic_ratio": 0.3,
        },
        "default": {
            "elastic_limit": 0.5,
            "plastic_persistence": 0.4,
            "crease_retention": 0.1,
            "decay_rate": 0.5,
            "relaxation_time": 1.0,
            "yield_threshold": 0.5,
            "plastic_ratio": 0.4,
        }
    }

    def __init__(self):
        self.memory: Dict[Tuple[str, int, int], MemoryTile] = {}
        self.frame = 0

    def _get_material_props(self, material_type: str) -> Dict:
        return self.MATERIAL_PROPS.get(material_type.lower(), self.MATERIAL_PROPS["default"])

    def _perp_vector(self, vec: Tuple[float, float]) -> Tuple[float, float]:
        dx, dy = vec
        return (-dy, dx)

    def update_tile(self, asset_id: str, tx: int, ty: int,
                    current_stress: float, stress_dir: Tuple[float, float],
                    material_type: str,
                    dt: float, ambient_vibration: float = 0.01) -> Dict[str, Any]:
        key = (asset_id, tx, ty)
        tile = self.memory.get(key)
        if tile is None:
            tile = MemoryTile()
            self.memory[key] = tile

        props = self._get_material_props(material_type)
        elastic_limit = props["elastic_limit"]
        plastic_persistence = props["plastic_persistence"]
        crease_retention = props["crease_retention"]
        decay_rate = props["decay_rate"]
        relaxation_time = max(props.get("relaxation_time", 1.0), 1e-6)  # 防止除以零
        yield_threshold = props.get("yield_threshold", elastic_limit)
        plastic_ratio = props.get("plastic_ratio", plastic_persistence)

        # 更新峰值应力
        if current_stress > tile.peak_stress:
            tile.peak_stress = current_stress
            tile.peak_stress_dir = stress_dir

        # 指数衰减
        decay_factor = math.exp(-dt / relaxation_time)
        tile.plastic_displacement *= decay_factor
        tile.crease_depth *= decay_factor

        # 塑性形变
        if current_stress > yield_threshold:
            plastic_increment = (current_stress - yield_threshold) * plastic_ratio * dt
            tile.plastic_displacement = min(1.0, tile.plastic_displacement + plastic_increment)

        # 折痕深度
        peak_factor = min(1.0, tile.peak_stress / 1.0)  # 假设峰值应力归一化到[0,1]
        target_crease = peak_factor * crease_retention
        blend = 1.0 - math.exp(-dt / relaxation_time)
        tile.crease_depth = tile.crease_depth * (1 - blend) + target_crease * blend
        tile.crease_depth = max(0.0, min(1.0, tile.crease_depth))

        # 环境微振动恢复
        if current_stress < elastic_limit * 0.5:
            recovery = decay_rate * dt * (1 + ambient_vibration)
            tile.crease_depth = max(0.0, tile.crease_depth - recovery)
            tile.plastic_displacement = max(0.0, tile.plastic_displacement - recovery * 0.5)

        # 折痕方向
        if tile.peak_stress_dir == (0.0, 0.0):
            crease_dir = (1.0, 0.0)  # 默认方向
        else:
            crease_dir = self._perp_vector(tile.peak_stress_dir)

        crease_data = {
            "active": tile.crease_depth > 0.01,
            "depth": tile.crease_depth,
            "direction": crease_dir,
            "plastic_displacement": tile.plastic_displacement,
            "peak_stress": tile.peak_stress
        }

        tile.last_update_frame = self.frame
        return crease_data

    def advance_frame(self):
        self.frame += 1

    def process_tiles(self, backbone, p3_package: Dict[str, Any],
                      dt: float, ambient_vibration: float = 0.01) -> Dict[str, Any]:
        assets = p3_package.get("assets", {})
        tile_info = p3_package.get("global_tile_info", {})

        # 将 tile_info 按键格式 "asset_id_tx_ty" 解析为 (asset_id, tx, ty)
        # 使用安全的解析方式：找到最后两个下划线分开的数字
        asset_tiles = defaultdict(list)
        for key, state in tile_info.items():
            # 尝试从右侧开始匹配 _数字_数字 的模式
            parts = key.rsplit('_', 2)
            if len(parts) == 3 and parts[-2].isdigit() and parts[-1].isdigit():
                aid = parts[0]
                tx = int(parts[-2])
                ty = int(parts[-1])
                asset_tiles[aid].append((tx, ty, state))
            else:
                # 格式不符合预期，跳过
                continue

        for asset_id, asset_data in assets.items():
            if not isinstance(asset_data, dict):
                continue
            sovereignty = asset_data.get("sovereignty", "")
            if sovereignty != "fabric_factory":
                continue

            slots = asset_data.get("slots", {})
            tex_slot = slots.get("slot_5_texture", {})
            material_type = tex_slot.get("material_type", "default")

            tiles_info = asset_tiles.get(asset_id, [])
            if not tiles_info:
                continue

            # 检查 backbone 是否提供了必要的接口
            if not hasattr(backbone, 'partition') or not hasattr(backbone.partition, 'tile_w'):
                continue
            tile_w = backbone.partition.tile_w
            tile_h = backbone.partition.tile_h

            if "crease_tiles" not in slots:
                slots["crease_tiles"] = {}

            for tx, ty, state in tiles_info:
                cx = (tx + 0.5) * tile_w
                cy = (ty + 0.5) * tile_h

                # 安全获取应力矢量，支持不同返回格式
                stress_result = backbone.get_stress_vector_at(cx, cy)
                if isinstance(stress_result, (tuple, list)) and len(stress_result) >= 2:
                    stress_mag = float(stress_result[0])
                    stress_dir = (float(stress_result[1][0]), float(stress_result[1][1])) if isinstance(stress_result[1], (tuple, list)) else (1.0, 0.0)
                else:
                    # 降级：只取标量值，方向默认
                    stress_mag = float(stress_result) if stress_result is not None else 0.0
                    stress_dir = (1.0, 0.0)

                crease = self.update_tile(
                    asset_id, tx, ty,
                    current_stress=stress_mag,
                    stress_dir=stress_dir,
                    material_type=material_type,
                    dt=dt,
                    ambient_vibration=ambient_vibration
                )
                slots["crease_tiles"][f"{tx}_{ty}"] = crease

            asset_data["slots"] = slots

        p3_package["assets"] = assets
        self.advance_frame()
        return p3_package