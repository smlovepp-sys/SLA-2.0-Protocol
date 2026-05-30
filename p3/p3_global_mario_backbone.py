# p3_global_mario_backbone.py (加固版：应力隔离警告 + 可控默认应力)
import math
from typing import Dict, Any, List, Tuple, Set, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class TileState:
    level: int
    last_update_frame: int
    stress_magnitude: float
    stress_vector: Tuple[float, float]
    snapshots: Dict[str, Any]
    hooks_triggered: Set[str]

class GlobalSpatialPartition:
    def __init__(self, world_size: Tuple[float, float], tile_resolution: Tuple[int, int]):
        self.world_w, self.world_h = world_size
        self.tiles_x, self.tiles_y = tile_resolution
        self.tile_w = self.world_w / self.tiles_x
        self.tile_h = self.world_h / self.tiles_y
        self.tiles: Dict[Tuple[str, int, int], TileState] = {}
        # ⚠️ 当前应力场不区分资产，多资产时可能混淆（设计简化）
        self.stress_field: Dict[Tuple[int, int], Tuple[float, Tuple[float, float]]] = {}
        self.frame_counter = 0

    def get_tile_coord(self, x: float, y: float) -> Tuple[int, int]:
        tx = max(0, min(self.tiles_x - 1, int(x / self.tile_w)))
        ty = max(0, min(self.tiles_y - 1, int(y / self.tile_h)))
        return tx, ty

    def get_stress_vector_at(self, x: float, y: float) -> Tuple[float, Tuple[float, float]]:
        fx = x / self.tile_w - 0.5
        fy = y / self.tile_h - 0.5
        ix = math.floor(fx)
        iy = math.floor(fy)
        wx = fx - ix
        wy = fy - iy

        def get_tile_stress(tx: int, ty: int):
            tx_clamped = max(0, min(tx, self.tiles_x - 1))
            ty_clamped = max(0, min(ty, self.tiles_y - 1))
            key = (tx_clamped, ty_clamped)
            if key in self.stress_field:
                return self.stress_field[key]
            return (0.0, (0.0, 0.0))

        v00 = get_tile_stress(ix, iy)
        v10 = get_tile_stress(ix + 1, iy)
        v01 = get_tile_stress(ix, iy + 1)
        v11 = get_tile_stress(ix + 1, iy + 1)

        mag = (1-wx)*(1-wy)*v00[0] + wx*(1-wy)*v10[0] + (1-wx)*wy*v01[0] + wx*wy*v11[0]
        dx = (1-wx)*(1-wy)*v00[1][0] + wx*(1-wy)*v10[1][0] + (1-wx)*wy*v01[1][0] + wx*wy*v11[1][0]
        dy = (1-wx)*(1-wy)*v00[1][1] + wx*(1-wy)*v10[1][1] + (1-wx)*wy*v01[1][1] + wx*wy*v11[1][1]

        norm = math.hypot(dx, dy)
        if norm > 1e-6:
            dx /= norm
            dy /= norm
        return (mag, (dx, dy))

    def update_tile_state(self, asset_id: str, tx: int, ty: int, new_level: int,
                          stress_mag: float, stress_vec: Tuple[float, float]) -> Set[str]:
        key = (asset_id, tx, ty)
        old_state = self.tiles.get(key)
        old_mag = old_state.stress_magnitude if old_state else 0.0

        # 动态阻尼
        if stress_mag > old_mag:
            alpha = 0.7
        else:
            alpha = 0.02
        stress_mag = old_mag * (1.0 - alpha) + stress_mag * alpha

        SAFE_THRESHOLD = 0.8
        stress_mag_limited = SAFE_THRESHOLD * math.tanh(stress_mag / SAFE_THRESHOLD)

        # 各向异性偏置
        dx, dy = stress_vec
        dx *= 1.15
        norm = math.hypot(dx, dy)
        if norm > 1e-6:
            stress_vec = (dx / norm, dy / norm)

        hooks = set()
        tear_threshold = SAFE_THRESHOLD * 0.5
        if stress_mag_limited > tear_threshold and (old_state is None or old_state.stress_magnitude <= tear_threshold):
            hooks.add("tear_trigger")
        if old_state is None:
            old_level = 2
        else:
            old_level = old_state.level
        if old_level != new_level:
            if old_level == 1 and new_level == 0:
                hooks.add("collision_enter")
            if old_level == 0 and new_level == 1:
                hooks.add("collision_exit")

        conservation_gain = 1.48 / math.sqrt(48)
        stress_mag_stored = stress_mag_limited * conservation_gain

        new_state = TileState(
            level=new_level,
            last_update_frame=self.frame_counter,
            stress_magnitude=stress_mag_stored,
            stress_vector=stress_vec,
            snapshots=old_state.snapshots if old_state else {},
            hooks_triggered=hooks
        )
        self.tiles[key] = new_state
        self.stress_field[(tx, ty)] = (stress_mag_stored, stress_vec)
        return hooks

    def advance_frame(self):
        self.frame_counter += 1


class P3GlobalMarioBackbone:
    def __init__(self, world_size: Tuple[float, float] = (1.0, 1.0),
                 tile_resolution: Tuple[int, int] = (32, 32),
                 micro_pool_size: int = 1024,
                 default_stress_magnitude: float = 0.0):
        """
        default_stress_magnitude: 无应力源时瓦片的默认应力，建议设为 0.0 避免虚假响应
        """
        self.partition = GlobalSpatialPartition(world_size, tile_resolution)
        self.tile_resolution = tile_resolution
        self.default_stress_mag = default_stress_magnitude
        self.event_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.micro_pool = [{
            "active": False,
            "type": "none",
            "transform": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "asset_id": "",
            "tile_coord": (0, 0),
            "life": 0.0
        } for _ in range(micro_pool_size)]
        self.stress_points: Dict[str, List[Tuple[float, float, float]]] = {}
        self.collision_points: Dict[str, List[Tuple[float, float]]] = {}
        self.camera_view = (0.5, 0.5, 1.0, 1.0)
        self._first_frame = True

    def register_event_listener(self, event_type: str, callback: Callable):
        self.event_listeners[event_type].append(callback)

    def _dispatch_event(self, event_type: str, data: Dict[str, Any]):
        for cb in self.event_listeners.get(event_type, []):
            try:
                cb(data)
            except Exception as e:
                print(f"Event callback error: {e}")

    def set_stress_points(self, stress_points: Dict[str, List[Tuple[float, float, float]]]):
        self.stress_points = stress_points

    def set_collision_points(self, collision_points: Dict[str, List[Tuple[float, float]]]):
        self.collision_points = collision_points

    def set_camera_view(self, cx: float, cy: float, width: float, height: float):
        self.camera_view = (cx, cy, width, height)

    def get_stress_vector_at(self, x: float, y: float) -> Tuple[float, Tuple[float, float]]:
        return self.partition.get_stress_vector_at(x, y)

    def update_tile_states(self, p3_package: Dict[str, Any]) -> Dict[str, Any]:
        assets = p3_package.get("assets", {})
        target_levels: Dict[Tuple[str, int, int], int] = {}

        # 日志（仅首次）
        if self._first_frame:
            print("\n========== [应力点 → 瓦片映射] ==========")
            printed = 0
            for asset_id, points in self.stress_points.items():
                for (x, y, mag) in points:
                    if printed >= 5:
                        break
                    tx, ty = self.partition.get_tile_coord(x, y)
                    print(f"  点[{printed}] 资产={asset_id} 坐标({x:.3f},{y:.3f}) → 瓦片({tx},{ty}) 强度={mag:.3f}")
                    printed += 1
                if printed >= 5:
                    break
            if printed == 0:
                print("  ⚠️ 没有应力点")
            print("============================================\n")
            self._first_frame = False

        # 1. 从应力点和碰撞点生成 L0 瓦片
        for asset_id, points in self.stress_points.items():
            for (x, y, mag) in points:
                tx, ty = self.partition.get_tile_coord(x, y)
                target_levels[(asset_id, tx, ty)] = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = tx + dx, ty + dy
                        if 0 <= nx < self.tile_resolution[0] and 0 <= ny < self.tile_resolution[1]:
                            target_levels[(asset_id, nx, ny)] = 0

        for asset_id, points in self.collision_points.items():
            for (x, y) in points:
                tx, ty = self.partition.get_tile_coord(x, y)
                target_levels[(asset_id, tx, ty)] = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = tx + dx, ty + dy
                        if 0 <= nx < self.tile_resolution[0] and 0 <= ny < self.tile_resolution[1]:
                            target_levels[(asset_id, nx, ny)] = 0

        # 2. 相机视锥分级
        cx, cy, vw, vh = self.camera_view
        for asset_id, asset_data in assets.items():
            if not isinstance(asset_data, dict):
                continue
            for tx in range(self.tile_resolution[0]):
                for ty in range(self.tile_resolution[1]):
                    key = (asset_id, tx, ty)
                    if key in target_levels:
                        continue
                    tile_cx = (tx + 0.5) * self.partition.tile_w
                    tile_cy = (ty + 0.5) * self.partition.tile_h
                    if abs(tile_cx - cx) <= vw/2 and abs(tile_cy - cy) <= vh/2:
                        target_levels[key] = 1
                    else:
                        target_levels[key] = 2

        # 3. 点强度查找表
        POINT_GAIN = 3.0
        point_strength = {}
        for asset_id, points in self.stress_points.items():
            for (x, y, mag) in points:
                tx, ty = self.partition.get_tile_coord(x, y)
                key = (asset_id, tx, ty)
                amplified = mag * POINT_GAIN
                if key not in point_strength or amplified > point_strength[key]:
                    point_strength[key] = amplified

        # 4. 更新瓦片状态
        for (asset_id, tx, ty), new_level in target_levels.items():
            tile_cx = (tx + 0.5) * self.partition.tile_w
            tile_cy = (ty + 0.5) * self.partition.tile_h

            pkey = (asset_id, tx, ty)
            if pkey in point_strength:
                stress_mag = point_strength[pkey]
                _, stress_dir = self.get_stress_vector_at(tile_cx, tile_cy)
                if stress_dir == (0.0, 0.0):
                    stress_dir = (1.0, 0.2)
            else:
                stress_mag, stress_dir = self.get_stress_vector_at(tile_cx, tile_cy)
                # 若无历史应力且当前应力为0，使用默认应力（可配置）
                if (tx, ty) not in self.partition.stress_field and stress_mag == 0.0:
                    stress_mag = self.default_stress_mag
                    if stress_dir == (0.0, 0.0):
                        stress_dir = (1.0, 0.2)

            hooks = self.partition.update_tile_state(asset_id, tx, ty, new_level,
                                                     stress_mag, stress_dir)
            for hook in hooks:
                self._dispatch_event(hook, {
                    "asset_id": asset_id,
                    "tile": (tx, ty),
                    "level": new_level,
                    "stress_magnitude": stress_mag,
                    "stress_vector": stress_dir,
                    "world_pos": (tile_cx, tile_cy)
                })

        # 构建 tile_info
        tile_info = {}
        for (asset_id, tx, ty), state in self.partition.tiles.items():
            tile_info[f"{asset_id}_{tx}_{ty}"] = {
                "level": state.level,
                "last_update": state.last_update_frame,
                "stress_mag": state.stress_magnitude
            }
        p3_package["global_tile_info"] = tile_info
        p3_package["frame_counter"] = self.partition.frame_counter
        return p3_package

    # ---- 微实例池方法保持不变 ----
    def allocate_micro_instance(self, inst_type: str, asset_id: str,
                                tile_coord: Tuple[int, int],
                                transform: List[float], life: float = 1.0) -> Optional[int]:
        for idx, inst in enumerate(self.micro_pool):
            if not inst["active"]:
                inst["active"] = True
                inst["type"] = inst_type
                inst["asset_id"] = asset_id
                inst["tile_coord"] = tile_coord
                inst["transform"] = transform[:]
                inst["life"] = life
                return idx
        return None

    def update_micro_instances(self, dt: float):
        for inst in self.micro_pool:
            if inst["active"]:
                inst["life"] -= dt
                if inst["life"] <= 0:
                    inst["active"] = False

    def batch_update_transforms(self, transforms: Dict[int, List[float]]):
        for idx, mat in transforms.items():
            if 0 <= idx < len(self.micro_pool):
                self.micro_pool[idx]["transform"] = mat[:]

    def get_micro_pool_snapshot(self) -> List[Dict]:
        return [inst for inst in self.micro_pool if inst["active"]]

    def process_frame(self, p3_package: Dict[str, Any], dt: float = 1.0/30.0) -> Dict[str, Any]:
        p3_package = self.update_tile_states(p3_package)
        self.update_micro_instances(dt)
        p3_package["micro_pool_snapshot"] = self.get_micro_pool_snapshot()
        self.partition.advance_frame()
        return p3_package