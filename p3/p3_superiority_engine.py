# p3_superiority_engine.py
# P3 高级物理引擎：动力学因果锚点、物质记忆与折痕衰减、容积坍缩与自排斥堆叠、多相材质特征映射
# 修复：应力增量正确使用 dt，增强容错
# ✅ 特别修复：引入硬核类型防御锁，自动过滤上游散落进 assets 的 float 常数（如 stiffness/cfg_scale），彻底阻断 'float' object has no attribute 'get' 崩溃

import math
from typing import Dict, Any, Optional, List, Tuple

class P3SuperiorityEngine:
    """
    高级物理引擎
    特点：
    - 动力学因果锚点：手部抓取时产生放射状张力场，法线褶皱实时生成
    - 物质记忆与折痕衰减：应力指数衰减，布料揉皱后缓慢恢复
    - 容积坍缩与自排斥堆叠：肢体脱离后衣物塌陷，多层织物斥力偏置
    - 多相材质特征：皮革粘滞、丝绸滑落等
    """

    def __init__(self):
        self.crease_decay_k = 0.1
        self.grab_stiffness_multiplier = 3.0
        self.collapse_radial_support = 0.0
        self.layer_threshold = 3
        self.repulsion_base = 0.005
        self.material_props = {
            "leather": {"friction_coeff": 0.8, "viscosity": 0.6, "elastic_recovery": 0.2},
            "silk":    {"friction_coeff": 0.2, "viscosity": 0.1, "elastic_recovery": 0.9},
            "denim":   {"friction_coeff": 0.6, "viscosity": 0.4, "elastic_recovery": 0.5},
            "cotton":  {"friction_coeff": 0.5, "viscosity": 0.3, "elastic_recovery": 0.4},
            "default": {"friction_coeff": 0.5, "viscosity": 0.3, "elastic_recovery": 0.4},
        }
        self.stress_buffer: Dict[str, Dict[str, float]] = {}

    def _safe_float(self, val: Any, default=0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _clamp(self, val: float, lo=0.0, hi=1.0) -> float:
        if val != val:
            return lo
        return max(lo, min(hi, val))

    def _get_evolution(self, asset_data: Dict) -> float:
        slots = asset_data.get("slots", {})
        status = slots.get("slot_1_status", [1, 1, 0.0])
        return self._safe_float(status[2] if len(status) > 2 else 0.0)

    def _is_fabric(self, asset_data: Dict) -> bool:
        # 🛡️ 类型防御层：如果资产数据由于上游污染变成了 float，安全拦截返回 False
        if not isinstance(asset_data, dict):
            return False
        return asset_data.get("sovereignty") == "fabric_factory"

    def _is_skin(self, asset_data: Dict) -> bool:
        # 🛡️ 类型防御层
        if not isinstance(asset_data, dict):
            return False
        return asset_data.get("sovereignty") == "skin_factory"

    def _is_joint(self, asset_data: Dict) -> bool:
        # 🛡️ 类型防御层
        if not isinstance(asset_data, dict):
            return False
        return asset_data.get("sovereignty") == "joint_factory"

    def _get_material_type(self, asset_data: Dict) -> str:
        slots = asset_data.get("slots", {})
        tex = slots.get("slot_5_texture", {})
        mat = tex.get("material_type", "default").lower()
        if mat not in self.material_props:
            mat = "default"
        return mat

    def _count_fabric_layers(self, assets: Dict[str, Dict], zone_map: Dict[str, str] = None) -> Dict[str, int]:
        if not isinstance(assets, dict):
            return {}
            
        if zone_map is None:
            zone_map = {aid: aid.split('_')[0] if '_' in aid else "default" for aid, d in assets.items() if isinstance(d, dict)}
        
        zone_layers = {}
        for aid, data in assets.items():
            # 🛡️ 物理常数熔断：如果遍历出的 data 是 float（例如 stiffness），直接跳过
            if not isinstance(data, dict):
                continue
            if not self._is_fabric(data):
                continue
            zone = zone_map.get(aid, "default")
            zone_layers[zone] = zone_layers.get(zone, 0) + 1
        return zone_layers

    def _apply_material_properties(self, asset_data: Dict, slots: Dict) -> Dict:
        mat = self._get_material_type(asset_data)
        props = self.material_props[mat]
        if "slot_6_physics_layer" not in slots:
            slots["slot_6_physics_layer"] = {}
        layer_slot = slots["slot_6_physics_layer"]
        if "friction_coeff" not in layer_slot:
            layer_slot["friction_coeff"] = props["friction_coeff"]
        if "viscosity" not in layer_slot:
            layer_slot["viscosity"] = props["viscosity"]
        if "elastic_recovery" not in layer_slot:
            layer_slot["elastic_recovery"] = props["elastic_recovery"]
        return slots

    def _update_stress_buffer(self, asset_id: str, stress_delta: float, dt: float) -> float:
        if asset_id not in self.stress_buffer:
            self.stress_buffer[asset_id] = {"stress_magnitude": 0.0, "crease_buffer": {}}
        buf = self.stress_buffer[asset_id]
        decay = math.exp(-self.crease_decay_k * dt)
        buf["stress_magnitude"] *= decay
        buf["stress_magnitude"] = self._clamp(buf["stress_magnitude"] + stress_delta, 0.0, 1.0)
        return buf["stress_magnitude"]

    def process_frame(self, p3_package: Dict[str, Any],
                      hand_grab_info: Optional[Dict[str, Tuple[str, Tuple[float, float, float]]]] = None,
                      dt: float = 1.0 / 30.0) -> Dict[str, Any]:
        assets = p3_package.get("assets", {})
        if not assets or not isinstance(assets, dict):
            return p3_package

        # 🛡️ 建立 zone_map 时，提前剔除资产字典中混入的非 dict 脏数据
        zone_map = {aid: aid.split('_')[0] if '_' in aid else "default" for aid, d in assets.items() if isinstance(d, dict)}
        zone_layers = self._count_fabric_layers(assets, zone_map)

        for asset_id, asset_data in list(assets.items()):
            # ==================== 🛡️ 核心常数熔断网 ====================
            # 如果发现上游把 float 或者 int 扔进了资产池，直接跳过它，防止逻辑击穿
            if not isinstance(asset_data, dict):
                continue
            # ========================================================

            sovereignty = asset_data.get("sovereignty", "unknown")
            slots = asset_data.get("slots", {})
            evolution = self._get_evolution(asset_data)

            # 1. 多相材质特征映射
            if sovereignty in ["fabric_factory", "skin_factory"]:
                slots = self._apply_material_properties(asset_data, slots)

            # 2. 动力学因果锚点（处理抓取）
            if self._is_fabric(asset_data) and hand_grab_info and asset_id in hand_grab_info:
                joint_id, grab_point = hand_grab_info[asset_id]
                joint_data = assets.get(joint_id, {})
                
                # 针对 joint_data 的安全防护
                if isinstance(joint_data, dict):
                    motion_vec = joint_data.get("motion_vector", [0.0, 0.0, 0.0])
                else:
                    motion_vec = [0.0, 0.0, 0.0]
                    
                if len(motion_vec) < 3:
                    motion_vec = [0.0, 0.0, 0.0]
                motion_mag = math.sqrt(motion_vec[0]**2 + motion_vec[1]**2 + motion_vec[2]**2)

                if "slot_6_physics_layer" not in slots:
                    slots["slot_6_physics_layer"] = {}
                layer_slot = slots["slot_6_physics_layer"]
                original_stiff = self._safe_float(layer_slot.get("stiffness", 0.5))
                layer_slot["stiffness"] = self._clamp(original_stiff * self.grab_stiffness_multiplier, 0.0, 1.0)
                layer_slot["tension_center"] = {
                    "active": True,
                    "position": grab_point,
                    "stiffness_multiplier": self.grab_stiffness_multiplier
                }

                if motion_mag > 0.01:
                    if "slot_5_texture" not in slots:
                        slots["slot_5_texture"] = {}
                    tex_slot = slots["slot_5_texture"]
                    dx, dy, dz = [v / motion_mag for v in motion_vec]
                    tex_slot["anisotropic_stretch"] = {
                        "direction": [dx, dy, dz],
                        "intensity": self._clamp(motion_mag * 1.5, 0.0, 1.0)
                    }
                    current_ns = self._safe_float(tex_slot.get("normal_strength", 0.5))
                    new_ns = current_ns + motion_mag * 0.8
                    tex_slot["normal_strength"] = self._clamp(new_ns, 0.0, 2.0)

                stress_delta = motion_mag * 0.5 * dt
                current_stress = self._update_stress_buffer(asset_id, stress_delta, dt)
                layer_slot["stress_magnitude"] = current_stress

            # 3. 物质记忆与折痕衰减（主动恢复）
            if self._is_fabric(asset_data):
                buf = self.stress_buffer.get(asset_id, {"stress_magnitude": 0.0})
                stress = buf["stress_magnitude"]

                mat_props = self.material_props.get(self._get_material_type(asset_data), self.material_props["default"])
                elastic_recovery = mat_props["elastic_recovery"]
                recovery = elastic_recovery * dt * 0.5
                stress = max(0.0, stress - recovery)
                if asset_id in self.stress_buffer:
                    self.stress_buffer[asset_id]["stress_magnitude"] = stress

                if "slot_5_texture" in slots:
                    tex_slot = slots["slot_5_texture"]
                    current_rough = self._safe_float(tex_slot.get("roughness", 0.5))
                    new_rough = current_rough + stress * 0.3
                    tex_slot["roughness"] = self._clamp(new_rough, 0.0, 1.0)
                    current_ns = self._safe_float(tex_slot.get("normal_strength", 0.5))
                    new_ns = current_ns + stress * 0.4
                    tex_slot["normal_strength"] = self._clamp(new_ns, 0.0, 2.0)

                if "slot_6_physics_layer" in slots:
                    layer_slot = slots["slot_6_physics_layer"]
                    current_damp = self._safe_float(layer_slot.get("damping", 0.2))
                    new_damp = current_damp + stress * 0.5
                    layer_slot["damping"] = self._clamp(new_damp, 0.0, 1.0)

            # 4. 容积坍缩与自排斥堆叠
            if self._is_fabric(asset_data):
                zone = zone_map.get(asset_id, "default")
                has_skin = any(self._is_skin(assets[aid]) and zone_map.get(aid) == zone for aid in assets if isinstance(assets[aid], dict))
                if "slot_6_physics_layer" not in slots:
                    slots["slot_6_physics_layer"] = {}
                layer_slot = slots["slot_6_physics_layer"]
                if not has_skin:
                    layer_slot["radial_support"] = self.collapse_radial_support
                    if "gravity_multiplier" not in layer_slot:
                        layer_slot["gravity_multiplier"] = 1.0
                    else:
                        layer_slot["gravity_multiplier"] += 0.5 * dt
                layer_count = zone_layers.get(zone, 1)
                if layer_count > self.layer_threshold:
                    extra = (layer_count - self.layer_threshold) * self.repulsion_base
                    repulsion = self._clamp(extra, 0.0, 0.05)
                    layer_slot["repulsion_bias"] = repulsion
                    if "collision_offset" in layer_slot:
                        layer_slot["collision_offset"] += repulsion * 0.5 * dt
                else:
                    layer_slot.pop("repulsion_bias", None)

            if self._is_joint(asset_data):
                continue

            asset_data["slots"] = slots

        p3_package["assets"] = assets
        p3_package["stress_buffer_snapshot"] = self.stress_buffer.copy()
        return p3_package


# 使用示例
if __name__ == "__main__":
    mock_p3 = {
        "assets": {
            "torso_shirt": {
                "sovereignty": "fabric_factory",
                "slots": {
                    "slot_1_status": [1, 1, 0.3],
                    "slot_6_physics_layer": {"collision_offset": 0.01, "stiffness": 0.4},
                    "slot_5_texture": {"normal_strength": 0.6, "roughness": 0.5, "material_type": "silk"}
                }
            },
            "hand_right": {
                "sovereignty": "joint_factory",
                "motion_vector": [0.5, 0.2, 0.0],
                "slots": {}
            },
            "stiffness": 0.5,  # 模拟上游乱丢的脏数据
            "cfg_scale": 4.0   # 模拟上游乱丢的脏数据
        }
    }
    engine = P3SuperiorityEngine()
    grab_info = {"torso_shirt": ("hand_right", (0.2, 0.5, 0.0))}
    result = engine.process_frame(mock_p3, hand_grab_info=grab_info, dt=1/30)
    shirt = result["assets"]["torso_shirt"]["slots"]
    print("Stiffness:", shirt["slot_6_physics_layer"].get("stiffness"))
    print("Stress magnitude:", shirt["slot_6_physics_layer"].get("stress_magnitude"))