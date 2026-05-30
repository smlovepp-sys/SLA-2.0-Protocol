# p2_vfx_adapter.py（无损结构化资产构建与增量合并版）
import torch
from typing import Dict, Any

class P2VFXAdapter:
    def __init__(self, debug: bool = False, normalize_normals: bool = False):
        self.debug = debug
        self.normalize_normals = normalize_normals
        self.clamping_keywords = ["normal", "mask", "alpha", "image", "texture"]

    def _clean_tensors(self, data: Any, key_hint: str = "") -> Any:
        if isinstance(data, torch.Tensor):
            cleaned = data.detach().clone()
            key_lower = key_hint.lower()
            is_normal = "normal" in key_lower or "slot_3_normal" in key_lower
            if is_normal:
                cleaned = torch.clamp(cleaned, -1.0, 1.0)
                if self.normalize_normals and cleaned.dim() >= 3 and cleaned.shape[-3] == 3:
                    norm = torch.norm(cleaned, dim=-3, keepdim=True)
                    norm = torch.clamp(norm, min=1e-6)
                    cleaned = cleaned / norm
                return cleaned
            elif any(kw in key_lower for kw in self.clamping_keywords):
                return torch.clamp(cleaned, 0.0, 1.0)
            else:
                if self.debug and cleaned.numel() > 0 and (cleaned.max() > 1.0 or cleaned.min() < 0.0):
                    print(f"ℹ️ [P2 防护]: 物理/潜空间张量放行 -> 键: '{key_hint}', 范围: {cleaned.min().item():.3f} ~ {cleaned.max().item():.3f}")
                return cleaned
        elif isinstance(data, dict):
            return {k: self._clean_tensors(v, key_hint=k) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_tensors(item, key_hint=key_hint) for item in data]
        else:
            return data

    def _build_entity_assets(self, factory_token_map: Dict[str, list], global_physics: Dict[str, float] = None) -> Dict[str, dict]:
        assets = {}
        for fac_name, tokens in factory_token_map.items():
            for token in tokens:
                if isinstance(token, (list, tuple)) and len(token) == 2:
                    obj_name, params = token
                    if isinstance(obj_name, str) and isinstance(params, dict):
                        assets[obj_name] = {
                            "sovereignty": fac_name,
                            "slots": {
                                "slot_6_physics_layer": params.copy(),
                                "slot_1_status": [1, 1, 1.0]
                            }
                        }
        if global_physics:
            assets["global_physics"] = {
                "sovereignty": "global",
                "slots": {
                    "slot_6_physics_layer": global_physics.copy(),
                    "slot_1_status": [1, 1, 1.0]
                }
            }
        return assets

    def adapt(self, p1_package: Dict[str, Any]) -> Dict[str, Any]:
        if self.debug:
            print("[P2VFXAdapter] 开始执行智能特征筛查与安全钳位...")
        p2_package = self._clean_tensors(p1_package)

        # 1. 安全提取经过张量清洗后的原始 assets 字典
        orig_assets = p2_package.get("assets", {})
        
        if isinstance(orig_assets, dict) and "factory_token_map" in orig_assets:
            if self.debug:
                print("[P2VFXAdapter] 检测到结构化令牌，启动解耦实体构建...")
                
            factory_token_map = orig_assets.get("factory_token_map", {})
            global_physics = orig_assets.get("global_physics", {})
            
            # 2. 从令牌地图构建基础物理资产
            token_built_assets = self._build_entity_assets(factory_token_map, global_physics)
            
            # 3. 🛡️ 非破坏性增量合并
            # 保留原 assets 中由 P1 显式挂载的所有高级细分实体，剔除控制元数据键
            merged_assets = {k: v for k, v in orig_assets.items() if k not in ["factory_token_map", "global_physics"]}
            
            # 增量注入：如果原实体与令牌同名，则安全融合物理槽位，否则全量保留
            for k, v in token_built_assets.items():
                if k in merged_assets and isinstance(merged_assets[k], dict):
                    merged_assets[k].setdefault("slots", {}).update(v["slots"])
                else:
                    merged_assets[k] = v
            
            p2_package["assets"] = merged_assets
            
            if "global_ciphers" in orig_assets:
                p2_package["global_ciphers"] = orig_assets["global_ciphers"]
                
            if self.debug:
                print(f"[P2VFXAdapter] 实体资产无损合并完成，送入下游共 {len(merged_assets)} 个实体: {list(merged_assets.keys())}")

        return p2_package