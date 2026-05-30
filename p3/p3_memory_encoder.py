# p3_memory_encoder.py (修复：兼容元组格式的 stress_field)
import torch
import math
import hashlib
from typing import Dict, Any, List

class P3MemoryEncoder:
    def __init__(self, seed: int = 42):
        self.seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.proj_matrix = torch.randn(8, 128, generator=generator) * 0.1
        self._cipher_embed_cache: Dict[str, torch.Tensor] = {}

    def _get_cipher_embed(self, cipher: str) -> torch.Tensor:
        if cipher in self._cipher_embed_cache:
            return self._cipher_embed_cache[cipher]
        hash_bytes = hashlib.md5(cipher.encode()).digest()
        seed_val = int.from_bytes(hash_bytes[:4], 'big') % (2**31)
        gen = torch.Generator()
        gen.manual_seed(seed_val)
        embed = torch.randn(8, generator=gen) * 0.1
        self._cipher_embed_cache[cipher] = embed
        return embed

    def encode(self, snapshot: Dict[str, Any], active_ciphers: List[str]) -> torch.Tensor:
        vec = torch.zeros(256)

        # 0-47: material_dynamics 统计
        dynamics = snapshot.get('material_dynamics', {})
        if isinstance(dynamics, dict) and len(dynamics) > 0:
            stiff_vals, fric_vals, wet_vals, tear_vals = [], [], [], []
            for params in dynamics.values():
                if isinstance(params, dict):
                    if 'stiffness' in params: stiff_vals.append(float(params['stiffness']))
                    if 'friction' in params: fric_vals.append(float(params['friction']))
                    if 'wetness' in params: wet_vals.append(float(params['wetness']))
                    if 'tear_intensity' in params: tear_vals.append(float(params['tear_intensity']))
            if stiff_vals:
                vec[0] = sum(stiff_vals) / len(stiff_vals)
                vec[1] = max(stiff_vals)
            if fric_vals:
                vec[2] = sum(fric_vals) / len(fric_vals)
                vec[3] = max(fric_vals)
            if wet_vals:
                vec[4] = sum(wet_vals) / len(wet_vals)
                vec[5] = max(wet_vals)
            if tear_vals:
                vec[6] = sum(tear_vals) / len(tear_vals)
                vec[7] = max(tear_vals)

        # 48-63: 应力场统计（兼容元组格式）
        stress_field = snapshot.get('stress_field', {})
        if isinstance(stress_field, dict) and len(stress_field) > 0:
            try:
                values = []
                for v in stress_field.values():
                    if isinstance(v, (tuple, list)) and len(v) >= 2:
                        # 新格式: (mag, (dx, dy))
                        mag = v[0]
                        if isinstance(mag, (int, float)):
                            values.append(float(mag))
                    elif isinstance(v, (int, float)):
                        values.append(float(v))
                    elif isinstance(v, torch.Tensor) and v.numel() == 1:
                        values.append(v.item())
                if values:
                    t = torch.tensor(values)
                    vec[48] = t.mean()
                    vec[49] = t.var(unbiased=False)
                    vec[50] = t.max()
                    vec[51] = self._skewness(t)
            except Exception as e:
                print(f"[MemoryEncoder] 应力场处理警告: {e}")

        # 64-79: 折痕深度统计
        crease_field = snapshot.get('crease_field', {})
        if isinstance(crease_field, dict) and len(crease_field) > 0:
            try:
                depths = []
                for v in crease_field.values():
                    if isinstance(v, dict) and 'depth' in v:
                        depths.append(float(v['depth']))
                    elif isinstance(v, (int, float)):
                        depths.append(float(v))
                if depths:
                    t = torch.tensor(depths)
                    vec[64] = t.mean()
                    vec[65] = t.max()
            except Exception as e:
                print(f"[MemoryEncoder] 折痕场处理警告: {e}")

        # 80-95: 撕裂强度统计
        tear_field = snapshot.get('tear_field', {})
        if isinstance(tear_field, dict) and len(tear_field) > 0:
            try:
                strengths = []
                for v in tear_field.values():
                    if isinstance(v, dict) and 'strength' in v:
                        strengths.append(float(v['strength']))
                    elif isinstance(v, (int, float)):
                        strengths.append(float(v))
                if strengths:
                    t = torch.tensor(strengths)
                    vec[80] = t.mean()
                    vec[81] = t.max()
            except Exception as e:
                print(f"[MemoryEncoder] 撕裂场处理警告: {e}")

        # 128-255: 主权材质嵌入
        if active_ciphers:
            embeds = [self._get_cipher_embed(c) for c in active_ciphers[:4]]
            combined = torch.cat(embeds)[:8]
            proj = combined @ self.proj_matrix
            vec[128:256] = proj

        return vec

    def _skewness(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean()
        std = x.std(unbiased=False)
        if std == 0:
            return torch.tensor(0.0)
        return ((x - mean) ** 3).mean() / (std ** 3)