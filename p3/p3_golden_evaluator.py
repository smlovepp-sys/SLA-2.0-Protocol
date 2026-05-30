# p3_golden_evaluator.py (修复：正确读取材料动力学)
import math
from typing import Dict, List, Any

class P3GoldenEvaluator:
    def __init__(self):
        pass

    def evaluate(self, snapshots: List[Dict[str, Any]],
                 active_ciphers: List[str],
                 anchor_mae: float,
                 slot_hit: bool,
                 total_frames: int,
                 user_steps: int) -> Dict[str, Any]:
        # 1. 难度系数 Cs
        diversity_score = self._calc_cipher_diversity(active_ciphers)
        max_stress = max(s.get('max_stress', 0.0) for s in snapshots)
        stress_norm = min(1.0, max_stress / 1.48)
        dynamics_score = self._calc_material_dynamics(snapshots)

        Cs = 0.4 * diversity_score + 0.3 * stress_norm + 0.3 * dynamics_score
        Cs = max(0.2, min(3.0, Cs))

        # 2. 控制度 Ca
        anchor_factor = 1.0 / (anchor_mae + 1e-6)
        anchor_factor = max(0.5, min(1.5, anchor_factor / 10.0))
        golden_steps_initial = 20 + Cs * 15
        eta = min(1.5, user_steps / golden_steps_initial) if golden_steps_initial > 0 else 1.0
        slot_factor = 1.1 if slot_hit else 1.0
        Ca = anchor_factor * eta * slot_factor
        Ca = max(0.1, min(2.0, Ca))

        # 3. 黄金步数/帧数
        golden_steps = int(20 + Cs * 15)
        golden_frames = int(total_frames * (1.0 + 0.3 * Cs))

        # 4. 强度缩放系数
        if user_steps <= golden_steps:
            scale_factor = user_steps / golden_steps
        else:
            scale_factor = 1.0 + 0.2 * math.log2(user_steps / golden_steps)
            scale_factor = min(1.5, scale_factor)

        return {
            'Cs': Cs,
            'Ca': Ca,
            'golden_steps': golden_steps,
            'golden_frames': golden_frames,
            'scale_factor': scale_factor,
            'complexity_score': Cs,
            'control_authority': Ca
        }

    def _calc_cipher_diversity(self, ciphers: List[str]) -> float:
        weights = {'ZZ': 1.0, 'H': 1.0, 'Fi': 1.5, 'Fl': 1.5, 'Fo': 1.5,
                   'Sf': 1.5, 'Sm': 1.5, 'Sc': 1.5, 'L': 2.5, 'Lh': 2.5,
                   'Ls': 2.5, 'Lm': 2.5, 'Lv': 2.5, 'Ft': 2.5}
        total = sum(weights.get(c, 1.0) for c in ciphers)
        return min(3.0, total / 5.0)

    def _calc_material_dynamics(self, snapshots: List[Dict[str, Any]]) -> float:
        if len(snapshots) < 2:
            return 0.0
        diffs = []
        for i in range(1, len(snapshots)):
            prev_dynamics = snapshots[i-1].get('material_dynamics', {})
            curr_dynamics = snapshots[i].get('material_dynamics', {})
            # 提取第一个资产的物理参数（兼容嵌套结构）
            prev_phys = self._extract_first_asset_physics(prev_dynamics)
            curr_phys = self._extract_first_asset_physics(curr_dynamics)
            if prev_phys and curr_phys:
                mae = 0.0
                count = 0
                for k in ['stiffness', 'wetness', 'tear_intensity']:
                    if k in prev_phys and k in curr_phys:
                        mae += abs(prev_phys[k] - curr_phys[k])
                        count += 1
                if count > 0:
                    diffs.append(mae / count)
        if not diffs:
            return 0.0
        avg_diff = sum(diffs) / len(diffs)
        return min(1.0, avg_diff * 5)

    def _extract_first_asset_physics(self, dynamics: Dict[str, Any]) -> Dict[str, Any]:
        """从 material_dynamics 中提取第一个资产的物理参数字典"""
        if not dynamics:
            return {}
        first_value = next(iter(dynamics.values()), None)
        if isinstance(first_value, dict):
            return first_value  # 嵌套情况，返回内层字典
        # 否则假设已经是扁平的物理字典
        return dynamics