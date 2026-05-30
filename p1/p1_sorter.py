# p1_sorter.py（终极版 + 扩展关键词库）
from typing import Dict, List, Any

class P1Sorter:
    """
    竞争制分发器 + 物理常识增强 + 数学模型优化
    - 布料厂得分 = 0.5 - stiffness（柔软度）
    - 关节厂张力引入 0.2 消噪阈值
    - 常识先验权重 0.5，确保明确语义绝对对齐
    - 关键词库扩展以覆盖更多材质与环境描述
    """

    PARAM_BOUNDS = {
        "stiffness": (0.0, 1.0),
        "friction": (0.0, 1.0),
        "elasticity": (0.0, 1.0),
        "wetness": (0.0, 1.0),
        "tear_intensity": (0.0, 1.0),
        "tension": (-1.0, 1.0),
        "body_type": (0.0, 1.0),
    }

    # 物理常识微调表
    MATERIAL_TUNING = {
        "silk": {"friction": -0.25, "stiffness": -0.15, "wetness": -0.1},
        "satin": {"friction": -0.25, "stiffness": -0.15},
        "velvet": {"friction": 0.2, "stiffness": -0.1},
        "cotton": {"friction": 0.1, "stiffness": -0.05},
        "leather": {"friction": 0.3, "stiffness": 0.2, "elasticity": -0.1},
        "denim": {"friction": 0.35, "stiffness": 0.25},
        "lace": {"stiffness": -0.2, "friction": -0.1},
        "rain": {"wetness": 0.4, "friction": -0.1},
        "water": {"wetness": 0.45, "friction": -0.15},
        "sweat": {"wetness": 0.35},
        "fluid": {"wetness": 0.4, "friction": -0.1},
        "liquid": {"wetness": 0.4, "friction": -0.1},
        "metal": {"stiffness": 0.35, "friction": 0.3, "elasticity": -0.2},
        "steel": {"stiffness": 0.4, "friction": 0.35, "elasticity": -0.25},
        "glass": {"stiffness": 0.3, "friction": 0.1, "elasticity": -0.3},
        "wood": {"stiffness": 0.25, "friction": 0.2},
        "stone": {"stiffness": 0.4, "friction": 0.4, "elasticity": -0.2},
        "rubber": {"elasticity": 0.3, "friction": 0.2},
        "skin": {"body_type": 0.15, "elasticity": 0.1},
        "body": {"body_type": 0.15},
        "girl": {"body_type": 0.1},
        "woman": {"body_type": 0.15},
        "man": {"body_type": 0.1},
        "joint": {"stiffness": 0.1, "tension": 0.1},
        "bone": {"stiffness": 0.2, "tension": 0.1},
    }

    def __init__(self):
        print("[P1Sorter] 物理驱动分发器（终极版+扩展关键词）已就绪。")

    def _clamp_value(self, param_name: str, value: float) -> float:
        lo, hi = self.PARAM_BOUNDS.get(param_name, (0.0, 1.0))
        return max(lo, min(hi, float(value)))

    def _apply_material_tuning(self, token: str, physics: Dict[str, float]) -> Dict[str, float]:
        tuned = physics.copy()
        token_lower = token.lower()
        for keyword, adjustments in self.MATERIAL_TUNING.items():
            if keyword in token_lower:
                for param, delta in adjustments.items():
                    if param in tuned:
                        tuned[param] = self._clamp_value(param, tuned[param] + delta)
        return tuned

    def distribute_from_t5(self, t5_extracted_payload: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        factory_token_map = {
            "fabric_factory": [],
            "fluid_factory": [],
            "joint_factory": [],
            "skin_factory": [],
        }

        param_sums: Dict[str, float] = {}
        param_counts: Dict[str, int] = {}

        for token, physics in t5_extracted_payload.items():
            if not isinstance(physics, dict):
                continue

            # 1. 钳位
            clamped = {}
            for pname, value in physics.items():
                if pname in self.PARAM_BOUNDS:
                    clamped[pname] = self._clamp_value(pname, value)

            # 2. 物理常识微调
            tuned = self._apply_material_tuning(token, clamped)

            # 3. 更新统计
            for pname, val in tuned.items():
                param_sums[pname] = param_sums.get(pname, 0.0) + val
                param_counts[pname] = param_counts.get(pname, 0) + 1

            # ---------------- 竞争制分发 ----------------
            wetness = tuned.get("wetness", 0.5)
            stiffness = tuned.get("stiffness", 0.5)
            tension = tuned.get("tension", 0.0)
            body_type = tuned.get("body_type", 0.5)

            factory_scores = {
                "fluid_factory": wetness - 0.5,
                "joint_factory": max(stiffness - 0.5, abs(tension) - 0.2),
                "skin_factory": body_type - 0.5,
                "fabric_factory": 0.5 - stiffness
            }

            # 强效常识先验增强（0.5）
            token_lower = token.lower()

            # 流体环境扩展（monsoon, downpour, storm, flood, mist, vapor, splash等）
            if any(w in token_lower for w in [
                "rain", "water", "fluid", "liquid", "sweat", "tear", "sea", "river",
                "monsoon", "downpour", "storm", "flood", "mist", "vapor", "splash", "damp"
            ]):
                factory_scores["fluid_factory"] += 0.5

            # 织物/穿戴物扩展（coat, armor, gauntlet, corset, trench, leather, steel, jacket等）
            if any(w in token_lower for w in [
                "dress", "silk", "cloth", "fabric", "skirt", "pants", "coat", "wear",
                "clothing", "corset", "armor", "gauntlet", "trench", "leather", "steel",
                "jacket", "robe", "gown", "shirt", "scarf", "glove", "boot", "heel",
                "stocking", "sock"
            ]):
                factory_scores["fabric_factory"] += 0.5

            # 人体/皮肤扩展（shoulder, muscle, flesh, chest, arm, leg等）
            if any(w in token_lower for w in [
                "girl", "man", "woman", "human", "skin", "face", "body", "player",
                "shoulder", "arm", "leg", "chest", "muscle", "flesh", "head", "hand"
            ]):
                factory_scores["skin_factory"] += 0.5

            # 刚体/机械扩展（不变）
            if any(w in token_lower for w in [
                "joint", "arm", "leg", "robot", "mechanical", "bone", "rigid",
                "steel", "metal", "hydraulic", "gear", "piston"
            ]):
                factory_scores["joint_factory"] += 0.5

            factory = max(factory_scores, key=factory_scores.get)
            factory_token_map[factory].append((token, tuned))

        # 4. 全局参数
        global_params = {}
        for pname in self.PARAM_BOUNDS:
            cnt = param_counts.get(pname, 0)
            if cnt > 0:
                global_params[pname] = round(param_sums[pname] / cnt, 4)
            else:
                global_params[pname] = 0.0 if pname == "tension" else 0.5

        print(f"[P1Sorter] 竞争制分发完成，共处理 {len(t5_extracted_payload)} 个 token")
        for fac, tokens in factory_token_map.items():
            for tok, params in tokens:
                print(f"  → {tok} -> {fac}: {params}")
        print(f"[P1Sorter] 全局物理参数: {global_params}")

        return {
            "factory_token_map": factory_token_map,
            "global_params": global_params,
        }