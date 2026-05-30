# p3_cipher_engine.py
# P3 暗号引擎：基于 P1 暗号协议的快速物理参数映射器
# 实现 O(1) 复杂度的属性查询与冲突解决

from typing import Dict, List, Any, Tuple

class P3CipherEngine:
    """
    暗号物理映射引擎
    - 内置 CIPHER_MAP，将暗号映射为物理属性
    - 提供空间主权校准：给定坐标的暗号列表，输出合并后的物理约束包
    - 冲突解决：按硬度主权（stiffness）竞争，高 stiffness 主导核心参数
    """

    CIPHER_MAP: Dict[str, Dict[str, float]] = {
        "Ah": {"stiffness": 1.0, "friction": 1.0, "rank": 100, "evolution_limit": 0.0, "depth_rank": 0},
        "Ft": {"stiffness": -0.5, "friction": 0.2, "rank": 90, "evolution_limit": 1.0, "depth_rank": 0},
        "L":  {"stiffness": 0.0, "friction": 0.3, "rank": 80, "evolution_limit": 1.0, "depth_rank": 0},
        "Lh": {"stiffness": 0.0, "friction": 0.1, "rank": 80, "evolution_limit": 1.0, "depth_rank": 0, "wetness": 0.8},
        "Ls": {"stiffness": 0.0, "friction": 0.2, "rank": 80, "evolution_limit": 1.0, "depth_rank": 0, "splash": 0.7},
        "Lm": {"stiffness": 0.0, "friction": 0.05, "rank": 80, "evolution_limit": 0.5, "depth_rank": 0, "mist": 0.9},
        "Lv": {"stiffness": 0.0, "friction": 0.02, "rank": 80, "evolution_limit": 0.3, "depth_rank": 0, "vapor": 1.0},
        "Fi": {"stiffness": 0.2, "friction": 0.8, "rank": 10, "evolution_limit": 0.2, "depth_rank": 1},
        "Fl": {"stiffness": 0.4, "friction": 0.6, "rank": 20, "evolution_limit": 0.4, "depth_rank": 2},
        "Fo": {"stiffness": 0.6, "friction": 0.5, "rank": 30, "evolution_limit": 0.6, "depth_rank": 3},
        "Sf": {"stiffness": 0.3, "friction": 0.7, "rank": 5, "evolution_limit": 0.3, "depth_rank": 0, "gender": "female"},
        "Sm": {"stiffness": 0.35, "friction": 0.7, "rank": 5, "evolution_limit": 0.3, "depth_rank": 0, "gender": "male"},
        "Sc": {"stiffness": 0.25, "friction": 0.65, "rank": 5, "evolution_limit": 0.3, "depth_rank": 0, "age": "child_elder"},
        "H":  {"stiffness": 1.0, "friction": 0.4, "rank": 70, "evolution_limit": 0.0, "depth_rank": 0},
        "ZZ": {"stiffness": 0.0, "friction": 0.0, "rank": 0, "evolution_limit": 0.0, "depth_rank": 0},
    }

    def __init__(self):
        pass

    def get_physics_params(self, cipher: str) -> Dict[str, float]:
        return self.CIPHER_MAP.get(cipher, self.CIPHER_MAP["ZZ"]).copy()

    def resolve_tile_physics(self, ciphers_at_coord: List[str]) -> Dict[str, float]:
        if not ciphers_at_coord:
            return self.CIPHER_MAP["ZZ"].copy()

        # 按 stiffness 降序排序，取最高作为主权材质
        sorted_by_stiffness = sorted(
            ciphers_at_coord,
            key=lambda c: self.CIPHER_MAP.get(c, {}).get("stiffness", -1),
            reverse=True
        )
        primary_cipher = sorted_by_stiffness[0]
        base_params = self.get_physics_params(primary_cipher)

        # 处理其他暗号的副作用
        for cipher in sorted_by_stiffness[1:]:
            other = self.get_physics_params(cipher)

            if cipher == "Ft":   # 破损：降低刚度，增加摩擦
                base_params["stiffness"] = max(0.0, base_params["stiffness"] + other.get("stiffness", 0.0))
                base_params["friction"] = min(1.0, base_params.get("friction", 0.0) + 0.2)

            elif cipher.startswith("L"):   # 流体：增加湿润、溅射等
                if "wetness" in other:
                    base_params["wetness"] = base_params.get("wetness", 0.0) + other["wetness"]
                for attr in ["splash", "mist", "vapor"]:
                    if attr in other:
                        base_params[attr] = base_params.get(attr, 0.0) + other[attr]
                base_params["lubrication"] = base_params.get("lubrication", 0.0) + other.get("friction", 0.0)

            elif cipher in ["Sf", "Sm", "Sc"]:   # 皮肤：增加湿润度
                base_params["skin_wetness"] = base_params.get("skin_wetness", 0.0) + 0.2

            # 其他材质：不覆盖核心参数

        # 钳制数值范围
        base_params["stiffness"] = max(0.0, min(1.0, base_params.get("stiffness", 0.0)))
        base_params["friction"] = max(0.0, min(1.0, base_params.get("friction", 0.0)))
        return base_params

    def calibrate_tile(self, tx: int, ty: int, ciphers_at_coord: List[str]) -> Dict[str, float]:
        """空间主权校准接口：根据潜空间坐标和暗号列表输出物理约束包"""
        return self.resolve_tile_physics(ciphers_at_coord)


# 使用示例
if __name__ == "__main__":
    engine = P3CipherEngine()
    params = engine.calibrate_tile(5, 5, ["Sf", "Lh"])
    print("合并后物理参数:", params)
    params2 = engine.calibrate_tile(2, 3, ["Fi", "Ft"])
    print("破损织物:", params2)