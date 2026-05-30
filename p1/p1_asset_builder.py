# p1_asset_builder.py（集成冲突裁决、动态演化率、自适应健康检查框架）
import torch
import math
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# 物理参数合法范围（用于合理性校验）
PARAM_BOUNDS = {
    "stiffness": (0.0, 1.0),
    "friction": (0.0, 1.0),
    "elasticity": (0.0, 1.0),
    "wetness": (0.0, 1.0),
    "tear_intensity": (0.0, 1.0),
    "tension": (-1.0, 1.0),
    "body_type": (0.0, 1.0),
}


def _harmonize_params(keyword_params: Dict[str, float]) -> Dict[str, float]:
    """
    冲突参数裁决层：
    - 处理 stiffnes + elasticity 冲突，采用能量上限归一化
    - 可扩展更多互斥参数对
    """
    if not keyword_params:
        return keyword_params
    # 深拷贝避免意外修改原始数据
    params = dict(keyword_params)
    if "stiffness" in params and "elasticity" in params:
        total = params["stiffness"] + params["elasticity"]
        if total > 1.2:
            factor = 1.2 / total
            params["stiffness"] *= factor
            params["elasticity"] *= factor
            logger.debug(f"参数冲突调和: stiffness={params['stiffness']:.3f}, elasticity={params['elasticity']:.3f}")
    # 可在此添加更多冲突规则，如 friction 与 wetness 等
    return params


def _evaluate_channel_diversity(components: List[Dict[str, Any]]) -> Dict[str, Any]:
    """评估各工厂输出张量的通道差异度（通道间标准差的均值）"""
    scores = {}
    for comp in components:
        tensor = comp.get("tensor")
        factory = comp.get("factory", "unknown")
        if tensor is None or tensor.numel() == 0:
            scores[factory] = {"channel_std_mean": 0.0, "score": 0.0}
            continue
        # 确保浮点类型
        t = tensor.float()
        # 通道维标准差 (假设 tensor shape: C, H, W 或 C, ...)
        if t.dim() >= 2:
            channel_std = t.std(dim=tuple(range(1, t.dim()))).mean().item()
        else:
            channel_std = 0.0
        score = min(1.0, channel_std / 0.1)  # 0.1 为满分阈值
        scores[factory] = {"channel_std_mean": channel_std, "score": score}
    return scores


def _check_numerical_health(tensor: torch.Tensor) -> Dict[str, Any]:
    """检查张量中 NaN / Inf 比例，返回健康评分"""
    total = tensor.numel()
    if total == 0:
        return {"has_nan": True, "has_inf": True, "nan_ratio": 1.0, "inf_ratio": 1.0, "score": 0.0}
    # 确保浮点张量，避免整数型无 NaN
    if not tensor.is_floating_point():
        t = tensor.float()
    else:
        t = tensor
    nan_ratio = torch.isnan(t).sum().item() / total
    inf_ratio = torch.isinf(t).sum().item() / total
    score = 1.0 - min(1.0, (nan_ratio + inf_ratio) * 10)  # 放大惩罚
    return {"nan_ratio": nan_ratio, "inf_ratio": inf_ratio, "score": score}


def _check_param_reasonability(keyword_params: Dict[str, float]) -> Dict[str, Any]:
    """检查关键词参数是否在物理常识范围内，并检查组合合理性"""
    if not keyword_params:
        return {"all_valid": True, "violations": [], "score": 1.0}

    violations = []
    for param, value in keyword_params.items():
        if param in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[param]
            if not (lo <= value <= hi):
                violations.append((param, value, lo, hi))

    # 组合规则示例：极硬且极滑的组合不合理
    combo_penalty = 0
    stiffness = keyword_params.get("stiffness", 0.5)
    friction = keyword_params.get("friction", 0.5)
    if stiffness > 0.9 and friction < 0.1:
        violations.append(("combo_stiff_friction", f"stiffness={stiffness}, friction={friction}", "", ""))
        combo_penalty += 1

    score = 1.0 - len(violations) * 0.2  # 每个违规减 0.2
    score = max(0.0, score)
    return {"all_valid": len(violations) == 0, "violations": violations, "score": score}


def _check_coverage(components: List[Dict[str, Any]]) -> Dict[str, Any]:
    """检查是否所有工厂都产出了非零张量"""
    total = len(components)
    active = 0
    details = {}
    for comp in components:
        tensor = comp.get("tensor")
        factory = comp.get("factory", "unknown")
        if tensor is not None and tensor.float().abs().max().item() > 0.0:
            active += 1
            details[factory] = "active"
        else:
            details[factory] = "inactive"
    score = active / max(total, 1)
    return {"active_count": active, "total_count": total, "details": details, "score": score}


def build_assets(components: List[Dict[str, Any]],
                 factory_tokens: Dict[str, List],
                 sovereignty_groups: Dict[str, List],
                 evolution_rate: float,
                 keyword_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    构造 P3 标准资产包，并附加 P1 质量自评报告。
    改进点：
    - 调用参数调和层处理冲突
    - 演化率根据湿度动态调整
    - 使用日志输出质量报告
    """
    # -------- 参数冲突裁决 ----------
    harmonized_params = _harmonize_params(keyword_params or {})

    # -------- P1 质量自评 ----------
    channel_scores = _evaluate_channel_diversity(components)

    worst_num_score = 1.0
    for comp in components:
        tensor = comp.get("tensor")
        if tensor is not None:
            num_health = _check_numerical_health(tensor)
            worst_num_score = min(worst_num_score, num_health["score"])

    param_check = _check_param_reasonability(harmonized_params)
    coverage = _check_coverage(components)

    channel_avg = sum(v["score"] for v in channel_scores.values()) / max(len(channel_scores), 1)
    total_score = (channel_avg * 0.35 +
                   worst_num_score * 0.25 +
                   param_check["score"] * 0.25 +
                   coverage["score"] * 0.15)

    quality_report = {
        "total_score": round(total_score, 4),
        "grade": ("A" if total_score >= 0.8 else
                  "B" if total_score >= 0.6 else
                  "C" if total_score >= 0.4 else
                  "D" if total_score >= 0.2 else "F"),
        "channel_diversity": channel_scores,
        "numerical_health": worst_num_score,
        "param_reasonability": param_check,
        "coverage": coverage,
    }

    logger.info(f"[P1 质量自评] 总分: {total_score:.3f} ({quality_report['grade']})")
    logger.info(f"  通道差异度: {channel_avg:.3f} | 数值健康: {worst_num_score:.3f} | 参数合理: {param_check['score']:.3f} | 覆盖度: {coverage['score']:.3f}")
    if param_check["violations"]:
        logger.warning(f"  ⚠️ 参数越界/冲突: {param_check['violations']}")

    # -------- 资产构造（应用调和后参数及动态演化率） --------
    assets = {}
    for comp in components:
        factory = comp.get("factory")
        if not factory:
            continue
        cid = comp.get("component_id", factory[:4])
        asset_id = f"{factory}_{cid}"[:16]
        tokens = factory_tokens.get(factory, [])
        total_weight = sum(abs(w) for _, w in tokens) if tokens else 0.0

        # 从调和后的参数中提取物理值
        stiffness = harmonized_params.get("stiffness", 0.5)
        friction = harmonized_params.get("friction", 0.5)
        wetness = harmonized_params.get("wetness", 0.0)
        tear = harmonized_params.get("tear_intensity", 0.0)

        # 动态演化率：湿度越高，演化越快
        dynamic_evolution = evolution_rate * (1.0 + wetness * 0.5)

        slots = {
            "slot_1_status": [1, 1, dynamic_evolution],
            "slot_5_texture": {"material_type": _guess_material(factory, tokens)},
            "slot_6_physics_layer": {
                "depth_rank": 0,
                "collision_offset": 0.01 * total_weight,
                "stiffness": stiffness,
                "friction": friction,
                "wetness": wetness,
                "tear_intensity": tear,
            }
        }
        assets[asset_id] = {
            "sovereignty": factory,  # 可后续利用 sovereignty_groups 分配组 ID
            "factory": factory,
            "slots": slots,
        }

    assets["p1_quality_report"] = quality_report
    return assets


def _guess_material(factory: str, tokens: List) -> str:
    """根据工厂名猜测材质类型"""
    if "fabric" in factory:
        return "cotton"
    if "fluid" in factory:
        return "fluid"
    if "skin" in factory:
        return "skin"
    return "default"