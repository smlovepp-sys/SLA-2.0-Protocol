# physics_extractor.py（增强版：高分辨力物理轴 + 冠词清理 + 句子边界分割 + 短语级提取）
import os
import re
import torch
from typing import Dict, Optional, Any, List


class PhysicsExtractor:
    """
    物理感知与提取器：
    - 负责加载、验证、自愈物理轴矩阵（physics_axes.pt）
    - 从文本提示词中提取物理参数（通过 token embedding 投影）
    - 支持句子级、token级、短语级提取
    - 兼容各类 T5/CLIP 模型，自动适配 embedding 权重矩阵
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

    # 多组描述性短语，增强物理轴的方向分辨力
    AXIS_PAIRS = {
        "stiffness": [
            ("hard rigid steel unyielding structure",
             "soft plush flexible flowing material")
        ],
        "friction": [
            ("rough coarse sandpaper gripping surface",
             "smooth polished ice slippery surface")
        ],
        "elasticity": [
            ("bouncing rubber elastic returning shape",
             "brittle shattered glass permanent deformation")
        ],
        "wetness": [
            ("water flowing dripping soaking wet liquid",
             "dry arid desiccated powdery solid")
        ],
        "tear_intensity": [
            ("torn ripped fragile easily separated pieces",
             "intact durable strong unbreakable whole")
        ],
        "tension": [
            ("stretched tight pulled internal stress compressed",
             "slack loose relaxed no internal force expanded")
        ],
        "body_type": [
            ("tight firm solid compact body",
             "loose soft spread relaxed body")
        ]
    }

    def __init__(self, axis_cache_dir: Optional[str] = None):
        if axis_cache_dir is None:
            axis_cache_dir = os.path.dirname(os.path.abspath(__file__))
        self.axis_cache_dir = axis_cache_dir
        self.axis_path = os.path.join(axis_cache_dir, "physics_axes.pt")
        self.physics_axes: Optional[torch.Tensor] = None
        self.expected_dim = len(self.PARAM_BOUNDS)  # 7

    # -----------------------------------------------------------------
    # 物理轴管理
    # -----------------------------------------------------------------
    def load_or_build_axes(self, clip: Any) -> torch.Tensor:
        """
        安全加载物理轴矩阵。若缺失、格式错误或维度不匹配，则利用 clip 自动重建。
        """
        need_rebuild = False
        if not os.path.exists(self.axis_path):
            print(f"[PhysicsExtractor] 物理轴文件不存在: {self.axis_path}")
            need_rebuild = True
        else:
            try:
                axes = torch.load(self.axis_path, map_location="cpu")
                if not isinstance(axes, torch.Tensor):
                    print("[PhysicsExtractor] 物理轴文件格式错误（非 Tensor）")
                    need_rebuild = True
                elif axes.shape[0] != self.expected_dim:
                    print(f"[PhysicsExtractor] 物理轴维度不匹配（期望 {self.expected_dim}，实际 {axes.shape[0]}）")
                    need_rebuild = True
                else:
                    self.physics_axes = axes
                    print(f"[PhysicsExtractor] 物理轴加载成功: {self.axis_path}")
                    return axes
            except Exception as e:
                print(f"[PhysicsExtractor] 加载物理轴失败: {e}")
                need_rebuild = True

        if need_rebuild:
            print("[PhysicsExtractor] 正在利用 CLIP 模型重新构建物理轴...")
            axes = self._build_axes_from_clip(clip)
            torch.save(axes.cpu(), self.axis_path)
            print(f"[PhysicsExtractor] 物理轴重建完成，已保存至: {self.axis_path}")
            self.physics_axes = axes
            return axes

    def _get_embedding_weight(self, model: Any):
        """从模型对象中获取 token embedding 权重矩阵，兼容多种 CLIP/T5 模型。"""
        for attr in ["shared", "transformer", "model"]:
            mod = getattr(model, attr, None)
            if mod is not None and hasattr(mod, "weight"):
                return mod.weight
        if hasattr(model, "get_input_embeddings"):
            return model.get_input_embeddings().weight
        raise AttributeError("无法从模型中获取 token embedding 权重矩阵")

    def _build_axes_from_clip(self, clip: Any) -> torch.Tensor:
        """根据 CLIP 模型的 token embedding 构建高分辨力物理轴（多组短语平均）"""
        model = clip.model
        tokenizer = clip.tokenizer
        w = self._get_embedding_weight(model)
        device = next(model.parameters()).device

        axes_list = []
        for attr in self.PARAM_BOUNDS.keys():
            if attr in self.AXIS_PAIRS:
                desc_pairs = self.AXIS_PAIRS[attr]  # list of (pos_desc, neg_desc)
                dirs = []
                for pos_desc, neg_desc in desc_pairs:
                    pos_ids = tokenizer.encode(pos_desc, add_special_tokens=False)
                    neg_ids = tokenizer.encode(neg_desc, add_special_tokens=False)
                    if not pos_ids or not neg_ids:
                        continue
                    pos_vec = w[torch.tensor(pos_ids, device=device)].mean(dim=0)
                    neg_vec = w[torch.tensor(neg_ids, device=device)].mean(dim=0)
                    diff = pos_vec - neg_vec
                    dirs.append(diff / (diff.norm() + 1e-6))
                if dirs:
                    # 多组方向取平均，增强鲁棒性
                    axis = torch.stack(dirs).mean(dim=0)
                    axis = axis / (axis.norm() + 1e-6)
                else:
                    axis = torch.randn(w.shape[1], device=device)
            else:
                axis = torch.randn(w.shape[1], device=device)
            axes_list.append(axis)

        axes = torch.stack(axes_list, dim=0)
        return axes

    # -----------------------------------------------------------------
    # 句子级物理参数提取
    # -----------------------------------------------------------------
    def extract_from_text(self, prompt: str, clip: Any) -> Dict[str, float]:
        """对整个句子进行物理参数提取（平均 token embedding 投影）"""
        if self.physics_axes is None:
            self.load_or_build_axes(clip)
        if self.physics_axes is None:
            raise RuntimeError("物理轴未能就绪，无法提取物理参数。")
        return self._extract_single_phrase(prompt, clip)

    def _extract_single_phrase(self, phrase: str, clip: Any) -> Dict[str, float]:
        """核心投影逻辑：对单个短语提取物理参数"""
        model = clip.model
        tokenizer = clip.tokenizer
        w = self._get_embedding_weight(model)
        device = next(model.parameters()).device
        axes = self.physics_axes.to(device)

        token_ids = tokenizer.encode(phrase, add_special_tokens=False)
        if not token_ids:
            token_ids = [0]
        tokens_tensor = torch.tensor(token_ids, device=device)
        embeddings = w[tokens_tensor]          # [N, D]
        vec = embeddings.mean(dim=0)           # [D]
        vec_norm = vec / (vec.norm() + 1e-6)   # 归一化

        raw_scores = torch.matmul(axes, vec_norm)   # [7]
        phys_values = (raw_scores * 0.5 + 0.5).clamp(0.0, 1.0)
        tension_idx = list(self.PARAM_BOUNDS.keys()).index("tension")
        phys_values[tension_idx] = raw_scores[tension_idx].clamp(-1.0, 1.0)

        keys = list(self.PARAM_BOUNDS.keys())
        return {keys[i]: round(phys_values[i].item(), 4) for i in range(len(keys))}

    # -----------------------------------------------------------------
    # Token 级提取（每个 token 独立）
    # -----------------------------------------------------------------
    def extract_multi_token(self, prompt: str, clip: Any) -> Dict[str, Dict[str, float]]:
        """按 token 粒度提取物理参数，返回 {token_str: params_dict}"""
        if self.physics_axes is None:
            self.load_or_build_axes(clip)
        if self.physics_axes is None:
            raise RuntimeError("物理轴未能就绪。")

        model = clip.model
        tokenizer = clip.tokenizer
        w = self._get_embedding_weight(model)
        device = next(model.parameters()).device
        axes = self.physics_axes.to(device)

        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            token_ids = [0]

        tokens_tensor = torch.tensor(token_ids, device=device)
        embeddings = w[tokens_tensor]          # [N, D]
        embeddings_norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-6)

        raw_scores = torch.matmul(axes, embeddings_norm.T)  # [7, N]

        keys = list(self.PARAM_BOUNDS.keys())
        tension_idx = keys.index("tension")

        result = {}
        for i, tok_id in enumerate(token_ids):
            tok_str = tokenizer.decode([tok_id])
            vals = {}
            for j, key in enumerate(keys):
                score = raw_scores[j, i]
                if key == "tension":
                    val = score.clamp(-1.0, 1.0).item()
                else:
                    val = (score * 0.5 + 0.5).clamp(0.0, 1.0).item()
                vals[key] = round(val, 4)
            result[tok_str] = vals
        return result

    # -----------------------------------------------------------------
    # 短语级提取（用于物体级分流）—— 改进句子边界与停用短语过滤
    # -----------------------------------------------------------------
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """
        简单的名词短语提取，去除权重标记，按逗号、介词、句子边界分割，并清理冠词和停用结构。
        """
        # 去除权重标记如 (word:1.2)
        clean = re.sub(r'\(.*?:\d+\.?\d*\)', '', text)
        clean = re.sub(r'[()]', '', clean).strip().lower()

        # 将句子边界符号（句号、感叹号、分号）统一替换为逗号，防止粘合
        clean = re.sub(r'[.!;]+', ',', clean)

        # 按逗号、常见介词以及连接词分割
        parts = re.split(r',|\bin\b|\bwith\b|\bwearing\b|\bon\b|\band\b', clean)
        phrases = []
        for p in parts:
            p = p.strip()
            if not p:
                continue

            # 强力清除头部冠词污染
            p = re.sub(r'^(a|an|the)\s+', '', p, flags=re.IGNORECASE).strip()

            # 过滤纯语法停用短语（无实际物理意义的代词/动词组合）
            if re.fullmatch(r'(she|he|it|they|we|you)\s+(is|are|was|were|has|have|had|been|being)\b.*', p):
                continue

            words = p.split()
            # 跳过纯连接词/无物理语义
            if all(w in {'is', 'are', 'was', 'were', 'on', 'in', 'at', 'with', 'of', 'to', 'and', 'or'} for w in words):
                continue
            if p:
                phrases.append(p)

        return phrases if phrases else [text.strip()]

    def extract_phrases_from_text(self, prompt: str, clip: Any) -> Dict[str, Dict[str, float]]:
        """
        对提示词进行短语分割，为每个短语独立提取物理参数。
        返回 {phrase: {param: value}, ...}
        """
        if self.physics_axes is None:
            self.load_or_build_axes(clip)
        if self.physics_axes is None:
            raise RuntimeError("物理轴未能就绪")

        phrases = self._extract_noun_phrases(prompt)
        result = {}
        for phrase in phrases:
            params = self._extract_single_phrase(phrase, clip)
            if params:
                result[phrase] = params
        return result

    # -----------------------------------------------------------------
    # 便捷方法
    # -----------------------------------------------------------------
    def get_physics_axes(self) -> Optional[torch.Tensor]:
        return self.physics_axes