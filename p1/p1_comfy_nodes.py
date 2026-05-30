# p1_comfy_nodes.py（结构化资产传递修复版 + 精简日志）
import torch
import torch.nn as nn
import os
from typing import List, Optional, Dict, Any
from .p1_master_manager import P1MasterManager
from .p1_sorter import P1Sorter
from .physics_extractor import PhysicsExtractor
from transformers import T5EncoderModel, T5Tokenizer


class NativeCLIPLoader:
    """加载本地 T5 模型并预生成物理轴，为 P1 桥接节点提供 CLIP 对象"""
    @classmethod
    def INPUT_TYPES(cls):
        clip_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "clip")
        model_list = []
        if os.path.exists(clip_dir):
            for d in os.listdir(clip_dir):
                full = os.path.join(clip_dir, d)
                if os.path.isdir(full):
                    model_list.append(d)
        if not model_list:
            model_list = ["t5-small"]
        return {
            "required": {
                "model_name": (model_list, {"default": model_list[0] if model_list else "t5-small"}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"
    CATEGORY = "P1Physics"

    def load(self, model_name):
        clip_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "clip")
        full_path = os.path.join(clip_dir, model_name)
        if not os.path.isdir(full_path):
            full_path = model_name

        print(f"[NativeCLIPLoader] 加载 T5 模型: {full_path}")
        tokenizer = T5Tokenizer.from_pretrained(full_path)
        model = T5EncoderModel.from_pretrained(full_path).eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 构建 PhysicsExtractor 并加载/生成物理轴
        extractor = PhysicsExtractor(axis_cache_dir=os.path.dirname(__file__))
        class _Adapter:
            pass
        adapter = _Adapter()
        adapter.model = model
        adapter.tokenizer = tokenizer
        extractor.load_or_build_axes(adapter)

        class EnhancedCLIP:
            def __init__(self, model, tokenizer, physics_extractor):
                self.model = model
                self.tokenizer = tokenizer
                self.cond_stage_model = model
                self.physics_extractor = physics_extractor

            def tokenize(self, text):
                return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)["input_ids"]

            def encode_from_tokens(self, tokens, return_pooled=True):
                device = next(self.model.parameters()).device
                tokens = tokens.to(device)
                with torch.no_grad():
                    outputs = self.model(input_ids=tokens)
                last_hidden = outputs.last_hidden_state
                pooled = last_hidden.mean(dim=1)
                if return_pooled:
                    return (last_hidden, pooled)
                return (last_hidden,)

        clip = EnhancedCLIP(model, tokenizer, extractor)
        return (clip,)


class P1PhysicalBridge:
    """
    P1 物理桥接节点：接收 T5 CLIP 和提示词，生成物理注入包。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "latent": ("LATENT",),
                "positive_text": ("STRING", {"multiline": True, "default": ""}),
                "negative_text": ("STRING", {"multiline": True, "default": ""}),
                "sensitivity": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0}),
                "friction_damp": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0}),
                "layer_index": ("INT", {"default": 5, "min": 0, "max": 8}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "P1_PACKAGE")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "p1_package")
    FUNCTION = "process"
    CATEGORY = "P1 Physics"

    def _safe_encode(self, clip, text):
        if not text.strip():
            text = " "
        tokens = clip.tokenize(text)
        result = clip.encode_from_tokens(tokens, return_pooled=True)
        if isinstance(result, tuple):
            cond_tensor, pooled = result
        else:
            cond_tensor = result
            pooled = None
        if pooled is None and cond_tensor is not None:
            pooled = cond_tensor.mean(dim=1).squeeze(1) if cond_tensor.dim() == 3 else cond_tensor.mean(dim=0)
        return cond_tensor, pooled

    def process(self, model, clip, latent, positive_text, negative_text, sensitivity, friction_damp, layer_index):
        # 安全检查
        if not hasattr(clip, 'physics_extractor'):
            raise RuntimeError(
                "[P1Bridge] ❌ CLIP 对象不兼容。请使用 'Native CLIP Loader (Local T5)' 节点加载 T5 模型并连接至此。"
            )

        samples = latent["samples"]
        if samples.dim() == 5:
            B, C, F_frames, H, W = samples.shape
            video_frames = F_frames
        elif samples.dim() == 4:
            B, C, H, W = samples.shape
            video_frames = 1
        else:
            raise ValueError(f"[P1Bridge] 无法识别的 Latent 维度: {samples.shape}")

        # 编码文本
        pos_cond_tensor, pos_pooled = self._safe_encode(clip, positive_text)
        neg_cond_tensor, neg_pooled = self._safe_encode(clip, negative_text)

        # 维度适配到 4096（WAN 2.1 要求）
        src_dim = pos_cond_tensor.shape[-1]
        target_dim = 4096
        if src_dim != target_dim:
            proj_seq = nn.Linear(src_dim, target_dim, bias=False).to(
                device=pos_cond_tensor.device, dtype=pos_cond_tensor.dtype)
            proj_pool = nn.Linear(src_dim, target_dim, bias=False).to(
                device=pos_pooled.device, dtype=pos_pooled.dtype)
            pos_cond_tensor = proj_seq(pos_cond_tensor)
            neg_cond_tensor = proj_seq(neg_cond_tensor)
            pos_pooled = proj_pool(pos_pooled)
            neg_pooled = proj_pool(neg_pooled)

        # 构造 conditioning
        pos_cond = [(pos_cond_tensor.cpu(), {"pooled_output": pos_pooled.cpu()})]
        neg_cond = [(neg_cond_tensor.cpu(), {"pooled_output": neg_pooled.cpu()})]

        # 物理参数提取与分发
        extractor = clip.physics_extractor
        phrase_physics = extractor.extract_phrases_from_text(positive_text, clip)

        sorter = P1Sorter()
        distribution = sorter.distribute_from_t5(phrase_physics)
        factory_token_map = distribution["factory_token_map"]
        global_physics = distribution["global_params"]

        # 打印关键摘要
        total_objects = sum(len(tokens) for tokens in factory_token_map.values())
        print(f"[P1Bridge] 物理提取: {total_objects} 个物体, 全局物理: {global_physics}")
        for fac, tokens in factory_token_map.items():
            for obj_name, params in tokens:
                print(f"  → {obj_name} → {fac}")

        # 启动 MasterManager 生成物理注入张量
        manager = P1MasterManager(
            global_resolution=(H, W),
            target_model_channels=48,
            target_device=str(samples.device)
        )

        result = manager.process_frame(
            conditioning_tensor=pos_cond_tensor,
            hidden_states=pos_pooled.unsqueeze(1) if pos_pooled.dim() == 2 else pos_pooled,
            global_physics=global_physics,
            factory_token_map=factory_token_map,
            external_timestamp=0.0
        )

        p1_shadow_frames = result.get("p1_shadow_frames")
        if p1_shadow_frames is None or p1_shadow_frames.numel() == 0:
            raise ValueError("[P1Bridge] p1_shadow_frames 为空！")

        # 构造首尾帧（单帧则相同）
        if p1_shadow_frames.shape[0] >= 2:
            start_frame = p1_shadow_frames[0].unsqueeze(0)
            end_frame = p1_shadow_frames[1].unsqueeze(0)
        else:
            start_frame = p1_shadow_frames[0].unsqueeze(0)
            end_frame = p1_shadow_frames[0].unsqueeze(0)

        # 暗号映射
        cipher_map = {"stiffness": "Sf", "friction": "Sm", "elasticity": "L",
                      "wetness": "Lh", "tear_intensity": "Fo", "tension": "Ft", "body_type": "Bd"}
        global_ciphers = {cipher_map.get(k, k): v for k, v in global_physics.items()}

        # 构建 p1_package，传递结构化资产（关键修复）
        p1_package = {
            "p1_shadow_frames": p1_shadow_frames,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "assets": {
                "factory_token_map": factory_token_map,   # 结构化令牌
                "global_physics": global_physics,
                "global_ciphers": global_ciphers
            },
            "ciphers": list(global_ciphers.keys()),
            "resolution": (H, W),
            "valid": True,
            "video_frames": video_frames,
        }

        print(f"[P1Bridge] 输出物理张量: {list(p1_shadow_frames.shape)}, 视频帧数: {video_frames}")
        return (model, pos_cond, neg_cond, latent, p1_package)


NODE_CLASS_MAPPINGS = {
    "NativeCLIPLoader": NativeCLIPLoader,
    "P1PhysicalBridge": P1PhysicalBridge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NativeCLIPLoader": "🌉 Native CLIP Loader (Local T5)",
    "P1PhysicalBridge": "🔌 P1 Physical Bridge"
}