"""
P1 分离架构集成测试（短语级提取 + 工厂统一 **kwargs 接口）
用法：python test_pipeline.py
说明：无需修改其他模块，脚本会自动模拟包环境，解决相对导入问题。
"""
import sys
import os
import torch
import json

# ===================== 模拟包环境 =====================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == "__main__" and __package__ is None:
    __package__ = "p1"
# =======================================================

try:
    from .physics_extractor import PhysicsExtractor
    from .p1_sorter import P1Sorter
    from .p1_master_manager import P1MasterManager
except ImportError as e:
    print(f"❌ 导入模块失败：{e}", flush=True)
    sys.exit(1)

from transformers import T5Tokenizer, T5EncoderModel


class MockCLIP:
    """模拟 ComfyUI 的 CLIP 对象"""
    def __init__(self, model_path: str):
        print(f"   加载本地 T5 模型: {model_path}", flush=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5EncoderModel.from_pretrained(model_path).eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)["input_ids"]

    def encode_from_tokens(self, tokens, return_pooled=True):
        with torch.no_grad():
            outputs = self.model(input_ids=tokens.to(self.device))
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        return (hidden, pooled) if return_pooled else (hidden,)


class MockFactory:
    """模拟工厂，统一 **kwargs 接收参数"""
    def __init__(self, name, channels):
        self.name = name
        self.channels = channels

    def produce(self, **kwargs):
        H, W = kwargs.get('global_resolution', (64, 64))
        tensor = torch.randn(1, self.channels, H, W)
        return {"tensor": tensor, "factory": self.name}


def run_test():
    print("\n" + "=" * 60, flush=True)
    print("  P1 分离架构集成测试（短语级提取 + 物体级分流）", flush=True)
    print("=" * 60, flush=True)

    local_model_path = r"D:\ComfyUI-aki-v3\ComfyUI\models\clip\t5-small"
    if not os.path.exists(local_model_path):
        print(f"\n❌ 模型路径不存在: {local_model_path}", flush=True)
        return

    # 1. 加载 CLIP
    print("\n[1] 加载本地 T5 模型...", flush=True)
    try:
        clip = MockCLIP(local_model_path)
        print("   ✅ 模型加载成功", flush=True)
    except Exception as e:
        print(f"   ❌ 加载失败: {e}", flush=True)
        return

    # 2. 初始化 PhysicsExtractor
    print("\n[2] 初始化 PhysicsExtractor...", flush=True)
    extractor = PhysicsExtractor(axis_cache_dir=current_dir)
    try:
        axes = extractor.load_or_build_axes(clip)
        print(f"   物理轴形状: {axes.shape}", flush=True)
    except Exception as e:
        print(f"   ❌ 物理轴初始化失败: {e}", flush=True)
        return

    # 3. 短语级物理提取（关键改动）
    prompt = "A highly-detailed cinematic shot of an athletic cybernetic girl with metallic porcelain skin and a soft organic face, sprinting fiercely in a neon alleyway during a torrential monsoon downpour. She is wearing a multi-layered rigid heavy leather trench coat, an ultra-thin translucent silk chiffon dress, a tight high-tension rubber corset, and a cold heavy steel armor gauntlet on her mechanical left arm. Massive splashing raindrops explode on her shoulders, generating dense misty water vapor and running liquid mud on the ground. She is forcefully stretching a high-elasticity composite longbow, her knees bending with high-pressure hydraulic joint friction, body muscles tensed with immense physical torque"
    print(f"\n[3] 短语级物理提取，提示词: '{prompt}'", flush=True)
    try:
        phrase_physics = extractor.extract_phrases_from_text(prompt, clip)
        print(f"   提取结果: {json.dumps(phrase_physics, indent=2)}", flush=True)
    except Exception as e:
        print(f"   ❌ 提取失败: {e}", flush=True)
        return

    # 4. P1Sorter 分发（直接传入短语字典）
    print("\n[4] P1Sorter 物理驱动分发...", flush=True)
    sorter = P1Sorter()
    dist = sorter.distribute_from_t5(phrase_physics)
    print(f"   全局物理参数: {dist['global_params']}", flush=True)
    for fac, tokens in dist['factory_token_map'].items():
        for tok, params in tokens:
            print(f"     {tok} → {fac}: {params}", flush=True)

    # 5. 初始化 MasterManager
    print("\n[5] 初始化 P1MasterManager...", flush=True)
    manager = P1MasterManager(
        raw_config={},
        global_resolution=(64, 64),
        target_model_channels=48,
        target_device="cpu"
    )

    # 6. 注入 Mock 工厂
    print("\n[6] 注入 Mock 工厂...", flush=True)
    manager.skin_factory = MockFactory("skin_factory", 16)
    manager.joint_factory = MockFactory("joint_factory", 16)
    manager.fabric_factory = MockFactory("fabric_factory", 8)
    manager.fluid_factory = MockFactory("fluid_factory", 12)

    # 7. 运行 process_frame
    print("\n[7] 执行 process_frame...", flush=True)
    conditioning_tensor = torch.randn(1, 77, 768)
    hidden_states = torch.randn(1, 1, 768)

    try:
        result = manager.process_frame(
            conditioning_tensor=conditioning_tensor,
            hidden_states=hidden_states,
            global_physics=dist["global_params"],
            factory_token_map=dist["factory_token_map"]
        )
    except Exception as e:
        print(f"   ❌ process_frame 失败: {e}", flush=True)
        return

    # 8. 检查结果
    print("\n[8] 测试结果:", flush=True)
    if result["p1_stat"] == "success":
        tensor = result["p1_shadow_frames"]
        print(f"   ✅ 成功！输出张量形状: {list(tensor.shape)}", flush=True)
        print(f"   均值: {tensor.mean().item():.4f}", flush=True)
        print(f"   标准差: {tensor.std().item():.4f}", flush=True)
        assets = result.get("assets", {})
        print(f"   全局物理参数: {assets.get('global_physics')}", flush=True)
        print(f"   暗号通道: {assets.get('global_ciphers')}", flush=True)
    else:
        print(f"   ❌ 失败，状态: {result['p1_stat']}", flush=True)


if __name__ == "__main__":
    run_test()