# P3 专用安全注册脚本
# 与P1、P2完全独立，互不影响
class P3_HealthCheck:
    """P3模块健康检查节点，确保能被ComfyUI识别"""
    CATEGORY = "🔥 P3 我的节点"
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "status"

    def status(self):
        return ("✅ P3 模块已成功加载",)

# 先定义基础映射
NODE_CLASS_MAPPINGS = {
    "P3_HealthCheck": P3_HealthCheck
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "P3_HealthCheck": "P3 健康检查节点"
}

# 安全导入你的核心P3节点，出错也不会炸整个模块
try:
    from .p3_comfy_nodes import NODE_CLASS_MAPPINGS as P3_NODES, NODE_DISPLAY_NAME_MAPPINGS as P3_NAMES
    NODE_CLASS_MAPPINGS.update(P3_NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(P3_NAMES)
    print("✅ P3 核心节点已成功加载！")
except Exception as e:
    print(f"⚠️ P3 核心节点加载出错，但健康检查节点仍可用：{e}")

# 暴露给 ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]