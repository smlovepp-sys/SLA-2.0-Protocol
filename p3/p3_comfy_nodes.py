# p3_comfy_nodes.py (清洁版：48ch 锁定，无调试打印)
import torch
from .p3_main_controller import P3Controller

class P3PhysicalOrchestrator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "p2_package": ("P2_PACKAGE",),
                "alpha": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "P3_PAYLOAD")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "p3_payload")
    FUNCTION = "process"
    CATEGORY = "P3 Physics"

    def process(self, model, positive, negative, latent, p2_package, alpha):
        controller = P3Controller(
            fps=30.0,
            alpha=alpha,
            vae_mode="48ch",
            storage_mode="disk",
            cache_dir=None
        )

        model_out, pos_out, neg_out, latent_out, payload = controller.process(
            model, positive, negative, latent, p2_package
        )

        return (model_out, pos_out, neg_out, latent_out, payload)


NODE_CLASS_MAPPINGS = {"P3PhysicalOrchestrator": P3PhysicalOrchestrator}
NODE_DISPLAY_NAME_MAPPINGS = {"P3PhysicalOrchestrator": "P3 Physical Orchestrator (48ch)"}