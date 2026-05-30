# p1_customs_clearance.py - 修复版
import gc, torch
from typing import Dict, Any, List, Optional

class P1CustomsClearance:
    def __init__(self):
        self.factory_name = "customs_clearance"
        self._death_list: List[Any] = []
    def register_for_cleanup(self, factory_instance: Any) -> None:
        if factory_instance not in self._death_list:
            self._death_list.append(factory_instance)
    def _enforce_dtype(self, tensor, target_dtype):
        if tensor is None: return None
        if tensor.dtype != target_dtype: tensor = tensor.to(target_dtype)
        return tensor
    def _align_device(self, tensor, target_device):
        if tensor is None: return None
        if tensor.device != target_device: tensor = tensor.to(target_device)
        return tensor
    def _check_nan_inf(self, tensor, name):
        if tensor is None: return None
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor
    def _adapt_channels(self, tensor, target_channels, strategy='zero'):
        if tensor is None: return None
        if tensor.ndim == 3: tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 5: tensor = tensor.view(-1, tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])
        elif tensor.ndim == 2: tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim != 4: raise ValueError(f"不支持维度 {tensor.ndim}")
        B, C_in, H, W = tensor.shape
        if C_in == target_channels: return tensor
        if C_in < target_channels:
            out = torch.zeros(B, target_channels, H, W, device=tensor.device, dtype=tensor.dtype)
            out[:, :C_in] = tensor
            return out
        else:
            return tensor[:, :target_channels]
    def _validate_channels(self, tensor, expected_channels, name):
        if tensor is None: return
        if tensor.shape[1] != expected_channels: raise ValueError(f"通道数不匹配")
    def _destroy_factories(self):
        for f in self._death_list:
            if hasattr(f, '_cache') and isinstance(f._cache, dict): f._cache.clear()
        self._death_list.clear()
    def clear_for_export(self, final_assembly_output, master_command):
        target_channels = master_command.get("target_channels")
        if target_channels is None: raise ValueError("缺少 target_channels")
        target_dtype_str = master_command.get("target_dtype", "float32")
        target_device_str = master_command.get("target_device", "cuda")
        clear_cache = master_command.get("clear_cache", True)
        adapter_strategy = master_command.get("channel_adapter_strategy", "zero")
        dtype_map = {"float32": torch.float32, "float16": torch.float16}
        target_dtype = dtype_map.get(target_dtype_str, torch.float32)
        target_device = torch.device(target_device_str if torch.cuda.is_available() else "cpu")
        final_tensor = final_assembly_output.get("final_tensor")
        if final_tensor is None: raise ValueError("缺少 final_tensor")
        final_tensor = self._adapt_channels(final_tensor, target_channels, adapter_strategy)
        self._validate_channels(final_tensor, target_channels, "final_tensor")
        final_tensor = self._check_nan_inf(final_tensor, "final_tensor")
        final_tensor = self._enforce_dtype(final_tensor, target_dtype)
        final_tensor = self._align_device(final_tensor, target_device)
        export_package = {"p1_final_tensor": final_tensor, "p1_stat": "success"}
        self._destroy_factories()
        del final_assembly_output, master_command
        gc.collect()
        if clear_cache and torch.cuda.is_available(): torch.cuda.empty_cache()
        return export_package
