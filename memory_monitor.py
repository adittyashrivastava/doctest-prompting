#!/usr/bin/env python3
"""
Memory monitoring utility for tracking GPU and system memory usage.
"""
import os
import torch
import psutil
from typing import Dict, Optional

def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information in GB."""
    if not torch.cuda.is_available():
        return {"available": 0, "used": 0, "total": 0, "free": 0}
    
    # Get memory info from torch
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
    reserved = torch.cuda.memory_reserved() / (1024**3)
    
    # Try to get total GPU memory
    try:
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        total = info.total / (1024**3)
        used = info.used / (1024**3)
        free = info.free / (1024**3)
        nvml.nvmlShutdown()
    except (ImportError, Exception):
        # Fallback to torch info
        total = reserved + allocated if reserved > 0 else allocated * 2  # Rough estimate
        used = allocated
        free = total - used
    
    return {
        "total": total,
        "used": used,
        "free": free,
        "allocated_torch": allocated,
        "reserved_torch": reserved
    }

def get_system_memory_info() -> Dict[str, float]:
    """Get system memory information in GB."""
    memory = psutil.virtual_memory()
    return {
        "total": memory.total / (1024**3),
        "used": memory.used / (1024**3),
        "free": memory.available / (1024**3),
        "percent": memory.percent
    }

def print_memory_status(label: str = "Memory Status"):
    """Print current memory status."""
    print(f"\n📊 {label}")
    print("=" * 50)
    
    # GPU Memory
    gpu_info = get_gpu_memory_info()
    if gpu_info["total"] > 0:
        print(f"🎮 GPU Memory:")
        print(f"   Total: {gpu_info['total']:.2f} GB")
        print(f"   Used:  {gpu_info['used']:.2f} GB ({gpu_info['used']/gpu_info['total']*100:.1f}%)")
        print(f"   Free:  {gpu_info['free']:.2f} GB")
        print(f"   PyTorch Allocated: {gpu_info['allocated_torch']:.2f} GB")
        print(f"   PyTorch Reserved:  {gpu_info['reserved_torch']:.2f} GB")
    else:
        print("🎮 GPU Memory: Not available")
    
    # System Memory
    sys_info = get_system_memory_info()
    print(f"💻 System Memory:")
    print(f"   Total: {sys_info['total']:.2f} GB")
    print(f"   Used:  {sys_info['used']:.2f} GB ({sys_info['percent']:.1f}%)")
    print(f"   Free:  {sys_info['free']:.2f} GB")
    print("=" * 50)

def clear_gpu_cache():
    """Clear GPU cache and print memory freed."""
    if torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / (1024**3)
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / (1024**3)
        freed = before - after
        if freed > 0.01:  # Only report if significant
            print(f"🧹 Cleared {freed:.2f} GB from GPU cache")

class MemoryMonitor:
    """Context manager for monitoring memory usage."""
    
    def __init__(self, label: str = "Operation"):
        self.label = label
        self.start_gpu_info = None
        self.start_sys_info = None
    
    def __enter__(self):
        print_memory_status(f"Before {self.label}")
        self.start_gpu_info = get_gpu_memory_info()
        self.start_sys_info = get_system_memory_info()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_gpu_info = get_gpu_memory_info()
        end_sys_info = get_system_memory_info()
        
        print_memory_status(f"After {self.label}")
        
        # Calculate differences
        if self.start_gpu_info["total"] > 0:
            gpu_diff = end_gpu_info["used"] - self.start_gpu_info["used"]
            print(f"📈 GPU Memory Change: {gpu_diff:+.2f} GB")
        
        sys_diff = end_sys_info["used"] - self.start_sys_info["used"]
        print(f"📈 System Memory Change: {sys_diff:+.2f} GB")

def memory_usage_recommendations():
    """Print memory optimization recommendations."""
    gpu_info = get_gpu_memory_info()
    
    print("\n💡 Memory Optimization Recommendations:")
    print("=" * 50)
    
    if gpu_info["total"] > 0:
        if gpu_info["total"] < 8:
            print("🔴 Low GPU Memory (<8GB):")
            print("   - Use 4-bit quantization (load_in_4bit=True)")
            print("   - Consider CPU offloading for large models")
            print("   - Use smaller models (1.5B instead of 7B)")
        elif gpu_info["total"] < 16:
            print("🟡 Medium GPU Memory (8-16GB):")
            print("   - Use 8-bit quantization for 7B models")
            print("   - Use 4-bit quantization for 13B+ models")
        else:
            print("🟢 High GPU Memory (>16GB):")
            print("   - Can use float16 for most models")
            print("   - 8-bit quantization still recommended for 70B+ models")
    
    sys_info = get_system_memory_info()
    if sys_info["free"] < 4:
        print("⚠️  Low system memory - consider closing other applications")
    
    print("=" * 50)

if __name__ == "__main__":
    print("🔍 Current Memory Status")
    print_memory_status()
    memory_usage_recommendations() 