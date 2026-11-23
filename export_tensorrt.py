#!/usr/bin/env python3
"""
Export all models to TensorRT INT8/FP16 for Jetson Orin deployment
Supports: YOLOv10, OSNet ReID, ST-GCN, PPO-GAT
"""

import argparse
import os
import torch
import tensorrt as trt
import onnx
from pathlib import Path


def export_yolo_tensorrt(onnx_path: str, engine_path: str, 
                         precision: str = 'int8', batch_size: int = 8,
                         workspace_gb: int = 4):
    """Export YOLOv10 ONNX to TensorRT engine"""
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # Network
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    
    # Parse ONNX
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(i)}")
            return False
    
    # Config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    
    # Precision
    if precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        # Note: INT8 requires calibration dataset
        print("⚠️ INT8 mode requires calibration. Using FP16 fallback for now.")
        config.clear_flag(trt.BuilderFlag.INT8)
    elif precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Optimization profiles for dynamic batch
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    # Dynamic batch: min=1, opt=batch_size, max=batch_size*2
    profile.set_shape(input_name, 
                     (1, *input_tensor.shape[1:]),
                     (batch_size, *input_tensor.shape[1:]),
                     (batch_size * 2, *input_tensor.shape[1:]))
    config.add_optimization_profile(profile)
    
    # Build
    print(f"Building TensorRT engine ({precision.upper()})...")
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes is None:
        print("❌ Engine build failed")
        return False
    
    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)
    
    print(f"✅ Engine saved to {engine_path}")
    return True


def export_reid_tensorrt(onnx_path: str, engine_path: str, precision: str = 'fp16'):
    """Export OSNet ReID ONNX to TensorRT"""
    return export_yolo_tensorrt(onnx_path, engine_path, precision, batch_size=32, workspace_gb=2)


def export_anomaly_tensorrt(onnx_path: str, engine_path: str, precision: str = 'fp16'):
    """Export ST-GCN Anomaly ONNX to TensorRT"""
    return export_yolo_tensorrt(onnx_path, engine_path, precision, batch_size=16, workspace_gb=2)


def export_signal_rl_torchscript(model_path: str, output_path: str):
    """Export PPO-GAT policy to TorchScript"""
    import sys
    sys.path.append('models/signal_rl')
    from ppo_gat_signal import GATActorCriticPolicy, IsfahanSignalEnv
    from stable_baselines3 import PPO
    
    # Load model
    model = PPO.load(model_path)
    policy = model.policy
    policy.eval()
    
    # Dummy input matching observation space
    dummy_obs = {
        'node_features': torch.randn(1, 5, 19),
        'edge_index': torch.tensor([[0,1,1,2,2,3,3,4,0,3,1,4],[1,0,2,1,3,2,4,3,3,0,4,1]], dtype=torch.long),
        'edge_attr': torch.rand(12, 3),
    }
    
    # Trace
    with torch.no_grad():
        traced = torch.jit.trace(policy, dummy_obs, strict=False)
    
    traced.save(output_path)
    print(f"✅ TorchScript saved to {output_path}")


def verify_engine(engine_path: str, input_shape: tuple = (1, 3, 640, 640)):
    """Verify TensorRT engine loads and runs"""
    import pycuda.driver as cuda
    import pycuda.autoinit
    import numpy as np
    
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print(f"❌ Failed to load engine: {engine_path}")
        return False
    
    context = engine.create_execution_context()
    
    # Set input shape
    context.set_input_shape(engine.get_tensor_name(0), input_shape)
    
    # Allocate buffers
    bindings = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = trt.volume(shape)
        
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            host_mem = np.random.randn(*shape).astype(dtype)
        else:
            host_mem = np.empty(shape, dtype=dtype)
        
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cuda.memcpy_htod(device_mem, host_mem)
        context.set_tensor_address(name, int(device_mem))
    
    # Execute
    context.execute_v2(bindings)
    
    # Copy output
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            host_mem = np.empty(shape, dtype=dtype)
            device_mem = bindings[i]
            cuda.memcpy_dtoh(host_mem, device_mem)
            print(f"  Output {name}: shape={host_mem.shape}, dtype={dtype}")
    
    print(f"✅ Engine verification passed: {engine_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Export models to TensorRT')
    parser.add_argument('--all', action='store_true', help='Export all models')
    parser.add_argument('--yolo', action='store_true', help='Export YOLOv10')
    parser.add_argument('--reid', action='store_true', help='Export ReID')
    parser.add_argument('--anomaly', action='store_true', help='Export Anomaly ST-GCN')
    parser.add_argument('--signal', action='store_true', help='Export Signal RL')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workspace', type=int, default=4, help='Workspace GB')
    parser.add_argument('--verify', action='store_true', help='Verify engines after export')
    args = parser.parse_args()
    
    if not any([args.all, args.yolo, args.reid, args.anomaly, args.signal]):
        parser.print_help()
        return
    
    models_dir = Path('models')
    engines_dir = Path('edge/triton_model_repo')
    engines_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLOv10 Detection
    if args.all or args.yolo:
        print("\n🔧 Exporting YOLOv10-ISF...")
        export_yolo_tensorrt(
            onnx_path=models_dir / 'detection/weights/yolov10n_isfahan.onnx',
            engine_path=engines_dir / 'detection_isfahan/yolov10n_isfahan_int8.engine',
            precision=args.precision,
            batch_size=args.batch_size,
            workspace_gb=args.workspace
        )
    
    # ReID
    if args.all or args.reid:
        print("\n🔧 Exporting OSNet ReID...")
        export_reid_tensorrt(
            onnx_path=models_dir / 'tracking/weights/osnet_x1_0_isfahan.onnx',
            engine_path=engines_dir / 'tracking_reid/osnet_x1_0_isfahan.engine',
            precision=args.precision
        )
    
    # Anomaly ST-GCN
    if args.all or args.anomaly:
        print("\n🔧 Exporting ST-GCN Anomaly...")
        export_anomaly_tensorrt(
            onnx_path=models_dir / 'anomaly/weights/stgcn_isfahan.onnx',
            engine_path=engines_dir / 'anomaly_stgcn/stgcn_isfahan.engine',
            precision=args.precision
        )
    
    # Signal RL
    if args.all or args.signal:
        print("\n🔧 Exporting PPO-GAT Signal RL...")
        export_signal_rl_torchscript(
            model_path=models_dir / 'signal_rl/weights/ppo_gat_isfahan.zip',
            output_path=engines_dir / 'signal_rl/ppo_gat_isfahan.ts'
        )
    
    # Verify
    if args.verify:
        print("\n🔍 Verifying engines...")
        for engine_path in engines_dir.rglob('*.engine'):
            verify_engine(engine_path)


if __name__ == '__main__':
    main()