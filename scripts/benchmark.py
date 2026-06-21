import os
import time
import numpy as np
import onnxruntime as ort

def benchmark_engine(model_path, name):
    # Initialize the session on CPU
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warm-up (10 runs)
    for _ in range(10):
        _ = session.run(None, {input_name: input_data})

    # Benchmark loop (100 runs)
    iterations = 100
    start_time = time.time()
    for _ in range(iterations):
        _ = session.run(None, {input_name: input_data})
    end_time = time.time()

    # Calculate metrics
    latency = (end_time - start_time) / iterations * 1000
    throughput = 1000 / latency
    file_size = os.path.getsize(model_path) / (1024 * 1024)

    print(f"\n[Results for {name}]:")
    print(f"   Model Size : {file_size:.2f} MB")
    print(f"   Latency    : {latency:.2f} ms per image")
    print(f"   Throughput : {throughput:.2f} FPS")
    
    return latency, throughput, file_size

if __name__ == "__main__":
    fp32_path = "models/onnx/mobilenet_v2_simplified.onnx"
    int8_path = "models/onnx/mobilenet_v2_int8.onnx"

    print("\nFinal Evaluation Engine")
    
    # Run benchmarks
    fp32_lat, fp32_fps, fp32_size = benchmark_engine(fp32_path, "Baseline FP32 Model")
    int8_lat, int8_fps, int8_size = benchmark_engine(int8_path, "Quantized INT8 Model")
    
    # Summary Analysis
    print("\n FINAL COMPARISON SUMMARY ")
    print(f" Speedup Factor   : {fp32_lat / int8_lat:.2f}x faster inference")
    print(f" Memory Reduction : {((fp32_size - int8_size) / fp32_size) * 100:.1f}% space saved")
