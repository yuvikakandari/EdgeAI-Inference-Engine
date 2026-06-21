import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(input_path="models/onnx/mobilenet_v2_simplified.onnx", output_path="models/onnx/mobilenet_v2_int8.onnx"):
    print("\nStarting Dynamic INT8 Quantization ---")
    
    # Check if the simplified model exists
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}. Please run optimize.py first.")
        return

    # Execute dynamic quantization converting FP32 weights to INT8 integers
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8
    )
    
    print(f" Success: Quantized model saved at: {output_path}")
    
    # Print size comparison metrics
    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    compression_ratio = orig_size / quant_size
    
    print("\n QUANTIZATION METRICS:")
    print(f"   Simplified FP32 Size : {orig_size:.2f} MB")
    print(f"   Quantized INT8 Size  : {quant_size:.2f} MB")
    print(f"   Compression Ratio    : {compression_ratio:.2f}x smaller")

if __name__ == "__main__":
    quantize_model()