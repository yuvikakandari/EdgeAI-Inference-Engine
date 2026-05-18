import time
import numpy as np
import torch
import torchvision.models as models
import onnx
import onnxruntime as ort

def export_mobilenet_to_onnx(output_path="models/onnx/mobilenet_v2.onnx"):
    """
    Loads a pre-trained MobileNetV2 model from PyTorch and exports it 
    to the standardized ONNX format.
    """
    print("\n 1.) Loading Pre-trained MobileNetV2")
    # Load the lightweight model built for edge devices
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()  # Set to evaluation mode (turns off dropout, batchnorm tracking)

    # Create dummy input simulating a single 3-channel (RGB) image of 224x224 pixels
    # Syntax: torch.randn(Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224)

    print("2.) Exporting to ONNX Format")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True,        # Store the trained weights inside the model file
        opset_version=12,          # Use a stable, widely supported hardware operator set
        do_constant_folding=True,  # Optimize by pre-calculating constant operations
        input_names=['input'],     # Name the input node for the deployment engine
        output_names=['output'],   # Name the output node
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Allow flexible batch sizes
    )
    print(f"✅ Success: Model saved locally at: {output_path}")