"""
Test script to verify YOLOv12 model initialization with the fixed configuration.
Similar to the Google Colab test.
"""

from ultralytics import YOLO
import torch

print("=" * 60)
print("Testing YOLOv12 Model Initialization")
print("=" * 60)

try:
    # Initialize model with the architecture configuration
    print("\n1. Initializing model from YAML config...")
    model = YOLO('ultralytics/cfg/models/v12/yolov12.yaml')
    
    print("✅ Model initialized successfully!")
    print(f"   Model type: {type(model)}")
    
    # Print model summary
    print("\n2. Model Summary:")
    print(f"   Model: {model.model}")
    
    # Test with a dummy input to verify the architecture works
    print("\n3. Testing forward pass with dummy input (small size for CPU)...")
    dummy_input = torch.randn(1, 3, 64, 64)  # Smaller size for CPU testing
    
    if torch.cuda.is_available():
        print("   CUDA available - using GPU")
        model.model.to('cuda')
        dummy_input = dummy_input.to('cuda')
    else:
        print("   CUDA not available - using CPU (with small input)")
    
    model.model.eval()
    with torch.no_grad():
        output = model.model(dummy_input)
    
    print("✅ Forward pass successful!")
    print(f"   Output type: {type(output)}")
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Model is ready for training.")
    print("=" * 60)
    
except AssertionError as e:
    print(f"\n❌ AssertionError: {e}")
    print("\nThis error typically means channel dimensions are not compatible.")
    print("Check that all A2C2f layers have sufficient channels for the scale factor.")
    
except Exception as e:
    print(f"\n❌ Error occurred: {type(e).__name__}")
    print(f"   Message: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

print("\n")
