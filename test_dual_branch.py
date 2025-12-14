"""
Test script to verify the dual-branch architecture with StandardBranch, DenoisingBranch, and AdaptiveFeatureFusion.
"""

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import StandardBranch, DenoisingBranch, AdaptiveFeatureFusion

def test_individual_components():
    """Test StandardBranch, DenoisingBranch, and AdaptiveFeatureFusion individually."""
    print("=" * 60)
    print("Testing Individual Components")
    print("=" * 60)
    
    # Create dummy input
    x = torch.randn(2, 128, 32, 32)
    
    # Test StandardBranch
    print("\n1. Testing StandardBranch (2 Conv layers)...")
    std_branch = StandardBranch(c1=128, c2=256)
    std_out = std_branch(x)
    print(f"   ✓ StandardBranch output shape: {std_out.shape}")
    assert std_out.shape == (2, 256, 32, 32), "StandardBranch output shape mismatch!"
    
    # Test DenoisingBranch
    print("\n2. Testing DenoisingBranch (Depthwise Separable Conv)...")
    denoise_branch = DenoisingBranch(c1=128, c2=256)
    denoise_out = denoise_branch(x)
    print(f"   ✓ DenoisingBranch output shape: {denoise_out.shape}")
    assert denoise_out.shape == (2, 256, 32, 32), "DenoisingBranch output shape mismatch!"
    
    # Test AdaptiveFeatureFusion
    print("\n3. Testing AdaptiveFeatureFusion...")
    fusion = AdaptiveFeatureFusion(c=256)
    fused_out = fusion(std_out, denoise_out)
    print(f"   ✓ AdaptiveFeatureFusion output shape: {fused_out.shape}")
    assert fused_out.shape == (2, 256, 32, 32), "AdaptiveFeatureFusion output shape mismatch!"
    
    print("\n✓ All individual components work correctly!\n")


def test_yaml_model():
    """Test loading the model from the modified YAML."""
    print("=" * 60)
    print("Testing YAML Model with Dual-Branch Architecture")
    print("=" * 60)
    
    try:
        print("\nLoading YOLOv12 model from yolov12.yaml...")
        model = YOLO("ultralytics/cfg/models/v12/yolov12.yaml")
        print("✓ Model loaded successfully!")
        
        # Print model summary
        print("\nModel Summary:")
        model.info()
        
        # Test forward pass
        print("\nTesting forward pass with dummy image...")
        x = torch.randn(1, 3, 640, 640)
        results = model(x)
        print(f"✓ Forward pass successful!")
        print(f"✓ Output shape: {results[0].shape if isinstance(results, list) else results.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading or testing model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "DUAL-BRANCH ARCHITECTURE TEST" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Test individual components
    test_individual_components()
    
    # Test YAML model
    success = test_yaml_model()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour dual-branch architecture is ready to use!")
        print("\nArchitecture Overview:")
        print("  • StandardBranch: 2 Conv layers for standard feature extraction")
        print("  • DenoisingBranch: Depthwise separable Conv for noise reduction")
        print("  • AdaptiveFeatureFusion: Learnable fusion of both branches")
        print("\nNext Steps:")
        print("  1. Train the model: python train.py --model yolov12.yaml --data coco8.yaml")
        print("  2. Monitor the learnable weights in AdaptiveFeatureFusion during training")
        print("=" * 60)
    else:
        print("✗ SOME TESTS FAILED - Check the errors above")
        print("=" * 60)


if __name__ == "__main__":
    main()
