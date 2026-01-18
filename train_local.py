#!/usr/bin/env python
"""
Local training script for YOLOv12 on RTX 3050 Ti
Supports A2C2f with flash attention (RTX 3050 Ti compute 8.6 ✅)
"""

import torch
import gc
from pathlib import Path
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# ============================================================================

# Dataset configuration
DATASET_PATH = "c:/Users/abass/Downloads/Thesis/yolov12/coral-dataset/data.yaml"  # Coral dataset (v4)
# Example: "C:/path/to/dataset/data.yaml"
# The YAML should have: path, train, val, nc, names

# Model configuration
MODEL_CONFIG = "c:/Users/abass/Downloads/Thesis/yolov12/ultralytics/cfg/models/v12/yolov12.yaml"
PRETRAINED_WEIGHTS = None  # Set to model path if you want to continue from checkpoint

# Training parameters
EPOCHS = 50
BATCH_SIZE = 2   # Safe for 4GB VRAM
IMGSZ = 320      # Larger than 320 but still safe
DEVICE = 0  # GPU device ID

# Optional training parameters
PATIENCE = 5  # Early stopping patience
CACHE = True  # Cache images for faster training
AMP = True  # Automatic Mixed Precision (saves ~40% memory)
WORKERS = 4  # DataLoader workers


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_gpu():
    """Check GPU capabilities and availability."""
    print("=" * 70)
    print("GPU INFORMATION")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU available!")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = device_props.major + device_props.minor / 10
    total_memory = device_props.total_memory / 1e9
    
    print(f"GPU Name: {device_name}")
    print(f"Compute Capability: {compute_capability}")
    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check flash attention compatibility
    if device_props.major >= 8:
        print(f"✅ Flash Attention: SUPPORTED (compute {compute_capability})")
    else:
        print(f"⚠️  Flash Attention: NOT supported (compute {compute_capability}, requires 8.0+)")
    
    return True

def clear_memory():
    """Clear GPU and system memory."""
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleared ✓")

def verify_dataset(dataset_path):
    """Verify dataset exists and has required structure."""
    print("\n" + "=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)
    
    yaml_path = Path(dataset_path)
    
    if not yaml_path.exists():
        print(f"❌ Dataset YAML not found: {dataset_path}")
        return False
    
    print(f"✅ Dataset YAML found: {yaml_path}")
    
    # Read YAML to check structure
    import yaml
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_root = yaml_path.parent
    
    # Check train/val directories
    train_path = dataset_root / data_config.get('train', 'images/train')
    val_path = dataset_root / data_config.get('val', 'images/val')
    
    print(f"Dataset Root: {dataset_root}")
    print(f"Train Path: {train_path}")
    print(f"Val Path: {val_path}")
    
    if train_path.exists():
        train_count = len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png')))
        print(f"  ✓ Training images: {train_count}")
    else:
        print(f"  ❌ Training path not found")
        return False
    
    if val_path.exists():
        val_count = len(list(val_path.glob('*.jpg'))) + len(list(val_path.glob('*.png')))
        print(f"  ✓ Validation images: {val_count}")
    else:
        print(f"  ❌ Validation path not found")
        return False
    
    print(f"Classes: {data_config.get('nc', 'Unknown')}")
    
    return True

def print_model_info(model):
    """Print model information."""
    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "YOLOv12 Local Training (RTX 3050 Ti)" + " " * 17 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Step 1: Check GPU
    if not check_gpu():
        print("\n❌ GPU check failed. Exiting.")
        return
    
    # Step 2: Clear memory
    clear_memory()
    
    # Step 3: Verify dataset
    if not verify_dataset(DATASET_PATH):
        print("\n❌ Dataset verification failed.")
        print(f"Please update DATASET_PATH in this script to point to your dataset")
        print(f"Current value: {DATASET_PATH}")
        return
    
    # Step 4: Load model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    
    try:
        if PRETRAINED_WEIGHTS:
            print(f"Loading from checkpoint: {PRETRAINED_WEIGHTS}")
            model = YOLO(PRETRAINED_WEIGHTS)
        else:
            print(f"Loading from config: {MODEL_CONFIG}")
            model = YOLO(MODEL_CONFIG)
        
        print("✅ Model loaded successfully")
        print_model_info(model)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Step 5: Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMGSZ}")
    print(f"AMP: {AMP}")
    print(f"Cache: {CACHE}")
    print("\n")

    try:
        results = model.train(
            data=DATASET_PATH,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            device=DEVICE,
            patience=5,
            save=True,
            project='runs/train',
            name='proposed_yolov12n_underwater_training',
            weight_decay=0.0005
        )
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        # Print results summary
        if hasattr(results, 'results_dict'):
            print("\nTraining Results:")
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        return results
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return None
    
    except RuntimeError as e:
        print(f"\n❌ Runtime Error during training: {e}")
        print("\nTroubleshooting:")
        print("1. Reduce BATCH_SIZE (try 4 or 6)")
        print("2. Reduce IMGSZ (try 512 or 416)")
        print("3. Set AMP=False to disable mixed precision")
        return None
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
