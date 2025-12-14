#!/usr/bin/env python3
"""
Simple YOLOv12 Training Test
"""

from ultralytics import YOLO

print("=" * 70)
print("YOLOv12 Test Training - Simple Version")
print("=" * 70)

try:
    print("\n[1/2] Loading YOLOv12-Nano model...")
    model = YOLO('yolov12n.pt')
    print("✓ Model loaded successfully")
    
    print("\n[2/2] Starting 1-epoch training...")
    print("  Training on COCO8 dataset...")
    print("  This will download COCO8 automatically")
    print("  Modified settings:")
    print("    - Image size: 416 (modified to test changes)")
    print("    - Batch size: 2 (small for testing)")
    print("    - Workers: 0 (no multiprocessing)")
    
    results = model.train(
        data='coco8.yaml',
        epochs=1,
        imgsz=416,
        batch=2,#16 daw
        patience=10,
        save=True,
        verbose=True,
        project='runs/detect',
        name='test_training',
        workers=0,
        close_mosaic=10,
        device='',
    )
    
    print("\n" + "=" * 70)
    print("✓ Training completed successfully!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
