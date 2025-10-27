import os
import torch
import ultralytics
from ultralytics import YOLO

# ⚠️ Force torch.load to use weights_only=False safely
_old_torch_load = torch.load
def unsafe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False  # explicitly disable the new safety flag
    return _old_torch_load(*args, **kwargs)
torch.load = unsafe_torch_load

# ✅ Add all required YOLO layer classes for safe unpickling
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    ultralytics.nn.modules.conv.Conv,
    ultralytics.nn.modules.block.C2f,
    ultralytics.nn.modules.block.SPPF,
    ultralytics.nn.modules.head.Detect,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
])

def main():
    # Set working directory to your dataset location
    os.chdir(r'C:\Users\sonka\OneDrive\Desktop\project_inter\project')

    print("="*60)
    print("🚀 WASTE CLASSIFICATION - YOLOV8 TRAINING")
    print("="*60)

    print("\n📥 Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Loads the YOLOv8n (nano) model

    print("🔥 Starting training...")

    results = model.train(
        data=r'C:\Users\sonka\OneDrive\Desktop\project_inter\project\data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device='0',  # change to 'cuda' if you have a GPU
        patience=20,
        save=True,
        verbose=True
    )

    print("\n✅ Training complete!")
    print(f"Best model saved at: runs/detect/train/weights/best.pt")
    print("="*60)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # ✅ needed on Windows
    main()