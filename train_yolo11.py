import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"  # RTX 5060 Blackwell 架構

import torch
from ultralytics import YOLO

if __name__ == "__main__":
    # 訓練前確認 GPU 環境（你已經確認過沒問題，這裡保留是為了每次訓練都留一份紀錄在 log 裡）
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    model = YOLO("yolo11s.pt")

    results = model.train(
        data="D:/NCKU_paper_Capstone_Project/dataset_merged/data.yaml",
        epochs=500,
        patience=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        seed=42,
        project="D:/NCKU_paper_Capstone_Project/runs/detect",
        name="chimney_smoke_fire_yolo11s",
        save=True,
        plots=True,
    )

    # 訓練結束後自動跑一次驗證，把 per-class 指標印出來，論文會用到
    print("\n=== 驗證結果 ===")
    metrics = model.val()
    print("mAP50 (整體):", metrics.box.map50)
    print("mAP50-95 (整體):", metrics.box.map)
    print("各類別 AP (依序對應 data.yaml 裡的類別順序):", metrics.box.maps)