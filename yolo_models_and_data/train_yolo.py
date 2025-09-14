from ultralytics import YOLO
import os

def train_yolo():
    data_yaml = os.path.join("synthetic_odlc", "odlc_data.yaml")

    model = YOLO("yolov8n.pt")

    model.train(
        data=data_yaml,
        epochs=10,
        imgsz=640,
        batch=16,
        project="runs_odlc",
        name="exp",
        exist_ok=True
    )

if __name__ == "__main__":
    train_yolo()
