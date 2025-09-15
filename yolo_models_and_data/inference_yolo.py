from ultralytics import YOLO
import cv2
import os

def run_inference(model_path="/workspaces/uavsapplication/yolo_models_and_data/runs_odlc/exp/weights/best.pt", source="/workspaces/uavsapplication/yolo_models_and_data/yolo_model_test_images"):
    model = YOLO(model_path)
    images = [os.path.join(source, f) for f in os.listdir(source) if f.endswith((".jpg", ".png"))]

    os.makedirs("predictions", exist_ok=True)

    for image in images:
        results = model(image)
        annotated = results[0].plot() 
        out_path = os.path.join("predictions", os.path.basename(image))
        cv2.imwrite(out_path, annotated)
        print(f"saved: {out_path}")

if __name__ == "__main__":
    run_inference()
