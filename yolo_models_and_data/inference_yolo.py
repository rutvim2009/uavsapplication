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

        filename = os.path.splitext(os.path.basename(image))[0]
        out_txt_path = os.path.join("predictions", filename + ".out")
        with open(out_txt_path, "w") as f:
            if len(results[0].boxes) == 0:
                f.write("Blank\n")
            else:
                best_box = results[0].boxes[results[0].boxes.conf.argmax()]
                x1, y1, x2, y2 = best_box.xyxy.cpu().numpy().flatten()
                xc = int((x1 + x2) / 2)
                yc = int((y1 + y2) / 2)
                f.write(f"{xc} {yc}\n")

if __name__ == "__main__":
    run_inference()
