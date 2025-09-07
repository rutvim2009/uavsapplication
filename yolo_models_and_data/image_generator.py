
import os
import random
import numpy as np
import cv2


def make_background(h=1080, w=1920):
    
    base = np.full((h, w, 3), 200, dtype=np.uint8)
    if random.random() < 0.5:
        grad = np.tile(np.linspace(0, 50, w, dtype=np.uint8), (h, 1))
    else:
        grad = np.tile(np.linspace(0, 50, h, dtype=np.uint8), (w, 1)).T
    grad = cv2.merge([grad, grad, grad])
    return cv2.add(base, grad)

def make_odlc(size=200):
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    shape = random.choice(["circle", "square", "triangle"])
    color = tuple(np.random.randint(50, 220, 3).tolist())

    if shape == "square":
        pad = int(size * 0.15)
        cv2.rectangle(patch, (pad, pad), (size-pad, size-pad), color, -1)
    elif shape == "circle":
        cv2.circle(patch, (size//2, size//2), int(size*0.38), color, -1)
    else: 
        pts = np.array([[size//2, int(size*0.15)],
                        [int(size*0.18), int(size*0.85)],
                        [int(size*0.82), int(size*0.85)]], np.int32)
        cv2.fillConvexPoly(patch, pts, color)

    return patch

def place_odlc(bg, odlc):
    h, w, _ = bg.shape
    ph, pw, _ = odlc.shape
    x = random.randint(0, w - pw)
    y = random.randint(0, h - ph)

    roi = bg[y:y+ph, x:x+pw]
    mask = cv2.cvtColor(odlc, cv2.COLOR_BGR2GRAY) > 0
    roi[mask] = odlc[mask]
    return x, y, x+pw, y+ph

def bbox_to_yolo(x1, y1, x2, y2):
    xc = (x1 + x2)/2 / 1920
    yc = (y1 + y2)/2 / 1080
    w = (x2 - x1) / 1920
    h = (y2 - y1) / 1080
    return float(np.clip(xc,0,1)), float(np.clip(yc,0,1)), float(np.clip(w,0,1)), float(np.clip(h,0,1))


def generate_dataset(out_root="synthetic_odlc", blank_ratio=0.2, train_ratio=0.8):

    for split in ["train", "val"]:
        os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", split), exist_ok=True)

    for i in range(1000):
        split = "train" if random.random() < train_ratio else "val"

        bg = make_background()
        label_file = os.path.join(out_root, "labels", split, f"{i:06d}.txt")
        img_file = os.path.join(out_root, "images", split, f"{i:06d}.jpg")

        if random.random() < blank_ratio:
            cv2.imwrite(img_file, bg)
            open(label_file, "w").close()
            continue

        odlc = make_odlc(size=random.choice([160, 192, 224, 256]))
        x1, y1, x2, y2 = place_odlc(bg, odlc)
        xc, yc, w, h = bbox_to_yolo(x1, y1, x2, y2)
        cv2.imwrite(img_file, bg)
        with open(label_file, "w") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print(f"1000 images in '{out_root}'")


if __name__ == "__main__":
    generate_dataset(out_root="synthetic_odlc", blank_ratio=0.2, train_ratio=0.8)
