
import os
import random
import numpy as np
import cv2


def make_background(h=1080, w=1920):
    base = np.full((h, w, 3), 200, dtype=np.uint8)
    if np.random.rand() < 0.5:
        grad = np.tile(np.linspace(0, 55, w, dtype=np.uint8), (h, 1))
    else:
        grad = np.tile(np.linspace(0, 55, h, dtype=np.uint8), (w, 1)).T
    grad = cv2.merge([grad, grad, grad])
    bg = cv2.add(base, grad)
    return bg

def render_odlc(size=200): #odlc is either circle, square or triangle
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    shape = random.choice(['circle', 'square', 'triangle'])
    color = tuple(int(c) for c in np.random.randint(50, 220, 3))

    if shape == 'square':
        pad = int(size*0.15)
        cv2.rectangle(patch, (pad,pad), (size-pad,size-pad), color, -1)
    elif shape == 'circle':
        cv2.circle(patch, (size//2, size//2), int(size*0.38), color, -1)
    else:  
        pts = np.array([[size//2,int(size*0.15)],[int(size*0.18),int(size*0.85)],[int(size*0.82),int(size*0.85)]], np.int32)
        cv2.fillConvexPoly(patch, pts, color)

    letter = chr(random.randint(ord('A'), ord('Z')))
    scale = size / 120
    thickness = max(2, size//80)
    (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    org = ((size-tw)//2, (size+th)//2 - 5)
    txt_color = (255,255,255) if sum(color) < 380 else (10,10,10)
    cv2.putText(patch, letter, org, cv2.FONT_HERSHEY_SIMPLEX, scale, txt_color, thickness)
    return patch

def place_patch(bg, patch):
    """Randomly place patch on background and return bounding box in pixels."""
    h, w, _ = bg.shape
    ph, pw, _ = patch.shape
    max_x = w - pw
    max_y = h - ph
    offx = random.randint(0, max_x)
    offy = random.randint(0, max_y)
    roi = bg[offy:offy+ph, offx:offx+pw]
    mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) > 0
    roi[mask] = patch[mask]
    x1, y1 = offx, offy
    x2, y2 = offx + pw, offy + ph
    return x1, y1, x2, y2

def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert pixel bbox to YOLO normalized format."""
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return float(np.clip(xc,0,1)), float(np.clip(yc,0,1)), float(np.clip(w,0,1)), float(np.clip(h,0,1))

def generate_dataset(out_root="synthetic_odlc", n_train=500, n_val=100, img_w=1920, img_h=1080, neg_ratio=0.2):
    random.seed(42)
    np.random.seed(42)

    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = os.path.join(out_root, split, "images")
        lbl_dir = os.path.join(out_root, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(n):
            bg = make_background(img_h, img_w)
            if random.random() < neg_ratio:
                noise = np.random.normal(0,3,bg.shape).astype(np.int16)
                bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                img_path = os.path.join(img_dir, f"{i:06d}.jpg")
                lbl_path = os.path.join(lbl_dir, f"{i:06d}.txt")
                cv2.imwrite(img_path, bg)
                open(lbl_path, "w").close()
                continue

            patch = render_odlc(size=random.choice([160,192,224,256]))
            x1,y1,x2,y2 = place_patch(bg, patch)
            xc, yc, w, h = bbox_to_yolo(x1,y1,x2,y2,img_w,img_h)

            img_path = os.path.join(img_dir, f"{i:06d}.jpg")
            lbl_path = os.path.join(lbl_dir, f"{i:06d}.txt")
            cv2.imwrite(img_path, bg)
            with open(lbl_path, "w") as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print(f"Dataset generated at '{out_root}'")

if __name__ == "__main__":
    generate_dataset(out_root="synthetic_odlc", n_train=500, n_val=100)
