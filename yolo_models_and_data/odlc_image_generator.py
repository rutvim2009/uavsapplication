import os, random
import numpy as np
import cv2

def make_background(h=1080, w=1920):
    base = np.random.randint(80, 175, (h, w), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (31,31), 0)
    grad = np.linspace(0, 30, w, dtype=np.uint8)
    bg = cv2.add(base, grad[np.newaxis,:])
    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    return bg

def render_patch(size=256):
    
    patch = np.zeros((size, size, 4), dtype=np.uint8)
    shape = random.choice(['circle','square','triangle'])
    color = tuple(int(c) for c in np.random.randint(30,225,3))
    txt_color = (255,255,255) if sum(color) < 380 else (10,10,10)
    
    if shape == 'square':
        pad = int(size*0.12)
        cv2.rectangle(patch, (pad,pad), (size-pad,size-pad), (*color,255), -1)
    elif shape == 'circle':
        cv2.circle(patch, (size//2,size//2), int(size*0.38), (*color,255), -1)
    else:
        pts = np.array([[size*0.5,size*0.15],[size*0.18,size*0.80],[size*0.82,size*0.80]], np.int32)
        cv2.fillConvexPoly(patch, pts, (*color,255))
    
    letter = chr(random.randint(ord('A'), ord('Z')))
    scale = 2.0
    thickness = max(2, size//80)
    (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    org = ((size-tw)//2, (size+th)//2 - 5)
    cv2.putText(patch, letter, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (*txt_color,255), thickness, cv2.LINE_AA)
    return patch  

def place_and_warp(patch_rgba, out_w, out_h):
    
    bgr = patch_rgba[..., :3].copy()
    alpha = patch_rgba[..., 3].copy()
    
    min_side = min(out_w, out_h)
    scale = np.random.uniform(0.05, 0.18) * (min_side / 1080.0)
    target = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    target_a = cv2.resize(alpha, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = target.shape[:2]
    
    angle = np.random.uniform(-40, 40)
    Mrot = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    target = cv2.warpAffine(target, Mrot, (w,h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    target_a = cv2.warpAffine(target_a, Mrot, (w,h), flags=cv2.INTER_LINEAR, borderValue=0)
    
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    jitter = 0.12 * min(w,h)
    dst = src + np.random.uniform(-jitter, jitter, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(target, M, (w, h), borderValue=(0,0,0))
    warped_a = cv2.warpPerspective(target_a, M, (w, h), borderValue=0)
    
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    max_x = max(1, out_w - w - 1)
    max_y = max(1, out_h - h - 1)
    offx = np.random.randint(0, max_x+1)
    offy = np.random.randint(0, max_y+1)
    canvas[offy:offy+h, offx:offx+w] = warped
    mask[offy:offy+h, offx:offx+w] = warped_a
    
    quad = cv2.perspectiveTransform(src.reshape(1,4,2), M).reshape(4,2)
    quad[:,0] += offx; quad[:,1] += offy
    return canvas, mask, quad

def quad_to_bbox(quad, W, H):
    x1 = max(0, np.min(quad[:,0])); y1 = max(0, np.min(quad[:,1]))
    x2 = min(W-1, np.max(quad[:,0])); y2 = min(H-1, np.max(quad[:,1]))
    return float(x1), float(y1), float(x2), float(y2)

def main(out_root="synthetic_odlc", n_train=2000, n_val=400, img_w=1920, img_h=1080, neg_ratio=0.2, seed=42):
    random.seed(seed); np.random.seed(seed)
    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = os.path.join(out_root, split, "images")
        lab_dir = os.path.join(out_root, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        for i in range(n):
            bg = make_background(img_h, img_w)
            if random.random() < neg_ratio:
                
                if random.random() < 0.5:
                    bg = cv2.GaussianBlur(bg, (3,3), 0)
                noise = np.random.normal(0,4,bg.shape).astype(np.int16)
                bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                im_path = os.path.join(img_dir, f"{i:06d}.jpg")
                lbl_path = os.path.join(lab_dir, f"{i:06d}.txt")
                cv2.imwrite(im_path, bg)
                open(lbl_path, "w").close()
                continue
            patch = render_patch(size=random.choice([160,192,224,256]))
            obj_bgr, obj_mask, quad = place_and_warp(patch, img_w, img_h)
            
            comp = bg.copy()
            m = (obj_mask>0)
            comp[m] = obj_bgr[m]
            
            if random.random() < 0.7:
                comp = cv2.GaussianBlur(comp, (3,3), 0)
            if random.random() < 0.7:
                noise = np.random.normal(0,3,comp.shape).astype(np.int16)
                comp = np.clip(comp.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            x1,y1,x2,y2 = quad_to_bbox(quad, img_w, img_h)
            xc = (x1+x2)/2.0 / img_w
            yc = (y1+y2)/2.0 / img_h
            ww = (x2-x1)/img_w
            hh = (y2-y1)/img_h
            xc, yc = float(np.clip(xc,0,1)), float(np.clip(yc,0,1))
            ww, hh = float(np.clip(ww,0,1)), float(np.clip(hh,0,1))
            im_path = os.path.join(img_dir, f"{i:06d}.jpg")
            lbl_path = os.path.join(lab_dir, f"{i:06d}.txt")
            cv2.imwrite(im_path, comp)
            with open(lbl_path, "w") as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
    print("Dataset created at:", out_root)

if __name__ == "__main__":
    
    main(out_root="synthetic_odlc", n_train=2000, n_val=400)