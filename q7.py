import cv2
import numpy as np
import os
import json
from utils import (
    load_color, 
    save_img, 
    nearest_neighbor_zoom, 
    bilinear_zoom, 
    normalized_ssd
)



#List of small-large image pairs
SMALLS = [
    ("docs/q7_images/input/im01small.png", "docs/q7_images/input/im01.png"),
    ("docs/q7_images/input/im02small.png", "docs/q7_images/input/im02.png"),
    ("docs/q7_images/input/im03small.png", "docs/q7_images/input/im03.png"),
    ("docs/q7_images/input/taylor_small.jpg", "docs/q7_images/input/taylor.jpg"),
    ("docs/q7_images/input/taylor_very_small.jpg", "docs/q7_images/input/taylor.jpg")
]

SCALE = 4.0

def process_pair(small_path, large_path, tag):
    small = load_color(small_path)
    large = load_color(large_path)

    nn = nearest_neighbor_zoom(small, SCALE)
    bl = bilinear_zoom(small, SCALE)

    # Resize to exact large image size
    h, w = large.shape[:2]
    nn = cv2.resize(nn, (w, h), interpolation=cv2.INTER_NEAREST)
    bl = cv2.resize(bl, (w, h), interpolation=cv2.INTER_LINEAR)

    # Compute normalized SSD
    ssd_nn = normalized_ssd(large, nn)
    ssd_bl = normalized_ssd(large, bl)

    # Save outputs
    save_img(f"docs/q7_images/output/q7_{tag}_large.png", large)
    save_img(f"docs/q7_images/output/q7_{tag}_nnx4.png", nn)
    save_img(f"docs/q7_images/output/q7_{tag}_blx4.png", bl)

    return {"pair": tag, "ssd_nn": float(ssd_nn), "ssd_bilinear": float(ssd_bl)}

def main():
    results = []
    for i, (s, l) in enumerate(SMALLS, start=1):
        if not (os.path.exists(s) and os.path.exists(l)):
            print(f"Skipping pair {i}: files not found.")
            continue
        results.append(process_pair(s, l, f"pair{i}"))

    # Save SSD results
    with open("docs/q7_images/output/q7_ssd.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Processing done. Results saved in 'docs/q7_images/output/' folder.")

if __name__ == "__main__":
    main()