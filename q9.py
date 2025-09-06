
import cv2, numpy as np, os, json
from utils import (
    load_gray, 
    save_img, 
    otsu_threshold, 
    morph_open_close
    )
os.makedirs("docs/q9_images/output", exist_ok=True)

# Gaussian noise corrupted.
IN_A = "docs/q9_images/input/rice_8a.png"
#  Salt-and-pepper noise corrected
IN_B = "docs/q9_images/input/rice_8b.png"
def process(path, tag):
    gray = load_gray(path)
    if "gauss" in path:
        den = cv2.GaussianBlur(gray, (5,5), 0)
    else:
        den = cv2.medianBlur(gray, 3)
    th = otsu_threshold(den)
    if np.mean(den[th>0]) < np.mean(den[th==0]):
        th = 255 - th
    clean = morph_open_close(th, open_k=3, close_k=5, iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    count = num_labels - 1
    save_img(f"docs/q9_images/output/q9_{tag}_den.png", den)
    save_img(f"docs/q9_images/output/q9_{tag}_th.png", th)
    save_img(f"docs/q9_images/output/q9_{tag}_clean.png", clean)
    return {"tag": tag, "count": int(count)}
def main():
    results = []
    for path, tag in [(IN_A, "gauss"), (IN_B, "sp")]:
        if os.path.exists(path):
            results.append(process(path, tag))
    with open("docs/q9_images/output/q9_counts.json","w") as f:
        json.dump(results, f, indent=2)
    print("Q9 done.")
if __name__ == "__main__":
    main()
