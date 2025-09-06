
import cv2, numpy as np, os, json, argparse
from utils import (
    load_color, 
    save_img, 
    otsu_threshold, 
    morph_open_close
    )
os.makedirs("docs/q10_images/output", exist_ok=True)

IN_PATH = "docs/q10_images/input/sapphire.jpg"
def compute_real_areas_px2_to_mm2(area_px, pixel_size_mm, object_distance_mm, focal_length_mm):
    gsd = (pixel_size_mm * object_distance_mm) / focal_length_mm
    return area_px * (gsd ** 2)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixel_size_mm", type=float, default=0.004)
    parser.add_argument("--f_mm", type=float, default=8.0)
    parser.add_argument("--Z_mm", type=float, default=480.0)
    args = parser.parse_args()
    bgr = load_color(IN_PATH)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    den = cv2.GaussianBlur(gray, (5,5), 0)
    th = otsu_threshold(den)
    if np.mean(den[th>0]) > np.mean(den[th==0]):
        th = 255 - th
    clean = morph_open_close(th, open_k=3, close_k=9, iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    areas_px = [int(s[cv2.CC_STAT_AREA]) for i,s in enumerate(stats) if i!=0]
    areas_mm2 = [compute_real_areas_px2_to_mm2(a, args.pixel_size_mm, args.Z_mm, args.f_mm) for a in areas_px]
    save_img("docs/q10_images/output/q10_original.png", bgr)
    save_img("docs/q10_images/output/q10_mask.png", th)
    save_img("docs/q10_images/output/q10_filled.png", clean)
    with open("docs/q10_images/output/q10_areas.json","w") as f:
        json.dump({"areas_px": areas_px, "areas_mm2": areas_mm2,
                   "pixel_size_mm": args.pixel_size_mm, "f_mm": args.f_mm, "Z_mm": args.Z_mm}, f, indent=2)
    print("Q10 done.")
if __name__ == "__main__":
    main()
