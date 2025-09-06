import cv2
import numpy as np
import os
from utils import load_color, save_img

IN_PATH = "docs/q5_images/input/jeniffer.jpg"

def main():
    # (a) Load and split into HSV
    bgr = load_color(IN_PATH)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    os.makedirs("docs/q5_images/output", exist_ok=True)
    save_img("docs/q5_images/output/q5_h.jpg", h)
    save_img("docs/q5_images/output/q5_s.jpg", s)
    save_img("docs/q5_images/output/q5_v.jpg", v)

    # (b) Better threshold for portrait - use manual threshold instead of Otsu
    _, mask = cv2.threshold(v, 40, 255, cv2.THRESH_BINARY)  # Lower threshold for dark background
    
    # Clean up mask with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    save_img("docs/q5_images/output/q5_mask.jpg", mask)

    # (c) Foreground histogram equalization
    fg_pixels = v[mask > 0]
    if len(fg_pixels) > 0:
        # Direct histogram equalization on foreground pixels
        v_eq = v.copy()
        v_eq[mask > 0] = cv2.equalizeHist(fg_pixels.reshape(-1, 1)).flatten()
    else:
        v_eq = v.copy()

    # (d) Blend foreground and background
    v_final = np.where(mask > 0, v_eq, v)

    # Recombine and convert back to BGR
    hsv_final = cv2.merge((h, s, v_final))
    out = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)

    save_img("docs/q5_images/output/q5_original.jpg", bgr)
    save_img("docs/q5_images/output/q5_result.jpg", out)

    print("Q5 done.")

if __name__ == "__main__":
    main()