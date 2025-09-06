
import cv2, numpy as np, os
from utils import (
    load_gray, 
    save_img, 
    sobel_filter_2d, 
    sobel_manual, 
    sobel_separable
    )
os.makedirs("docs/q6_images/output", exist_ok=True)
IN_PATH = "docs/q6_images/input/einstein.png"
def main():
    img = load_gray(IN_PATH)
    mag_a, gx_a, gy_a = sobel_filter_2d(img)
    mag_b, gx_b, gy_b = sobel_manual(img)
    mag_c, gx_c, gy_c = sobel_separable(img)
    save_img("docs/q6_images/output/q6_orig.png", img)
    save_img("docs/q6_images/output/q6_a_mag.png", mag_a)
    save_img("docs/q6_images/output/q6_b_mag.png", mag_b)
    save_img("docs/q6_images/output/q6_c_mag.png", mag_c)
    print("Q6 done.")
if __name__ == "__main__":
    main()
