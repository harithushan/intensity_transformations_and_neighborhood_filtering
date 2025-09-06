
import os
import cv2
import numpy as np
from utils import (
    load_gray, save_img, 
    plot_transfer, 
    apply_lut
)

IN_PATH = "docs/q1_images/input/emma.jpg"

# Fixed control points
control_points = [(0,0),(50,100),(150,150),(200,220),(255,255)]

def build_lut(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lut = np.interp(np.arange(256), xs, ys).astype(np.uint8)
    return lut

def main():
    img = load_gray(IN_PATH)
    lut = build_lut(control_points)
    mapped = apply_lut(img, lut)


    x = np.arange(256); y = lut
    plot_transfer(x, y, "Q1 Intensity Transformation", 
                  "docs/q1_images/output/emma_intensity_transformation.jpg")

    save_img("docs/q1_images/output/emma_original.jpg", img)
    save_img("docs/q1_images/output/emma_mapped.jpg", mapped)

    print("Q1 done.")

if __name__ == "__main__":
    main()
