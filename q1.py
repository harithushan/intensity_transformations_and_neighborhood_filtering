
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    load_gray, save_img, 
    plot_transfer, 
    apply_lut, 
    intensity_map_from_points
)

IN_PATH = "docs/q1_images/input/emma.jpg"
control_points = [(0,0),(50, 50),(50, 100),(150, 255),(150, 150),(255,255)]
def main():
    img = load_gray(IN_PATH)
    lut = intensity_map_from_points(control_points)
    mapped = apply_lut(img, lut)
    x = np.arange(256); y = lut
    plot_transfer(x, y, "Q1 Intensity Transformation", "docs/q1_images/output/emma_intensity_transformation.jpg")
    save_img("docs/q1_images/output/emma_original.jpg", img)
    save_img("docs/q1_images/output/emma_mapped.jpg", mapped)
    print("Q1 done.")
if __name__ == "__main__":
    main()
