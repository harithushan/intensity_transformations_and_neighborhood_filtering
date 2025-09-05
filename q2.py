import os
import cv2
import numpy as np
from utils import (
    load_gray, 
    save_img, 
    plot_transfer, 
    apply_lut, 
    intensity_map_from_points
)
IN_PATH = "docs/q2_images/input/brain_proton_density_slice.png"
def accentuate_range(img, low, high):
    cps = [(0,0),(low, low//2),(high, 255),(255,255)]
    lut = intensity_map_from_points(cps)
    return apply_lut(img, lut), lut
def main():
    img = load_gray(IN_PATH)
    wm_out, wm_lut = accentuate_range(img, low=120, high=200)
    gm_out, gm_lut = accentuate_range(img, low=80,  high=150)
    plot_transfer(np.arange(256), wm_lut, "Q2(a) White-matter LUT", "docs/q2_images/output/q2_wm_transfer.png")
    plot_transfer(np.arange(256), gm_lut, "Q2(b) Gray-matter LUT", "docs/q2_images/output/q2_gm_transfer.png")
    save_img("docs/q2_images/output/q2_original.png", img)
    save_img("docs/q2_images/output/q2_white_matter.png", wm_out)
    save_img("docs/q2_images/output/q2_gray_matter.png", gm_out)
    print("Q2 done.")
if __name__ == "__main__":
    main()
