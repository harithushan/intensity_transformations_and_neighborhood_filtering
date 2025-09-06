
import cv2, numpy as np, os
from utils import load_color, save_img
os.makedirs("docs/q8_images/output", exist_ok=True)
IN_PATH = "docs/q8_images/input/daisy.jpg"
def main():
    img = load_color(IN_PATH)
    h,w = img.shape[:2]
    mask = np.zeros((h,w), np.uint8)
    rect = (int(0.1*w), int(0.1*h), int(0.8*w), int(0.8*h))
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
    fg = cv2.bitwise_and(img, img, mask=mask2)
    bg = cv2.bitwise_and(img, img, mask=(255-mask2))
    save_img("docs/q8_images/output/q8_mask.png", mask2)
    save_img("docs/q8_images/output/q8_fg.png", fg)
    save_img("docs/q8_images/output/q8_bg.png", bg)
    blurred = cv2.GaussianBlur(img, (31,31), 0)
    enhanced = np.where(mask2[...,None]>0, img, blurred)
    save_img("docs/q8_images/output/q8_original.png", img)
    save_img("docs/q8_images/output/q8_enhanced.png", enhanced)
    with open("docs/q8_images/output/q8_explanation.txt","w",encoding="utf-8") as f:
        f.write("Dark rim occurs due to uncertain boundary labeling and blur mixing darker background near edges.")
    print("Q8 done.")
if __name__ == "__main__":
    main()
