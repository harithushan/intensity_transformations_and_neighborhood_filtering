
import cv2, numpy as np, os
from utils import load_color, save_img

IN_PATH = "docs/q5_images/input/jeniffer.jpg"
def main():
    bgr = load_color(IN_PATH)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    os.makedirs("docs/q5_images/output", exist_ok=True)
    cv2.imwrite("docs/q5_images/output/q5_h.jpg", h)
    cv2.imwrite("docs/q5_images/output/q5_s.jpg", s)
    cv2.imwrite("docs/q5_images/output/q5_v.jpg", v)

    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite("docs/q5_images/output/q5_mask.jpg", mask)
    fg = cv2.bitwise_and(v, v, mask=mask)
    hist = cv2.calcHist([fg],[0],mask,[256],[0,256]).ravel()
    cdf = np.cumsum(hist)
    cdf_norm = (cdf - cdf.min())/(cdf.max()-cdf.min()+1e-8) * 255.0
    cdf_norm = cdf_norm.clip(0,255).astype(np.uint8)
    v_eq = v.copy()
    v_eq[mask>0] = cdf_norm[v[mask>0]]
    v_final = v_eq
    hsv2 = cv2.merge((h, s, v_final))
    out = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    save_img("docs/q5_images/output/q5_original.jpg", bgr)
    save_img("docs/q5_images/output/q5_result.jpg", out)
    print("Q5 done.")
if __name__ == "__main__":
    main()