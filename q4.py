import cv2, numpy as np
from utils import load_color, save_img, plot_transfer, vibrance_s_curve

IN_PATH = "docs/q4_images/input/spider.png"
A = 0.6
SIGMA = 70.0

def main():
    bgr = load_color(IN_PATH)

    # Convert to HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # Save H, S, V planes
    save_img("docs/q4_images/output/q4_spider_h.png", h.astype(np.uint8))
    save_img("docs/q4_images/output/q4_spider_s.png", s.astype(np.uint8))
    save_img("docs/q4_images/output/q4_spider_v.png", v.astype(np.uint8))

    # Build vibrance transform
    x = np.arange(256, dtype=np.float32)
    y = vibrance_s_curve(x, a=A, sigma=SIGMA).astype(np.float32)
    plot_transfer(x, y, f"Q4 Vibrance Transform (a={A:.2f}, sigma={SIGMA})",
                  "docs/q4_images/output/q4_spider_transfer.png")

    # Apply vibrance to saturation plane
    s2 = np.interp(s, x, y).astype(np.float32)

    # Recombine HSV and convert back to BGR
    hsv2 = cv2.merge((h, s2, v))
    out = cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2BGR)
    save_img("docs/q4_images/output/q4_spider_original.png", bgr)
    save_img("docs/q4_images/output/q4_spider_vibrance.png", out)

    print(f"Q4 done with a={A}.")

if __name__ == "__main__":
    main()
