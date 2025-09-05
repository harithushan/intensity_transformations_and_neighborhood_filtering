
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def save_img(path: str, img):
    cv2.imwrite(path, img)

def apply_lut(gray, lut):
    return cv2.LUT(gray, lut)


def plot_transfer(x, y, title, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Input intensity")
    plt.ylabel("Output intensity")
    plt.xlim([0,255])
    plt.ylim([0,255])
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close()


def intensity_map_from_points(points):
    pts = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lut = np.interp(np.arange(256), xs, ys).astype(np.uint8)
    return lut

os.makedirs("out", exist_ok=True)
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
