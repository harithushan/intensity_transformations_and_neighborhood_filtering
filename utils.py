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

def load_color(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def ensure_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def hist_img(img, title, outpath):
    gray = ensure_gray(img)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    plt.figure()
    plt.plot(hist)
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    
    plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close()

    plt.close()

def gamma_correct_Lab(bgr, gamma: float):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)
    Ln = L / 255.0
    Lc = np.power(np.clip(Ln, 0, 1), gamma) * 255.0
    lab_corr = cv2.merge((Lc, a, b)).astype(np.uint8)
    out = cv2.cvtColor(lab_corr, cv2.COLOR_LAB2BGR)
    return out