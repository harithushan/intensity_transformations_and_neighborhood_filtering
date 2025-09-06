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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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

def vibrance_s_curve(x, a: float, sigma: float = 70.0):
    # x in [0,255]; returns mapped intensity with a Gaussian bump around 128
    return np.minimum(x + a * 128.0 * np.exp(-((x - 128.0)**2)/(2*sigma**2)), 255.0)


def sobel_filter_2d(gray):
    gray = ensure_gray(gray).astype(np.float32)
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    gx = cv2.filter2D(gray, -1, kx)
    gy = cv2.filter2D(gray, -1, ky)
    mag = np.sqrt(gx*gx + gy*gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag, gx.astype(np.float32), gy.astype(np.float32)

def sobel_manual(gray):
    gray = ensure_gray(gray).astype(np.float32)
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    ky = np.array([[1,2,1],[0,0,0],[-1, -2, -1]], dtype=np.float32)
    gx = conv2(gray, kx)
    gy = conv2(gray, ky)
    mag = np.sqrt(gx*gx + gy*gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag, gx, gy

def conv2(img, kernel):
    kh, kw = kernel.shape
    pad_y = kh//2
    pad_x = kw//2
    padded = np.pad(img, ((pad_y,pad_y),(pad_x,pad_x)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            region = padded[y:y+kh, x:x+kw]
            out[y,x] = np.sum(region * kernel[::-1, ::-1])
    return out

def sobel_separable(gray):

    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    kx1 = np.array([1, 2, 1], dtype=np.float32).reshape(-1, 1)
    kx2 = np.array([1, 0, -1], dtype=np.float32).reshape(1, -1)
    ky1 = np.array([1, 0, -1], dtype=np.float32).reshape(-1, 1)
    ky2 = np.array([1, 2, 1], dtype=np.float32).reshape(1, -1)

    gx = cv2.filter2D(gray, cv2.CV_32F, kx1)
    gx = cv2.filter2D(gx, cv2.CV_32F, kx2)

    gy = cv2.filter2D(gray, cv2.CV_32F, ky1)
    gy = cv2.filter2D(gy, cv2.CV_32F, ky2)

    mag = np.sqrt(gx**2 + gy**2)
    mag = np.clip(mag, 0, 255).astype(np.uint8)

    return mag, gx, gy
