from utils import (load_color, 
                   save_img, 
                   hist_img, 
                   gamma_correct_Lab
                   )
IN_PATH = "docs/q3_images/input/highlights_and_shadows.jpg"
GAMMA = 0.8
def main():
    bgr = load_color(IN_PATH)
    out = gamma_correct_Lab(bgr, GAMMA)
    save_img("docs/q3_images/output/q3_original.jpg", bgr)
    save_img("docs/q3_images/output/q3_gamma.jpg", out)
    hist_img(bgr, "Q3 Original Histogram (Gray)", "docs/q3_images/output/q3_hist_orig.jpg")
    hist_img(out, "Q3 Gamma-corrected Histogram (Gray)", "docs/q3_images/output/q3_hist_gamma.jpg")
    print(f"Q3 done with gamma={GAMMA}.")
if __name__ == "__main__":
    main()