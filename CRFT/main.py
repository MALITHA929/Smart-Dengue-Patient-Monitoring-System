import cv2
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
REFERENCE_IMAGE_PATH = "original.jpeg"
TEST_IMAGES_FOLDER = "test_images"

# -----------------------------
# FUNCTIONS
# -----------------------------

def select_roi(image):
    print("Select the nail region and press ENTER or SPACE")
    roi = cv2.selectROI("Select Nail ROI", image, showCrosshair=True)
    cv2.destroyAllWindows()
    return roi  # (x, y, w, h)

def extract_mean_lab(image, roi):
    x, y, w, h = roi
    nail_region = image[y:y+h, x:x+w]

    lab = cv2.cvtColor(nail_region, cv2.COLOR_BGR2LAB)
    mean_lab = np.mean(lab.reshape(-1, 3), axis=0)

    return mean_lab

def delta_e(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)

def similarity_percentage(delta):
    # Tunable value: 100 is a reasonable max difference
    similarity = max(0, 100 - delta)
    return similarity

# -----------------------------
# MAIN
# -----------------------------

# Load reference image
ref_img = cv2.imread(REFERENCE_IMAGE_PATH)
if ref_img is None:
    raise FileNotFoundError("Reference image not found!")

# Select ROI
roi = select_roi(ref_img)

# Extract reference LAB color
ref_lab = extract_mean_lab(ref_img, roi)

print("\n--- CRT COLOR RECOVERY RESULTS ---\n")

# Process test images
for file in sorted(os.listdir(TEST_IMAGES_FOLDER)):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(TEST_IMAGES_FOLDER, file)
        img = cv2.imread(path)

        if img is None:
            continue

        test_lab = extract_mean_lab(img, roi)
        dE = delta_e(ref_lab, test_lab)
        similarity = similarity_percentage(dE)

        status = "Normal"
        if similarity < 60:
            status = "Low recovery"
        elif similarity < 80:
            status = "Moderate recovery"

        print(f"{file}")
        print(f"  Delta-E      : {dE:.2f}")
        print(f"  Similarity % : {similarity:.2f}")
        print(f"  Status       : {status}\n")
