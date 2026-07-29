import cv2
import numpy as np

# --- CONFIGURATION ---
# Gamma > 1.0 darkens the image (brings out overexposed plasma edges).
# Gamma < 1.0 brightens the image.
# Try values between 1.5 and 2.5 depending on your lighting.
GAMMA_VALUE = 2.0


def adjust_gamma(image, gamma=1.0):
    """
    Applies non-linear gamma correction to decrease brightness and boost contrast
    in overexposed clear fluids using a high-speed Lookup Table (LUT).
    """
    invGamma = 1.0 / gamma
    # Pre-calculate the mapping for pixel values 0-255
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def main():
    # Replace with the exact IP address shown on your phone app screen
    phone_ip_url = "http://100.83.15.220:8080/video"

    cap = cv2.VideoCapture(phone_ip_url)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Force High Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Aggressive Sharpening Kernel
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

    # Warm up camera
    for _ in range(30):
        ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        return

    # Apply Gamma Correction THEN Sharpening to the setup frame
    darkened_frame = adjust_gamma(frame, gamma=GAMMA_VALUE)
    processed_setup_frame = cv2.filter2D(darkened_frame, -1, sharpen_kernel)

    print("\n" + "=" * 50)
    print("      --- SETUP INSTRUCTIONS ---")
    print(f"Current Gamma (Darkness) Multiplier: {GAMMA_VALUE}")
    print("1. Draw a box around each tube.")
    print("2. IMPORTANT: Include a little bit of the empty space")
    print("   above the plasma and the clay below the RBCs.")
    print("3. Press SPACE or ENTER to confirm the box.")
    print("4. Repeat for all tubes, then press ESC to start.")
    print("=" * 50 + "\n")

    rois = cv2.selectROIs("Select Tubes (SPACE to confirm, ESC to finish)", processed_setup_frame, fromCenter=False,
                          showCrosshair=True)
    cv2.destroyWindow("Select Tubes (SPACE to confirm, ESC to finish)")

    if len(rois) == 0:
        print("No tubes selected. Exiting...")
        cap.release()
        return

    print(f"Locked {len(rois)} tubes. Press 'q' to exit the analyzer.")

    tracked_tubes = {
        i: {
            "box": rois[i],
            "samples_top": [],
            "samples_mid": [],
            "samples_bot": []
        } for i in range(len(rois))
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Darken to reveal plasma
        darkened_frame = adjust_gamma(frame, gamma=GAMMA_VALUE)
        # 2. Sharpen edges
        processed_frame = cv2.filter2D(darkened_frame, -1, sharpen_kernel)
        # 3. Convert to HSV for gradient analysis
        hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)

        for i, data in tracked_tubes.items():
            x, y, w, h = data["box"]

            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.putText(
                processed_frame, f"Tube {i + 1}", (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

            tube_roi = hsv[y: y + h, x: x + w]

            if tube_roi.size > 0:
                row_hue = np.mean(tube_roi[:, :, 0], axis=1)
                row_sat = np.mean(tube_roi[:, :, 1], axis=1)
                row_val = np.mean(tube_roi[:, :, 2], axis=1)

                if len(row_val) > 15:
                    grad_h = np.gradient(row_hue)
                    grad_s = np.gradient(row_sat)
                    grad_v = np.gradient(row_val)

                    combined_gradient = np.abs(grad_v) + (0.5 * np.abs(grad_s)) + (0.5 * np.abs(grad_h))
                    smoothed_grad = np.convolve(combined_gradient, np.ones(5) / 5, mode='same')

                    margin = int(len(smoothed_grad) * 0.05)
                    inner_grad = smoothed_grad[margin:-margin]

                    temp_grad = inner_grad.copy()
                    peaks = []

                    for _ in range(3):
                        if len(temp_grad) == 0 or np.max(temp_grad) < 1.0:
                            break

                        local_p = np.argmax(temp_grad)
                        peaks.append(local_p)

                        suppress_start = max(0, local_p - 15)
                        suppress_end = min(len(temp_grad), local_p + 15)
                        temp_grad[suppress_start:suppress_end] = 0

                    if len(peaks) == 3:
                        peaks.sort()

                        raw_top = y + margin + peaks[0]
                        raw_mid = y + margin + peaks[1]
                        raw_bot = y + margin + peaks[2]

                        data["samples_top"].append(raw_top)
                        data["samples_mid"].append(raw_mid)
                        data["samples_bot"].append(raw_bot)

                        if len(data["samples_top"]) > 30:
                            data["samples_top"].pop(0)
                            data["samples_mid"].pop(0)
                            data["samples_bot"].pop(0)

            if len(data["samples_top"]) >= 10:
                final_top = int(np.median(data["samples_top"]))
                final_mid = int(np.median(data["samples_mid"]))
                final_bot = int(np.median(data["samples_bot"]))

                total_fluid_height = final_bot - final_top
                rbc_height = final_bot - final_mid

                pcv_percentage = 0.0
                if total_fluid_height > 0:
                    pcv_percentage = (rbc_height / total_fluid_height) * 100

                cv2.line(processed_frame, (x - 6, final_top), (x + w + 6, final_top), (0, 255, 255), 2)
                cv2.putText(processed_frame, "Plasma Top", (x + w + 8, final_top + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 255), 1)

                cv2.line(processed_frame, (x - 6, final_mid), (x + w + 6, final_mid), (255, 255, 0), 2)
                cv2.putText(processed_frame, "Buffy Coat", (x + w + 8, final_mid + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 0), 1)

                cv2.line(processed_frame, (x - 6, final_bot), (x + w + 6, final_bot), (0, 0, 255), 2)
                cv2.putText(processed_frame, "Clay Base", (x + w + 8, final_bot + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 255), 1)

                cv2.putText(
                    processed_frame, f"PCV: {pcv_percentage:.1f}%",
                    (x, final_bot + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

        cv2.imshow("Gamma-Corrected PCV Analyzer", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()