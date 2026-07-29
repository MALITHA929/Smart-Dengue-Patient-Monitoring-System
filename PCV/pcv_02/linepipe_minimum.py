import cv2
import numpy as np


def main():
  cap = cv2.VideoCapture(0)

  if not cap.isOpened():
    print("Error: Could not open webcam.")
    return

  print("Press 'q' to exit the video stream.")

  tracked_tubes = {}
  alpha_box = 0.2  # Smoothness for the bounding box
  alpha_interface = (
      0.08  # Much heavier smoothing for the interface to stop jitter
  )

  while True:
    ret, frame = cap.read()

    if not ret:
      print("Error: Can't receive frame. Exiting ...")
      break

    # 1. Apply Unsharp Masking for maximum clarity
    blurred = cv2.GaussianBlur(frame, (0, 0), 3)
    sharpened_frame = cv2.addWeighted(frame, 1.8, blurred, -0.8, 0)

    gray = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2HSV)

    # 2. Detect vertical tube profiles via edges
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    current_detected_tubes = []

    for cnt in contours:
      x, y, w, h = cv2.boundingRect(cnt)
      # Filter for long, thin vertical structures
      if h > 100 and w < 25:
        current_detected_tubes.append((x, y, w, h))

    updated_tubes = {}
    for x, y, w, h in current_detected_tubes:
      matched_id = None
      for tid, data in tracked_tubes.items():
        ox, oy, _, _ = data["box"]
        if abs(x - ox) < 30 and abs(y - oy) < 30:
          matched_id = tid
          break

      if matched_id is None:
        matched_id = len(tracked_tubes) + len(updated_tubes)

      # Smooth bounding box coordinates
      if matched_id in tracked_tubes:
        ox, oy, ow, oh = tracked_tubes[matched_id]["box"]
        smooth_x = int(alpha_box * x + (1 - alpha_box) * ox)
        smooth_y = int(alpha_box * y + (1 - alpha_box) * oy)
        smooth_w = int(alpha_box * w + (1 - alpha_box) * ow)
        smooth_h = int(alpha_box * h + (1 - alpha_box) * oh)
      else:
        smooth_x, smooth_y, smooth_w, smooth_h = x, y, w, h

      # Draw tube box
      cv2.rectangle(
          sharpened_frame,
          (smooth_x, smooth_y),
          (smooth_x + smooth_w, smooth_y + smooth_h),
          (255, 255, 255),
          1,
      )

      # 3. Advanced Multi-Channel Interface Analysis inside the tube ROI
      tube_roi = hsv[
          smooth_y : smooth_y + smooth_h, smooth_x : smooth_x + smooth_w
      ]
      interface_global_y = smooth_y + int(smooth_h / 2)  # Fallback middle

      if tube_roi.size > 0:
        row_hue = np.mean(tube_roi[:, :, 0], axis=1)
        row_sat = np.mean(tube_roi[:, :, 1], axis=1)
        row_val = np.mean(tube_roi[:, :, 2], axis=1)

        if len(row_val) > 15:
          grad_h = np.gradient(row_hue)
          grad_s = np.gradient(row_sat)
          grad_v = np.gradient(row_val)

          combined_gradient = (
              np.abs(grad_v) + (0.5 * np.abs(grad_s)) + (0.5 * np.abs(grad_h))
          )

          margin = int(len(combined_gradient) * 0.1)
          if len(combined_gradient) > (2 * margin + 5):
            inner_gradient = combined_gradient[margin:-margin]
            interface_local_y = np.argmax(inner_gradient) + margin
            interface_global_y = smooth_y + interface_local_y

          # Apply specialized heavy smoothing to the interface line
          if matched_id in tracked_tubes:
            prev_iy = tracked_tubes[matched_id]["interface_y"]
            # Only update if the change is significant enough, otherwise lock it down
            if abs(interface_global_y - prev_iy) > 1:
              interface_global_y = int(
                  alpha_interface * interface_global_y
                  + (1 - alpha_interface) * prev_iy
              )
            else:
              interface_global_y = prev_iy

      # Draw precise interface marker line (Cyan)
      cv2.line(
          sharpened_frame,
          (smooth_x - 6, interface_global_y),
          (smooth_x + smooth_w + 6, interface_global_y),
          (0, 255, 255),
          2,
      )

      cv2.putText(
          sharpened_frame,
          "Interface",
          (smooth_x + smooth_w + 8, interface_global_y + 4),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.4,
          (0, 255, 255),
          1,
      )

      updated_tubes[matched_id] = {
          "box": [smooth_x, smooth_y, smooth_w, smooth_h],
          "interface_y": interface_global_y,
      }

    tracked_tubes = updated_tubes

    cv2.imshow("Stable Interface Detector", sharpened_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()