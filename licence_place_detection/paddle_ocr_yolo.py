import cv2
from paddleocr import PaddleOCR
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Load the YOLO model and the original image
model = YOLO(r"C:\Users\Tvrtko\Desktop\licence_place_dataset\runs\detect\train2\weights\last.pt")
img = cv2.imread('./train/images/3572232-bilbasen-suzuki_swift_2008-4_jpg.rf.90e12144a99c10116beff823d5cdb7b3.jpg')

# Run YOLO prediction
results = model(img)

def is_valid_character(char):
    # Define acceptable character patterns including special characters
    valid_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return char in valid_characters

# Loop through detected objects
for result in results:
    for bbox in result.boxes.xyxy:  # bbox in format (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, bbox)

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box, thickness = 2

        # Crop the license plate area
        plate_region = img[y1:y2, x1:x2]

        # Rescale the plate region if it's too small
        height, width = plate_region.shape[:2]
        if height < 50 or width < 150:  # Arbitrary threshold; adjust based on your data
            plate_region = cv2.resize(plate_region, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

        # Convert to RGB for PaddleOCR (expects RGB)
        plate_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB)

        # Perform OCR on the license plate region
        ocr_results = ocr.ocr(plate_region, cls=True)

        # Extract and filter the recognized text
        recognized_text = ''
        for line in ocr_results:
            for word_info in line:
                word = word_info[1][0]  # Extract the recognized word
                word = ''.join(char for char in word if is_valid_character(char))  # Filter invalid characters
                recognized_text += word + ' '

        recognized_text = recognized_text.strip()  # Remove trailing spaces

        # Print the recognized text
        print(f"Detected License Plate Text: {recognized_text}")

        # Draw the recognized text on the image
        cv2.putText(img, recognized_text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)

# Display the image with the bounding box and recognized text
cv2.imshow("Detected License Plate", img)
cv2.waitKey(0)  # Wait for a key press to close the image window
cv2.destroyAllWindows()

# Save the image with the bounding box and recognized text (optional)
cv2.imwrite("output_with_bounding_box_and_text.jpg", img)