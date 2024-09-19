import cv2
import pytesseract
import re
import numpy as np
from ultralytics import YOLO

# Specify the path to tesseract.exe (adjust to your installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the YOLO model and the original image
model = YOLO(r"C:\Users\Tvrtko\Desktop\licence_place_dataset\runs\detect\train2\weights\last.pt")
img = cv2.imread(r"C:\Users\Tvrtko\Downloads\adcf1fa8-58cd-42fd-85d0-4cee8ea39284.jpg")

# Run YOLO prediction
results = model(img)

def preprocess_image(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV)
    return binary_img

def is_valid_character(char):
    # Define acceptable character patterns including the special characters
    valid_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789·-'
    return char in valid_characters

# Loop through detected objects
for result in results:
    for bbox in result.boxes.xyxy:  # bbox in format (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, bbox)

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box, thickness = 2

        # Crop the license plate area
        plate_region = img[y1:y2, x1:x2]

        # Preprocess image
        processed_img = preprocess_image(plate_region)

        # Perform OCR on the preprocessed license plate region
        plate_text = pytesseract.image_to_string(processed_img, config='--psm 7')

        # Filter out invalid characters
        filtered_text = ''.join(char for char in plate_text if is_valid_character(char))

        # Print the recognized text
        print(f"Detected License Plate Text: {filtered_text}")

        # Draw the recognized text on the image
        cv2.putText(img, filtered_text.strip(), (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)

# Display the image with the bounding box and recognized text
cv2.imshow("Detected License Plate", img)
cv2.waitKey(0)  # Wait for a key press to close the image window
cv2.destroyAllWindows()

# Save the image with the bounding box and recognized text (optional)
cv2.imwrite("output_with_bounding_box_and_text.jpg", img)