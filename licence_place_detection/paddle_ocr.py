import cv2
import numpy as np
from paddleocr import PaddleOCR
import tensorflow as tf

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Load your Keras model
model = tf.keras.models.load_model('./model_300.keras')

def is_valid_character(char):
    # Define acceptable character patterns including special characters
    valid_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return char in valid_characters

def predict_bounding_box(keras_model, image):
    # Preprocess the image to match the input shape of your Keras model
    original_height, original_width = image.shape[:2]
    image_resized = cv2.resize(image, (224, 224))  # Assuming your model was trained on 224x224 images
    image_array = np.expand_dims(image_resized / 255.0, axis=0)

    # Predict bounding box using the Keras model
    pred_bbox = keras_model.predict(image_array)[0]

    # Denormalize the predicted bounding box
    xmin, xmax, ymin, ymax = pred_bbox
    xmin = int(xmin * original_width)
    xmax = int(xmax * original_width)
    ymin = int(ymin * original_height)
    ymax = int(ymax * original_height)

    return xmin, ymin, xmax, ymax

# Load the image
img = cv2.imread('./train/images/3572232-bilbasen-suzuki_swift_2008-4_jpg.rf.90e12144a99c10116beff823d5cdb7b3.jpg')

# Predict the bounding box using your Keras model
x1, y1, x2, y2 = predict_bounding_box(model, img)

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