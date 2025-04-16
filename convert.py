import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# Load TFLite model
model_path = 'tflite-model/best_float32.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load image
image_path = 'test.jpg'
image = Image.open(image_path)

# Resize the image to model's expected input size (640x640)
image_resized = image.resize((640, 640))
image_array = np.array(image_resized) / 255.0  # Normalize

# Add batch dimension
image_input = np.expand_dims(image_array, axis=0).astype(np.float32)

# Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], image_input)

# Run inference
interpreter.invoke()

# Get output data
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the shape and check the output data
print(f"Output data shape: {output_data.shape}")
print(f"Sample output: {output_data[0]}")

# Post-process results
threshold = 0.3  # Confidence threshold

# Initialize PIL ImageDraw for drawing
draw = ImageDraw.Draw(image_resized)

# Class names (modify based on your classes)
class_names = ['fire', 'smoke']

# Iterate over detections (8400 possible detections)
for i in range(output_data.shape[2]):  # Loop through each detection
    # Extract the 6 values for each detection (bbox + confidence + class probs)
    detection = output_data[0, :, i]

    # Extract bounding box (xywh format)
    bbox = detection[0:4]  # First 4 values are the bounding box
    confidence = detection[4]  # Confidence score
    
    # If confidence is above the threshold, consider this detection
    if confidence > threshold:
        class_probs = detection[5:]  # Class probabilities
        class_id = np.argmax(class_probs)  # Class with highest probability
        class_name = class_names[class_id]  # Get class name
        
        # Convert bbox (xywh to xyxy)
        x_center, y_center, width, height = bbox
        x1 = int((x_center - width / 2) * image_resized.width)
        y1 = int((y_center - height / 2) * image_resized.height)
        x2 = int((x_center + width / 2) * image_resized.width)
        y2 = int((y_center + height / 2) * image_resized.height)

        # Draw bounding box on the image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = f"{class_name} {confidence:.2f}"
        draw.text((x1, y1), label, fill="red")

# Show image with bounding boxes
image_resized.show()

# Save the image with bounding boxes if needed
image_resized.save('output_with_bboxes.jpg')
