import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from roboflow import Roboflow
import random
from google.cloud import storage



NEON_COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (255, 102, 0), (102, 255, 0), (0, 255, 102), (255, 0, 102), (102, 0, 255), (255, 255, 102)]

# pylint: disable=C0103
app = Flask(__name__)

def upload_blob(bucket_name='car_combined_results', source_file_name='prediction.jpg', destination_blob_name='prediction.jpg'):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    blob=bucket.get_blob(destination_blob_name)
    attrs = vars(blob) #mediaLink
    print(attrs)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

    link = (blob._properties)['mediaLink']
    return link

def run_models(image_path, confidence=30):
    print('RUNNING MODELS')

    rf1 = Roboflow(api_key="L50aJwNZ3zWDrgp1VYCT")
    project1 = rf1.workspace().project("car_pa")
    model1 = project1.version(1).model

    rf2 = Roboflow(api_key="L50aJwNZ3zWDrgp1VYCT")
    project2 = rf2.workspace().project("part-autolabeld")
    model2 = project2.version(1).model

    print('PREDICTING 1')
    prediction1 = model1.predict(image_path, confidence)
    print('PREDICTION 1 DONE')
    print('PREDICTING 2')
    prediction2 = model2.predict(image_path, confidence)
    print('PREDICTION 2 DONE')

    output1 = prediction1.json()
    output2 = prediction2.json()

        # Recalculate and reassign bounding box coordinates for output1
    for prediction in output1['predictions']:
        points = prediction['points']
        min_x = min(point['x'] for point in points)
        max_x = max(point['x'] for point in points)
        min_y = min(point['y'] for point in points)
        max_y = max(point['y'] for point in points)

        prediction['x'] = int(min_x)
        prediction['y'] = int(min_y)
        prediction['width'] = int(max_x - min_x)
        prediction['height'] = int(max_y - min_y)

    # Recalculate and reassign bounding box coordinates for output2
    for prediction in output2['predictions']:
        points = prediction['points']
        min_x = min(point['x'] for point in points)
        max_x = max(point['x'] for point in points)
        min_y = min(point['y'] for point in points)
        max_y = max(point['y'] for point in points)

        prediction['x'] = int(min_x)
        prediction['y'] = int(min_y)
        prediction['width'] = int(max_x - min_x)
        prediction['height'] = int(max_y - min_y)

    # Combine the predictions from both models and remove overlapping predictions
    combined_predictions = []
    for prediction1 in output1['predictions']:
        add_prediction = True
        for prediction2 in output2['predictions']:
            overlap_percentage = calculate_overlap(prediction1, prediction2)
            print(f"OVERLAP PERCENTAGE FOR {prediction1['class']} and {prediction2['class']} is {overlap_percentage}")

            if overlap_percentage > 0.2:
                add_prediction = False
                break
        if add_prediction:
            print(f"adding {prediction1['class']}")
            combined_predictions.append(prediction1)

    for prediction2 in output2['predictions']:
        print(f"adding {prediction2['class']}")
        combined_predictions.append(prediction2)

    combined_output = {'predictions': combined_predictions}
    annotated_image = draw_annotations(image_path, combined_output)
    return annotated_image

def calculate_overlap(box1, box2):
    x1, y1, w1, h1 = box1['x'], box1['y'], box1['width'], box1['height']
    x2, y2, w2, h2 = box2['x'], box2['y'], box2['width'], box2['height']

    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    overlap_area = overlap_x * overlap_y
    area1 = w1 * h1
    area2 = w2 * h2

    overlap_percentage = overlap_area / min(area1, area2)
    return overlap_percentage

def draw_annotations(image_path, predictions):
    image = cv2.imread(image_path)
    
    for prediction1 in predictions['predictions']:
        class_name = prediction1['class']
        points = prediction1['points']
        color = random.choice(NEON_COLOURS)
        
        # Draw polygon segments
        points_array = [(int(point['x']), int(point['y'])) for point in points]
        cv2.polylines(image, [np.array(points_array)], isClosed=True, color=color, thickness=1)

        # Fill the polygons with translucent color
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array(points_array)], color)
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

        # Draw bounding box
        x, y, w, h = prediction1['x'], prediction1['y'], prediction1['width'], prediction1['height']
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

        # Draw class name at the center of the bounding box
        font_scale = 0.8
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2

        # Calculate text size with a dummy text to get the actual width and height
        (text_width, text_height), _ = cv2.getTextSize(class_name, font, font_scale, font_thickness)

        # Calculate text position to ensure it's centered within the bounding box
        text_x = x + (w - text_width) // 2
        text_y = y + (h + text_height) // 2

        cv2.putText(image, class_name, (int(text_x), int(text_y)), font, font_scale, color, font_thickness)

    return image

@app.route('/annotate', methods=['POST'])
def annotate_image():
    # data = request.get_json()
    base64_image = request.data.decode('utf-8')

    # Decode base64 image data
    image_data = base64.b64decode(base64_image)
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    cv2.imwrite('test.jpg',image)
    annotated_image = run_models('test.jpg')
    print('ANNOTATED SUCCESFULLY')
    cv2.imwrite('prediction.jpg',annotated_image)
    # return jsonify({'bruh':'momentos'})

    # Convert annotated image to base64
    final_image = cv2.imread('prediction.jpg')
    media_link = upload_blob(source_file_name='prediction.jpg')
    # media_link='bruh'
    _, buffer = cv2.imencode('.jpg', final_image)
    annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'media_link': media_link})

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=True, port=server_port, host='0.0.0.0')
