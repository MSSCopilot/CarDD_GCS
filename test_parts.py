import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from roboflow import Roboflow
import random
from google.cloud import storage
import concurrent.futures
import time
from shapely.geometry import Polygon


NEON_COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (255, 102, 0), (102, 255, 0), (0, 255, 102), (255, 0, 102), (102, 0, 255), (255, 255, 102)]



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
    # attrs = vars(blob) #mediaLink
    # print(attrs)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

    link = (blob._properties)['mediaLink']
    return link

def run_models_parallel(image_path, confidence=20, file_name='prediction.jpg'):
    print('RUNNING MODELS')
    time_start_models=time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(run_model, image_path, confidence, 1)
        future2 = executor.submit(run_model, image_path, confidence, 2)

        # Wait for both futures to complete
        concurrent.futures.wait([future1, future2])

        output1 = future1.result()
        output2 = future2.result()
        time_end_models=time.time()
        time_taken_models=time_end_models-time_start_models
        print(f'TIME TAKEN FOR MODELS: {time_taken_models}')
    # draw output 1 annotations as blue and output 2 annotations as red and output the image as no_check.jpg
    img=cv2.imread(image_path)
    def annotate_2color(img, output1, output2):
        for prediction1 in output1['predictions']:
            class_name = prediction1['class']
            points = prediction1['points']
            color = (255,0,0)
            # Draw polygon segments
            points_array = [(int(point['x']), int(point['y'])) for point in points]
            cv2.polylines(img, [np.array(points_array)], isClosed=True, color=color, thickness=1)

            # Fill the polygons with translucent color
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.array(points_array)], color)
            opacity = 0.4
            cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

            # Draw bounding box
            x, y, w, h = prediction1['x'], prediction1['y'], prediction1['width'], prediction1['height']
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

            # Draw class name at the center of the bounding box
            font_scale = 0.8
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_thickness = 2

            # Calculate text size with a dummy text to get the actual width and height
            (text_width, text_height), _ = cv2.getTextSize(class_name, font, font_scale, font_thickness)

            # Calculate text position to ensure it's centered within the bounding box
            text_x = x + (w - text_width) // 2
            text_y = y + (h + text_height) // 2

            cv2.putText(img, class_name, (int(text_x), int(text_y)), font, font_scale, color, font_thickness)
        for prediction2 in output2['predictions']:
            class_name = prediction2['class']
            points = prediction2['points']
            color = (0,0,255)
            # Draw polygon segments
            points_array = [(int(point['x']), int(point['y'])) for point in points]
            cv2.polylines(img, [np.array(points_array)], isClosed=True, color=color, thickness=1)

            # Fill the polygons with translucent color
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.array(points_array)], color)
            opacity = 0.4
            cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

            # Draw bounding box
            x, y, w, h = prediction2['x'], prediction2['y'], prediction2['width'], prediction2['height']
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

            # Draw class name at the center of the bounding box
            font_scale = 0.8
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_thickness = 2

            # Calculate text size with a dummy text to get the actual width and height
            (text_width, text_height), _ = cv2.getTextSize(class_name, font, font_scale, font_thickness)

            # Calculate text position to ensure it's centered within the bounding box
            text_x = x + (w - text_width) // 2
            text_y = y + (h + text_height) // 2

            cv2.putText(img, class_name, (int(text_x), int(text_y)), font, font_scale, color, font_thickness)

        return img
    img = annotate_2color(img, output1, output2)
    cv2.imwrite("predictions_part\\no_check\\"+file_name, img)

    # Combine the predictions from both models and remove overlapping predictions
    combined_predictions = []
    for prediction1 in output1['predictions']:
        add_prediction = True
        for prediction2 in output2['predictions']:
            overlap_percentage = calculate_overlap(prediction1['points'], prediction2['points'])
            print(f"OVERLAP PERCENTAGE FOR {prediction1['class']} and {prediction2['class']} is {overlap_percentage}")

            if overlap_percentage > 0.8:
                # Compare confidences and choose the prediction with higher confidence
                if prediction1['confidence'] > prediction2['confidence']:
                    print(f"adding {prediction1['class']} with conf={prediction1['confidence']} instead of {prediction2['class']} with conf={prediction2['confidence']}")
                    combined_predictions.append(prediction1)
                else:
                    combined_predictions.append(prediction2)
                add_prediction = False
                break
        if add_prediction:
            print(f"adding {prediction1['class']}")
            combined_predictions.append(prediction1)

    for prediction2 in output2['predictions']:
        # Add predictions from the second model that are not overlapping with the first model
        if all(calculate_overlap(prediction2['points'], p['points']) <= 0.8 for p in combined_predictions):
            print(f"adding {prediction2['class']}")
            combined_predictions.append(prediction2)

    combined_output = {'predictions': combined_predictions}
    annotated_image = draw_annotations(image_path, combined_output)
    return annotated_image, time_taken_models

def run_model(image_path, confidence, model_number):
    rf = Roboflow(api_key="L50aJwNZ3zWDrgp1VYCT")
    project = rf.workspace().project("car_pa") if model_number == 1 else rf.workspace().project("part-autolabeld")
    model = project.version(1).model

    print(f'PREDICTING MODEL {model_number}')
    prediction = model.predict(image_path, confidence)
    print(f'PREDICTION {model_number} DONE')

    output = prediction.json()

    # Recalculate and reassign bounding box coordinates
    for prediction in output['predictions']:
        points = prediction['points']
        min_x = min(point['x'] for point in points)
        max_x = max(point['x'] for point in points)
        min_y = min(point['y'] for point in points)
        max_y = max(point['y'] for point in points)

        prediction['x'] = int(min_x)
        prediction['y'] = int(min_y)
        prediction['width'] = int(max_x - min_x)
        prediction['height'] = int(max_y - min_y)

    return output


def calculate_overlap(points1, points2):
    # Extract x and y coordinates from the input points
    x_coords1 = [point['x'] for point in points1]
    y_coords1 = [point['y'] for point in points1]
    x_coords2 = [point['x'] for point in points2]
    y_coords2 = [point['y'] for point in points2]

    # Create polygons using the x and y coordinates
    polygon1 = Polygon(list(zip(x_coords1, y_coords1)))
    polygon2 = Polygon(list(zip(x_coords2, y_coords2)))

    # Calculate the intersection area and the ratio of overlap
    overlap_area = polygon1.intersection(polygon2).area
    area1 = polygon1.area
    area2 = polygon2.area

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
# pylint: disable=C0103
app = Flask(__name__)
@app.route('/annotate', methods=['POST'])
def annotate_image():
    time_start=time.time()
    # Read image and file name data = json.dumps([base64_image,{ "file_name": image_path }])
    try:
        base64_image = request.json[0]
    except:
        base64_image = request.data.decode('utf-8')
    try:
        file_name = request.json[1]['file_name']
    except:
        file_name = 'input.jpg'
    

    # Decode base64 image data
    image_data = base64.b64decode(base64_image)
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    cv2.imwrite('input.jpg', image)

    annotated_image,time_taken_models = run_models_parallel('input.jpg',file_name=file_name)
    print('ANNOTATED SUCCESSFULLY')
    output_location = "predictions_part\\"+file_name
    cv2.imwrite(output_location, annotated_image)
    
    # Convert annotated image to base64
    final_image = cv2.imread(output_location)
    # media_link = upload_blob(source_file_name=output_location)
    media_link = ' '
    time_end=time.time()
    time_taken=time_end-time_start
    return jsonify({'media_link': media_link, 'total_time_taken': time_taken, 'time_taken_models': time_taken_models})

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '5000')
    app.run(debug=True, port=server_port, host='0.0.0.0')
