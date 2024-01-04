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
                (255, 102, 0), (102, 255, 0), (0, 255, 102), (255, 0, 102)]



def upload_blob(bucket_name='car_combined_results', source_file_name='prediction.jpg', destination_blob_name='prediction_damage.jpg'):
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


def annotate_model_colors(img, outputs, model_names,model_colors=NEON_COLOURS):
    for output_idx, output in enumerate(outputs):
        model_color = model_colors[output_idx]
        for prediction in output['predictions']:
            class_name = prediction['class']
            points = prediction['points']
            confidence = prediction['confidence']
            # take 2 decimal places
            confidence = round(confidence, 2)
            color = model_color

            # Draw polygon segments
            points_array = [(int(point['x']), int(point['y'])) for point in points]
            cv2.polylines(img, [np.array(points_array)], isClosed=True, color=color, thickness=1)

            # Fill the polygons with translucent color
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.array(points_array)], color)
            opacity = 0.4
            cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

            # Draw bounding box
            x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
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

            # cv2.putText(img, class_name+" "+ confidence, (int(text_x), int(text_y)), font, font_scale, color, font_thickness)
            # put text with black background for better readability
            cv2.putText(img, class_name+" "+ str(confidence), (int(text_x), int(text_y)), font, font_scale, (0,0,0), font_thickness+1)
            cv2.putText(img, class_name+" "+ str(confidence), (int(text_x), int(text_y)), font, font_scale, color, font_thickness)

    # put a legend on the top left corner with model names and colors
    legend_y = 50
    legend_x = 50
    for model_idx, model_name in enumerate(model_names):
        cv2.putText(img, model_name, (legend_x, legend_y), font, font_scale, model_colors[model_idx], font_thickness)
        legend_y += 50

    return img

def run_models_parallel(image_path, confidence=30, file_name='prediction.jpg',model_names=['carddseg', 'damage-type-nogzj','etc-fi5yf','engenfix-car_damages-dents']):
    print('RUNNING MODELS')
    time_start_models = time.time()
    outputs=[]
    combined_predictions = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for model_name in model_names:
            future = executor.submit(run_model, image_path, confidence, model_name)
            futures.append(future)

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

        for future in futures:
            output = future.result()
            outputs.append(output)
            combined_predictions.extend(output['predictions'])

    time_end_models = time.time()
    time_taken_models = time_end_models - time_start_models
    print(f'TIME TAKEN FOR MODELS: {time_taken_models}')
    
    img = cv2.imread(image_path)
    img = annotate_model_colors(img, outputs, model_names)
    cv2.imwrite("predictions_damage\\no_check\\" + file_name, img)

    # Combine the predictions from all models and remove overlapping predictions
    non_overlapping_predictions = []
    for prediction in combined_predictions:
        add_prediction = True
        for existing_prediction in non_overlapping_predictions:
            overlap_percentage = calculate_overlap(prediction['points'], existing_prediction['points'])
            # print(f"OVERLAP PERCENTAGE FOR {prediction['class']} and {existing_prediction['class']} is {overlap_percentage}")

            if overlap_percentage > 0.5:
                print(f"OVERLAP PERCENTAGE FOR {prediction['class']} and {existing_prediction['class']} is {overlap_percentage}")
                # reverse check the overlap
                rev_overlap_percentage = calculate_overlap(existing_prediction['points'], prediction['points'])
                if rev_overlap_percentage > 0.5:
                    print(f"REVERSE OVERLAP PERCENTAGE FOR {prediction['class']} and {existing_prediction['class']} is {rev_overlap_percentage}")
                    # Compare confidences and choose the prediction with higher confidence
                    if prediction['confidence'] > existing_prediction['confidence']:
                        print(f"adding {prediction['class']} with conf={prediction['confidence']} instead of {existing_prediction['class']} with conf={existing_prediction['confidence']}")
                        non_overlapping_predictions.remove(existing_prediction)
                        non_overlapping_predictions.append(prediction)
                    add_prediction = False
                    break
        if add_prediction:
            print(f"adding {prediction['class'] } as no overlap")
            non_overlapping_predictions.append(prediction)

    combined_output = {'predictions': non_overlapping_predictions}
    annotated_image = draw_annotations(image_path, combined_output)
    return annotated_image, time_taken_models, combined_output

def run_model(image_path, confidence, model_name):
    rf = Roboflow(api_key="mQMgcPnrQmBsKM3lOZiX")
    project = rf.workspace().project(model_name)
    model = project.version(1).model
    if model_name == 'damage-type-nogzj':
        model = project.version(2).model

    print(f'PREDICTING MODEL {model_name}')
    prediction = model.predict(image_path, confidence)
    print(f'PREDICTION {model_name} DONE')

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

    overlap_percentage = overlap_area / area1
    return overlap_percentage

def draw_annotations(image_path, predictions):
    image = cv2.imread(image_path)
    
    for prediction1 in predictions['predictions']:
        class_name = prediction1['class']
        points = prediction1['points']
        confidence = prediction1['confidence']
        confidence = round(confidence, 2)
        colors=NEON_COLOURS.copy()
        color = random.choice(colors)

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
        (text_width, text_height), _ = cv2.getTextSize(class_name+" "+ str(confidence), font, font_scale, font_thickness)

        # Calculate text position to ensure it's centered within the bounding box
        text_x = x + (w - text_width) // 2
        text_y = y + (h + text_height) // 2

        cv2.putText(image, class_name+" "+ str(confidence), (int(text_x), int(text_y)), font, font_scale, (0,0,0), font_thickness+1)
        cv2.putText(image, class_name+" "+ str(confidence) ,(int(text_x), int(text_y)), font, font_scale, color, font_thickness)

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
        file_name = 'input_damage.jpg'
    

    # Decode base64 image data
    image_data = base64.b64decode(base64_image)
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    cv2.imwrite('input_damage.jpg', image)

    annotated_image,time_taken_models,combined_outputs = run_models_parallel('input_damage.jpg',file_name=file_name)
    print('ANNOTATED SUCCESSFULLY')
    output_location = 'output'+file_name
    cv2.imwrite(output_location, annotated_image)
    
    # Convert annotated image to base64
    media_link = upload_blob(source_file_name=output_location)
    time_end=time.time()
    time_taken=time_end-time_start

    
    return jsonify({'damage_media_link': media_link, 'total_time_taken': time_taken, 'time_taken_models': time_taken_models, 'damage_predictions': combined_outputs})

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '5000')
    app.run(debug=True, port=server_port, host='0.0.0.0')
