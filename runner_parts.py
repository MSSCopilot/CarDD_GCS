# base64 -w 0 test.jpg > base64_image.txt
# curl -X POST -H "Content-Type: application/json" -d @base64_image.txt http://127.0.0.1:8080/annotate

# convert the above bash commands to python and run all images in /images folder
import os
import base64
import requests
import json
import time
import cv2
import numpy as np

def run_models_parallel(image_path):
    # Read image
    image = cv2.imread(image_path)
    # Ensure image is loaded correctly
    if image is None:
        raise Exception("Failed to load image")

    # Encode image to base64 string
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)

    # # Convert bytes to string
    base64_image = jpg_as_text.decode()

    # Send HTTP request to the server
    url = 'http://127.0.0.1:8080/annotate'
    headers = {'content-type': 'application/json'}
    # strip everthing except the image name from the image path
    image_path = image_path.split('\\')[-1]
    data = json.dumps([base64_image,{ "file_name": image_path }])
    response = requests.post(url, data=data, headers=headers)
    response.raise_for_status()

    # Decode response
    response_data = response.json()
    media_link = response_data['media_link']
    time_taken_models = response_data['time_taken_models']


    return time_taken_models


if __name__ == '__main__':
    # Get all images in the folder
    image_folder = 'archive_saudi_images\\archive_images'
    image_paths = [os.path.join(image_folder, file_name) for file_name in os.listdir(image_folder)]

    # Run models in parallel
    time_start = time.time()
    for image_path in image_paths:
        try:
            time_taken_models = run_models_parallel(image_path)
            print(f'{image_path} annotated successfully')
            print('Time taken by models: {} seconds'.format(time_taken_models))
        except:
            print(f'{image_path} failed')
    time_end = time.time()
    time_taken = time_end - time_start
    print('Total time taken: {} seconds'.format(time_taken))

