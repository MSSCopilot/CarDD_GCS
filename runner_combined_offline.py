import os
import base64
import requests
import json
import cv2
import concurrent.futures

def run_models_parallel(image_path):
    print(f"RUNNING MODELS FOR {image_path}")
    # Read image
    image = cv2.imread(image_path)
    # Ensure image is loaded correctly
    if image is None:
        raise Exception("Failed to load image")

    # Encode image to base64 string
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)

    # Convert bytes to string
    base64_image = jpg_as_text.decode()

    # Send HTTP requests to the servers in parallel
    part_url = 'https://carpart-5a4aan2gca-el.a.run.app/annotate'
    damage_url = 'https://cardamage-5a4aan2gca-el.a.run.app/annotate'

    headers = {'content-type': 'application/json'}
    # strip the image/ from the image path
    image_path = image_path.split('\\')[-1]
    data = json.dumps([base64_image, {"file_name": image_path}])

    part_response = requests.post(part_url, data=data, headers=headers)
    part_response.raise_for_status()
    damage_response = requests.post(damage_url, data=data, headers=headers)
    damage_response.raise_for_status()

    # Decode response
    part_response_data = part_response.json()
    damage_response_data = damage_response.json()

    return part_response_data, damage_response_data 

if __name__ == '__main__':
    # Get all images in the folder
    image_folder = 'archive_saudi_images\\archive_images'
    image_paths = [os.path.join(image_folder, file_name) for file_name in os.listdir(image_folder)]

    # Run models in parallel
    for image_path in image_paths:
        results = run_models_parallel(image_path)
        print(f'{image_path} annotated successfully')
