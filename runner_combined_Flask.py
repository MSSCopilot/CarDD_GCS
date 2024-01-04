import os
from flask import Flask, request, jsonify
import time
import requests
import json
from shapely.geometry import Polygon



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

    print(f"RUNNING MODELS FOR {file_name}")

    # Send HTTP requests to the servers in parallel
    part_url = 'https://carpart-5a4aan2gca-el.a.run.app/annotate'
    damage_url = 'https://cardamage-5a4aan2gca-el.a.run.app/annotate'

    headers = {'content-type': 'application/json'}
    data = json.dumps([base64_image])

    part_response = requests.post(part_url, data=data, headers=headers)
    part_response.raise_for_status()
    damage_response = requests.post(damage_url, data=data, headers=headers)
    damage_response.raise_for_status()

    # Decode response
    part_response_data = part_response.json()
    damage_response_data = damage_response.json()

    # create a dict containing all parts and their respective damage
    # we will use overlap to determine which part is damaged

    # append the two dicts part_response_data and damage_response_data
    combined_response_data = {**part_response_data, **damage_response_data} # ** unpacks the dicts
    return jsonify(combined_response_data)

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '5000')
    app.run(debug=True, port=server_port, host='0.0.0.0')
