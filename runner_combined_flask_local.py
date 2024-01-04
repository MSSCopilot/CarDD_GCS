import os
from flask import Flask, request, jsonify
import time
import requests
import json
from shapely.geometry import Polygon
import re


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
    part_url = 'http://127.0.0.1:8080/annotate'
    damage_url = 'http://127.0.0.1:5000/annotate'

    headers = {'content-type': 'application/json'}
    data = json.dumps([base64_image])

    part_response = requests.post(part_url, data=data, headers=headers)
    part_response.raise_for_status()
    damage_response = requests.post(damage_url, data=data, headers=headers)
    damage_response.raise_for_status()

    # Decode response
    part_response_data = part_response.json()
    damage_response_data = damage_response.json()

    #    eg output:{
    #   "damage_media_link": "https://storage.googleapis.com/download/storage/v1/b/car_combined_results/o/prediction_damage.jpg?generation=1702988809992669&alt=media",
    #   "damage_predictions": {
    #     "predictions": [
    #       {
    #         "class": "scratch",
    #         "class_id": 4,
    #         "confidence": 0.6432067155838013,
    #         "height": 375,
    #         "image_path": "input_damage.jpg",
    #         "points": [
    #           {
    #             "x": 2.0,
    #             "y": 532.5
    #           },
    #           {

    # create a dict containing all parts and their respective damage
    # we will use overlap to determine which part is damaged
    parts = {}
    for part in part_response_data['part_predictions']['predictions']:
        # if part with some index (classname_x) already exists, extract the index and increment it
        #use regex to check if part name with some index already exists
        # Find all matches
        matches = re.findall(part['class'] + '_\d', str(parts.keys()))

        # Extract indices
        indices = [int(match.split('_')[-1]) for match in matches]

        # Find the maximum index
        if indices:
            index = max(indices) + 1
        else:
            index = 0
            

        parts[part['class']+f'_{index}'] = {'points': part['points'], 'damage': []}

        # assign indexed class name to original part name
        part['class'] = part['class']+f'_{index}'


    # loop through all damage predictions and append the damage to the respective part
    for damage in damage_response_data['damage_predictions']['predictions']:
        for part in parts:
            overlap = calculate_overlap(parts[part]['points'], damage['points'])
            reverse_overlap = calculate_overlap(damage['points'], parts[part]['points'])
            # print the overlap with classnames
            print(f'{part} - {damage["class"]} - {overlap} - {reverse_overlap}')
            if overlap > 0.1 or reverse_overlap > 0.1:

                parts[part]['damage'].append(damage)

    # loop through all parts and determine the damage
    for part in parts:
        if len(parts[part]['damage']) > 0:
            parts[part]['damage'] = f'damaged, type: {parts[part]["damage"][0]["class"]}'
        else:
            parts[part]['damage'] = 'not damaged'

    

    #remove points from parts
    for part in parts:
        del parts[part]['points']


    # remove 'points' from both part_response_data and damage_response_data
    for part in part_response_data['part_predictions']['predictions']:
        del part['points']
    for damage in damage_response_data['damage_predictions']['predictions']:
        del damage['points']

    # append the two dicts part_response_data and damage_response_data
    combined_response_data = {'PARTS_ALL':part_response_data, 'DAMAGE_ALL':damage_response_data,'OVERLAP PARTS WITH DMG': parts} # ** unpacks the dicts
    return jsonify(combined_response_data)

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '3086')
    app.run(debug=True, port=server_port, host='0.0.0.0')
