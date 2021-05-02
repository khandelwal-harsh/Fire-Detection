from flask import Flask,jsonify,request,send_file
from run_fire_detection import Fire_Detection
import cv2
import numpy
import json
import os


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


app = Flask(__name__)
detector_object = Fire_Detection()

@app.route('/send_image_input', methods=['GET','POST'])
def run_detection():

	file = request.files.get("image","")
	print('[INFO] Running Face Mask Detection.')
	file_path = 'input_image.jpg'
	file.save(file_path)
	# input_file = cv2.imdecode(numpy.fromstring(file.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
	# print(input_image)
	output_file_path = detector_object.run_fire_detection(input_file,'image')
	print('[INFO] Detection Successfully Done.')
	return send_file(output_file_path,mimetype='image/gif')
	# return json.dumps({"output_image":output_image},cls=NumpyEncoder)

@app.route('/send_video_input', methods=['GET','POST'])
def run_detection():

	file = request.files.get("video","")
	print('[INFO] Running Face Mask Detection.')
	# input_file = cv2.imdecode(numpy.fromstring(file.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
	# print(input_image)
	file_path = 'input_video.mp4'
	file.save(file_path)
	output_file_path = detector_object.run_fire_detection(input_file,'video')
	print('[INFO] Detection Successfully Done.')
	return send_file(output_file_path)
	# return json.dumps({"output_image":output_image},cls=NumpyEncoder)

if __name__ == '__main__':
    app.run(host = "0.0.0.0",port = 5001)
