import requests
import base64
import json
import cv2
import numpy as np

api = 'http://localhost:5000/send_image_input'
image_file = 'fire.jpeg'

with open(image_file, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8")

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

payload = json.dumps({"image": im_b64, "other_key": "value"})
response = requests.post(api, data=payload, headers=headers)
output_image = np.asarray(json.loads(response.json())["output_image"])
cv2.imwrite('test123.png',output_image)
