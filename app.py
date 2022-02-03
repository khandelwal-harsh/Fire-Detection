from run_fire_detection import Fire_Detection
import numpy
import json
import base64
from PIL import Image
import numpy as np
import io
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import requests
import cloudinary
import cloudinary.uploader
import cloudinary.api

detector_object = Fire_Detection()
app =  FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["POST", "GET"],
		allow_headers=["*"],
    max_age=3600,
)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.post("/send_image_input")
def run_detection(request: dict):
    if request["image_type"] == "url":
        response = requests.get(request["image"])
        img = Image.open(io.BytesIO(response.content))
    else:
        im_b64 = request['image']
        img_bytes = base64.b64decode(im_b64.encode('utf-8'))
        img = Image.open(io.BytesIO(img_bytes))
    img_arr = np.asarray(img) 
    print('[INFO] Running Fire Detection.')
    image = detector_object.run_fire_detection(img_arr,'image')
    cloudinary.config( 
    cloud_name = "instaimages", 
    api_key = "539269294439513", 
    api_secret = "mGHcTrXbNsIcgo9hGWVhIK1knmw",
    secure = False
    )    
    resp = cloudinary.uploader.upload(image,resource_type="image")

    print('[INFO] Detection Successfully Done.',resp)
  
    return {
        "success":True,
        "data":{
            "url":resp["url"]
        }
    }


if __name__ == '__main__':
    uvicorn.run(app = app,host = "0.0.0.0",port = 5000,debug = True)
