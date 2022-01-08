from run_fire_detection import Fire_Detection
import numpy
import json
import base64
from PIL import Image
import numpy as np
import io
from fastapi import FastAPI
import uvicorn

detector_object = Fire_Detection()
app =  FastAPI()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.post("/send_image_input")
def run_detection(request: dict):
    im_b64 = request['image']
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))
    img = Image.open(io.BytesIO(img_bytes))
    img_arr = np.asarray(img) 
    print('[INFO] Running Fire Detection.')
    image = detector_object.run_fire_detection(img_arr,'image')
    print('[INFO] Detection Successfully Done.')
    bytes_image = io.BytesIO()
    img.save(bytes_image, format='PNG')    
    return json.dumps({"output_image":image},cls=NumpyEncoder)


if __name__ == '__main__':
    uvicorn.run(app = app,host = "0.0.0.0",port = 5000,debug = True)
