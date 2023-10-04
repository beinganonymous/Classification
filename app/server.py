import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf  # Import TensorFlow

import tensorflow as tf

from tensorflow.keras.models import load_model

import tensorflow_hub as hub
import cv2


export_file_url = 'https://drive.google.com/uc?export=download&id=1yctEfO7_nTZjvqsR61yp7ZL-0vfljcF3'  # You can put your model URL. If it's too big
export_file_name = '/models/model.h5'

classes = ['good', 'bad']  # You can put your own classes here
path = Path(__file__).parent
print('abc')

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        # Load the model within a custom_object_scope
        my_reloaded_model = tf.keras.models.load_model(
       	(os.path.dirname(os.path.abspath(__file__)) + export_file_name),
       	custom_objects={'KerasLayer':hub.KerasLayer}
	)
        return my_reloaded_model
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    images_test=[]
    file_ = (await request.form())['file']
    image_binary = await file_.read()

# Convert the binary string to a NumPy array using OpenCV
    nparr = np.fromstring(image_binary, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Resize the image to the desired dimensions (e.g., 224x224)
    img = cv2.resize(img, (224, 224))

    images_test.append(img)

    images_test = np.array(images_test)

    images_test = images_test.astype('float32') / 255.0

    pred=learn.predict(images_test)

    val = ""

    if pred[0][0]>pred[0][1]:
      val = "Bad"
    else:
      val = "Good"
    
   
    return JSONResponse({'result': str(val)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
