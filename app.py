from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras import applications 
from tensorflow.keras.models import load_model
import numpy as np 
import os
new_model=load_model("commomModel.h5")
vggModel=load_model("vggmodel.h5")


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

def read_image(file_path):
    #print("[INFO] loading and preprocessing image…") 
    image = load_img(file_path, target_size=(128, 128))
    image = img_to_array(image) 
    image = np.expand_dims(image, axis=0)
    image /= 255.
    return image

def test_single_image(k):
    bt_prediction = vggModel.predict(read_image(k)) 
    preds = new_model.predict(bt_prediction)
    class_predicted = new_model.predict_classes(bt_prediction)
    class_dictionary = {'COVID': 0, 'NORMAL': 1, 'PNEUMONIA': 2} 
    inv_map = {v: k for k, v in class_dictionary.items()} 
    k=inv_map[class_predicted[0]]
    return k

app = Flask(__name__)

upload_folder="static/css/"

@app.route('/',methods=['GET','POST'])
def predict():
	if(request.method == 'POST'):
		f = request.files['file']
		if(f):
			img_loc=os.path.join(upload_folder,f.filename)
		f.save(img_loc)
		return render_template('main.html',prediction=(test_single_image(img_loc)),image=f.filename)
	return render_template("main.html",prediction="",image="")
if __name__ == '__main__':
    app.debug = True
    app.run()
