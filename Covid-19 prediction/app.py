from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request
import os
import numpy as np

app = Flask(__name__)
model = load_model(r"xrayprediction.h5",compile= False)
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','Post'])
def upload():
    if request.method=='POST':
        f= request.files['images'] 
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        img = image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
        pred =np.argmax(model.predict(x),axis=1)
        index =['Covid','Lung Opacity','Normal ','Viral Pneumonia']
        diagnosis = "The diagnosis is:",+str(index[pred[0]])

    return diagnosis

if __name__=='__main__':
    app.run(debug=True)




