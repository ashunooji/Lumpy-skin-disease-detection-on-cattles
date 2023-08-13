from flask import Flask, request,render_template,url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps  
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("Models\keras_model.h5", compile=False)

categories = ['Normal','Diseased']

app = Flask(__name__)

def model_predict(img_path,model):
    test_image=image.load_img(img_path,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    test_image=test_image/255
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    return result


@app.route('/',methods=['POST','GET'])
def fun():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            image.save('static/uploads/'+image.filename)
            image_filename = 'uploads/'+image.filename
            file_path = 'static/uploads/'+image.filename
            result = model_predict(file_path,model)
            pred_class=result.argmax()
            output=categories[pred_class]
            print(output)
            return render_template('index.html',image_filename=image_filename,out = output)
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True,port=4856)