import os
from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import cv2

app = Flask(__name__)

# Set the path to the local folder to save uploaded images
UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

dic = {0 : 'Flutter Waves', 1 : 'Murmur', 2 : 'Normal Sinus Rhythm', 3 : 'Q Wave', 4 : 'Sinus Arrest', 5 : 'Ventricular Prematured Depolarization'}

@app.route("/", methods=["GET", "POST"])
def homepage():
    return render_template('homepage.html')

#### Machine Learning Code
img_size_x=432
img_size_y=288
model = load_model('model.h5')

def predict_label(img_path):
    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    resized=cv2.resize(gray,(img_size_x,img_size_y)) 
    i = img_to_array(resized)/255.0
    i = i.reshape(1,img_size_x,img_size_y,1)
    predict_x=model.predict(i) 
    p=np.argmax(predict_x,axis=1)
    return dic[p[0]]

@app.route("/upload", methods=["GET", "POST"])
def upload():
    p = None
    img_path = None
    if request.method == "POST" and 'photo' in request.files:
        # Get the uploaded file from the form data
        file = request.files['photo']

        # Save the file to the local folder
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            img_path = file_path

        p = predict_label(img_path)

    cp = str(p).lower() if p is not None else ""
    src = img_path if img_path is not None else ""

    return render_template('upload.html', cp=cp, src=src)




if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
