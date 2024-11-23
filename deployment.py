from flask import Flask, render_template,request,Response
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

TF_ENABLE_MKL_NATIVE_FORMAT=0

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])

def index():
    model = load_model(r'C:\Users\HP\OneDrive\Desktop\AI New\Deploy Models\Handwritten Digits\model.h5')
    if request.method == 'POST':
        fu = request.form['file_upload']
        print(fu,'====')
        img = Image.open(fr"C:\Users\HP\Downloads\{fu}").convert('L')  # Load as grayscale
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = img_array.reshape(1, 28, 28, 1)  # Add batch dimension  
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        print("Predicted digit:", predicted_digit,'========================================')

        return render_template('index.html', predict = predicted_digit )
    return render_template('index.html')

app.run(debug=True)
