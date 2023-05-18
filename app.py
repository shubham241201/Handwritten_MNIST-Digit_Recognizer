from flask import Flask, render_template, request, send_file
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np


app = Flask(__name__)

loaded_model=tf.keras.models.load_model('notebook\my_model')




@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    
    image = cv2.imread(image_path)
    image=cv2.medianBlur(image,7)
    
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(grey,200,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,33,25)
    
    contours,hierarchy= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    preprocessed_digits = []
    
    # initialize the reverse flag and sort index
    # handle if we need to sort in reverse
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                    key=lambda b:b[1][0], reverse=False))

    
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

            # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(255, 0, 0), thickness=2)
    
        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]
    
        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18,18))
    
            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
    
            # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)
        
    inp = np.array(preprocessed_digits)
    i=1
    nums=[]
    for digit in preprocessed_digits:
        [prediction] = loaded_model.predict(digit.reshape(1, 28, 28, 1)/255.)
        pred=np.argmax(prediction)
        nums.append(pred)   
        
    return render_template('index.html',list1 = nums)

if __name__ == '__main__':
    app.run(port=3000, debug = True)







