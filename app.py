from flask import Flask,request
from PIL import Image as im
from io import BytesIO
import base64
import numpy
import numpy as np
import cv2
from keras.models import load_model

model = load_model('ObjectDetect.h5')

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def welcome():
    requestfromuser = request.data
    img = im.open(BytesIO(base64.b64decode(requestfromuser)))
    image = img.resize((32,32))
    ycbcr = image.convert('YCbCr')
    # output of ycbcr.getbands() put in order
    Y = 0
    Cb = 1
    Cr = 2

    YCbCr=list(ycbcr.getdata()) # flat list of tuples
    # reshape
    imYCbCr = numpy.reshape(YCbCr, (image.size[1], image.size[0], 3))
    # Convert 32-bit elements to 8-bit
    imYCbCr = imYCbCr.astype(numpy.uint8)
    img_test = np.array(imYCbCr).astype(np.float32)
    img_test = np.reshape(img_test,(32,32,3))
    img = cv2.resize(img_test,(32,32))     # resize image to match model's expected sizing
    img = img.reshape(1,32,32,3)
    hasil = model.predict(img)
    cari = max(hasil[0])

    x=0
    for i in hasil[0]:
        if i == cari:
            if x==0:
                return "Uang"
            elif x==1:
                return "Kartu"
            elif x==2:
                return "Jam Tangan"
            break
        else:
            return "Barang Tidak Diketahui"


        
if __name__ == '__main__':
    app.run()
