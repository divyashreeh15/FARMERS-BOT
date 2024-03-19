from flask import Flask, render_template, url_for, request
import sqlite3
import shutil
import os
import sys
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from gtts import gTTS 
import time
import base64
import pandas as pd
import pygame
from mutagen.mp3 import MP3
from difflib import SequenceMatcher

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

command = """CREATE TABLE IF NOT EXISTS seller(Id INTEGER PRIMARY KEY AUTOINCREMENT, crop TEXT, cost TEXT, district TEXT, image BLOB)"""
cursor.execute(command)

command = """CREATE TABLE IF NOT EXISTS buyer(Id INTEGER PRIMARY KEY AUTOINCREMENT, crop TEXT, cost TEXT, district TEXT, image BLOB)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('userlog.html')

@app.route('/leaf')
def leaf():
    return render_template('leaf.html')

@app.route('/leaf_disease', methods=['GET', 'POST'])
def leaf_disease():
    if request.method == 'POST':
        image = request.form['img']

        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'healthyvsunhealthynew-{}-{}.model'.format(LR, '2conv-basic')

        def process_verify_data():
            verifying_data = []
            path = 'static/test/'+image
            img_num = image.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 6, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        # fig = plt.figure()
        diseasename=" "
        rem=" "
        rem1=" "
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            # y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'Healthy'
            elif np.argmax(model_out) == 1:
                str_label = 'Bacterial'
            elif np.argmax(model_out) == 2:
                str_label = 'Viral'
            elif np.argmax(model_out) == 3:
                str_label = 'Specto'
            elif np.argmax(model_out) == 4:
                str_label = 'Leafmod'
            elif np.argmax(model_out) == 5:
                str_label = 'X'

            if str_label == 'Bacterial':
                diseasename = "Bacterial Spot "
                rem = "The remedies for Bacterial Spot are:\n\n "
                rem1 = [" Discard or destroy any affected plants",  
                "Do not compost them.", 
                "Rotate yoour tomato plants yearly to prevent re-infection next year.", 
                "Use copper fungicites"]

            if str_label == 'Viral':
                diseasename = "Yellow leaf curl virus "
                rem = "The remedies for Yellow leaf curl virus are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates", 
                "carbametes during the seedliing stage.", "Use copper fungicites"]
                
            if str_label == 'Specto':
                diseasename = "Spectr0 "
                rem = "The remedies for Yellow leaf curl virus are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates",
                "carbametes during the seedliing stage.",
                "Use copper fungicites"]      

            if str_label == 'Leafmod':
                diseasename = "Leafmold"
                rem = "The remedies for Late Blight are: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]

            if str_label == 'X':
                rem = "The remedies for Yellow leaf curl virus are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates",
                "carbametes during the seedliing stage.",
                "Use copper fungicites"] 

        return render_template('leaf.html', status=str_label, disease=diseasename, remedie=rem, remedie1=rem1, ImageDisplay="http://127.0.0.1:5000/static/test/"+image)

    return render_template('leaf.html')

@app.route('/voice')
def voice():
    return render_template('voice.html')

@app.route('/voice_assistant', methods=['GET', 'POST'])
def voice_assistant():
    if request.method == 'POST':
        lang = request.form['lang']
        query = request.form['query']
        query = query.lower()
        if os.path.exists('static/voice.mp3'):
            os.remove('static/voice.mp3')
        if lang == 'en':
            answer = 'result not found'
            List = pd.read_csv('English.csv')
            Question = List['question']
            Answer = List['answer']
            A=0
            for qsn in Question:
                if SequenceMatcher(None,query,qsn).ratio()*100 > 75:
                    answer = List.loc[A]['answer']
                    break
                A += 1
            song = gTTS(text=answer, lang=lang, slow=False)
            song.save('static/voice.mp3')
            return render_template('voice.html', answer=answer, song='http://127.0.0.1:5000/static/voice.mp3')
        
        if lang == 'kn':
            answer = 'result not found'
            Question = ['ಉತ್ತಮ ವ್ಯವಹಾರಗಳು', 'ರೋಗಗಳಿಂದ ಹೇಗೆ ಕಾಪಾಡಬೇಕು', 'ಮಾರ್ಕೆಟ್ ಮಾಡುವುದು ಹೇಗೆ']
            ANS = ['ಭೂಮಿ ಆರೋಗ್ಯವನ್ನು ಉಳಿಸುವುದು ಕೃಷಿ ಉತ್ಪನ್ನತೆಗೆ ನೆರವಾಗುವುದು. ಉತ್ತಮ ವ್ಯವಹಾರಗಳು ಸರಿಯಾದ ಭೂಮಿ ತಯಾರಿಕೆ, ಪದರ ವ್ಯವಸ್ಥೆ, ಬೆಳೆಯ ಉಳಿಸಿಕೆ, ಭೂಮಿ ಪರೀಕ್ಷೆ, ಜೈವ ಮಣುರುಗಳು ಮತ್ತು ಬಯೋಫರ್ಟಿಲೈಝರ್ ಬಳಸುವುದು ಮೊದಲಾದವುಗಳನ್ನು ಒಳಗೊಂಡಿರುತ್ತವೆ.',
                   'ಬೆಳೆ ಪೀಳಿಗೆಗಳು ಮತ್ತು ರೋಗಗಳು ಹೆಚ್ಚು ಬೆಳೆ ಉತ್ಪನ್ನದ ಹಾನಿಗೆ ಕಾರಣವಾಗಬಹುದು. ಪರಿಣಾಮಕಾರಿ ಪೀಳಿಗೆ ಮತ್ತು ರೋಗ ನಿರ್ವಹಣೆ ರೋಗಕ್ಕೆ ತೊಡಿಗೆ ಹೊಂದಿದ ಬೆಳೆ ಜಾತಿಗಳನ್ನು ಬಳಸುವುದು,ಸಮಯೋಚಿತ ನೋಟವೂ ಮತ್ತು ಅವಲಂಬಿತ ಪೀಳಿಗೆ ನಿರ್ವಹಣೆ, ಬೆಳೆ ಚಕ್ರವರ್ತಿ, ಜೈವಿಕ ನಿಯಂತ್ರಣ, ಸಾಂಸ್ಕೃತಿಕ ನಿಯಂತ್ರ ಣಮತ್ತು ರಸಾಯನ ನಿಯಂತ್ರಣ ಮೊದಲಾದ ಒಳ್ಳೇ ಪರಿಣಾಮಕಾರಿ ಪೀಳಿಗೆ ನಿಯಮನದ ಮೂಲಕ ಸಾಧ್ಯವಾಗುತ್ತದೆ',
                   'ನಿಮ್ಮ ಬೆಳೆಗಳಿಗೆ ಒಳ್ಳೆಯ ಬೆಲೆಯನ್ನು ಪಡೆಯಲು, ನೀವು ಅನ್ನದ ಮಾರುಕಟ್ಟೆಗೆ ಬೆಳೆಗಳನ್ನು ಮಾರ್ಕೆಟ್ ಮಾಡಬಹುದು, ಕೃಷಿ ವ್ಯಾಪಾರ ಪ್ರದರ್ಶನಗಳಲ್ಲಿ ಭಾಗವಹಿಸಬಹುದು ಅಥವಾ ಈ-ಮಾರುಕಟ್ಟೆ ಪ್ಲಾಟ್ಫಾರ್ಮ್ಗಳನ್ನು ಬಳಸಬಹುದು. ನೀವು ಖರೀದಿಕೆದಾರರೊಂದಿಗೆ ಸಂಯುಕ್ತವಾಗಿ ಬೆಲೆಗಳನ್ನು ವಾಣಿಜ್ಯದ ಮೂಲಕ ವಿವರಿಸಲು ಸಮೂಹ ಅಥವಾ ಸಹಕಾರಿ ತೋಟದ ರೂಪದಲ್ಲಿ ನಿರ್ಣಯಿಸಬಹುದು.']
            for qsn in Question:
                if SequenceMatcher(None, query, qsn).ratio()*100 > 75:
                    answer = ANS[Question.index(qsn)]

            song = gTTS(text=answer, lang=lang, slow=False)
            song.save('static/voice.mp3')
            print('\n------------Playing--------------\n')
            song = MP3("static/voice.mp3")
            pygame.mixer.init()
            pygame.mixer.music.load('static/voice.mp3')
            pygame.mixer.music.play()
            time.sleep(song.info.length)
            pygame.quit()
            return render_template('voice.html', answer=answer, song='http://127.0.0.1:5000/static/voice.mp3')

    return render_template('voice.html')

@app.route('/buyer')
def buyer():
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        cursor.execute("select * from seller")
        result = cursor.fetchall()

        if result:
            profile = []
            for row in result:
                profile.append(row[4].decode('utf-8'))

            return render_template('buyer.html', result=result, profile=profile)
        else:
            return render_template('buyer.html')

@app.route('/buy_crop', methods=['POST', 'GET'])
def buy_crop():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        data=request.form
        keys = []
        values = []
        for key in data:
            keys.append(key)
            values.append(data[key])
        print(keys)
        print(values)
        # for i in range(len(keys)):
        #     cursor.execute("select * from seller where crop = '"+keys[i]+"' and cost = '"+values[i]+"'")
        #     result = cursor.fetchall()

        #     cursor.execute("INSERT INTO buyer (crop, cost, district, image) VALUES (?,?,?,?)",result[0][1:])
        #     connection.commit()
        total = 0
        for price in values:
            total += int(price)
        return render_template('payment.html', total=total)
    return render_template('userlog.html')

@app.route('/market')
def market():
    try:
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        cursor.execute("select * from seller order by cost desc")
        result = cursor.fetchall()
        profile = []
        for row in result:
            profile.append(row[4].decode('utf-8'))

        return render_template('market.html', result=result,  profile=profile)
    except:
        return render_template('market.html')
@app.route('/seller')
def seller():
    return render_template('seller.html')

@app.route('/sell_crop', methods=['POST', 'GET'])
def sell_crop():
    if request.method == 'POST':
        crop = request.form['crop']
        cost = request.form['cost']
        dist = request.form['dist']
        img = request.form['img']

        with open('static/test/'+img, "rb") as img_file:
            my_string = base64.b64encode(img_file.read())

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        cursor.execute("INSERT INTO seller (crop, cost, district, image) VALUES (?,?,?,?)",[crop, cost, dist, my_string])
        connection.commit()

        return render_template('seller.html', msg="data uploaded successfully")

    return render_template('seller.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

@app.route('/payment', methods=['POST', 'GET'])
def payment():
    if request.method == 'POST':
        data = request.form
        print(data)
    return render_template('userlog.html' ,msg="crop buy successfully")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
