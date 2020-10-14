import tensorflow
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout,LSTM
import pickle
import numpy as np
from nltk import tokenize
from nltk import WordNetLemmatizer
import pandas as pd

with open('wp_data.pickle','rb') as f:
    (all_words,all_tags,training,output) = pickle.load(f)
all_tags = [t for t in all_tags]
training = np.array(training)
output = np.array(output)
try:
    model = load_model('wp_bot_model.sequential')
    # model.evaluate(training,output,batch_size=128)
except:
    model = Sequential()
    model.add(Dense(128,input_shape = (len(training[0]),),activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64,activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(output[0]), activation='softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(training,output,batch_size=128,epochs=50)
    model.save('wp_bot_model.sequential')

def bag_of_words(msg):
    lemm = WordNetLemmatizer()
    words = tokenize.word_tokenize(msg.lower())
    l_words = [lemm.lemmatize(w) for w in words if w not in ['!','?','.']]
    word_array = np.zeros(len(all_words))
    for idx,wrd in enumerate(all_words):
        if wrd in l_words:
            word_array[idx]=1
    return word_array

def chat():
    print('Start Chatting')
    while True:
        msg = input('You: ')
        if msg == 'quit':
            exit()
        else:
            word_array = bag_of_words(msg)
            probs = model.predict(np.array([word_array]))
            tag_idx = np.argmax(probs)
            out = all_tags[tag_idx]
            # with open('recieve_response.csv','r') as f:
            df = pd.read_csv('recieve_response.csv')
            # breakpoint()
            res = df[(df.Sender=='response') & (df.Tag == out+1)]
            for r in res.Msg:
                print('Bot: ',r)
chat()