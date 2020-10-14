import pandas as pd
import matplotlib.pyplot as plt
import datetime
import nltk
from nltk import NLTKWordTokenizer
import numpy as np
import seaborn as sns
import emoji
import re
import collections
import pickle
import csv

def remove_unwanted_char(filename):
    unwanted_chars = {'/','?','!','<','>','.',','}
    with open(filename,'rt') as f:
        all_data = f.read()
        f.close()
    for char in unwanted_chars:
        all_data = all_data.replace(char,'')
    with open(filename,'wt') as f:
        f.write(all_data)
        f.close()

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def whatsapp_datapreprocess(filename):
    with open(filename,'rt',encoding = "utf-8") as f:
        chats = f.read()
        f.close()
    lines = chats.splitlines()
    msgs = []
    errors = []
    for line in lines:
        try:
            line = remove_emoji(line)
            msgs.append([line[0:10],line[12:17],line[20:].split(':')[0],line[20:].split(':')[1].strip()])
                
        except:
            errors.append(line)
    df = pd.DataFrame(msgs,columns=["Date","Time","Sender","Msg"])
    df.Date = pd.to_datetime(df.Date,format='%d/%m/%Y')
    df.replace('', np.nan).dropna(subset=['Msg'])
    return df
# df = whatsapp_datapreprocess('WhatsApp Chat with Rawaa.txt')

# def emoji_count(filename):
#     with open(filename,'rt',encoding = "utf-8") as f:
#         chats = f.read()
#         f.close()
#     lines = chats.splitlines()
#     msgs = []
#     errors = []
#     for line in lines:
#         try:
#             msgs.append([line[0:10],line[12:17],line[20:].split(':')[0],line[20:].split(':')[1].strip()])
#         except:
#             errors.append(line)
#     print(errors)
#     print(len(errors))
#     df = pd.DataFrame(msgs,columns=["Date","Time","Sender","Msg"])
#     df.Date = pd.to_datetime(df.Date,format='%d/%m/%Y')

#     emojis = df['Msg'].apply(lambda x: get_emoji(x))

#     word_count = collections.Counter(emojis)
#     print(word_count)
#     print(emojis.head(10))
#     print(len(emojis))

# emoji_count("Whatsapp Chat with Rawaa.txt")


def recieve_response():
    df = whatsapp_datapreprocess("Whatsapp Chat with Rawaa.txt")
    sender_msg = pd.DataFrame(df[['Sender','Msg']])
    sender_msg = sender_msg.replace(to_replace = {'Sender':['Bhagwaan','Rawaa']},value = {'Sender':['recieve','response']})
    sender_msg.replace('',np.NaN,inplace=True)
    sender_msg.dropna(how = 'any',subset=['Msg'],axis=0,inplace=True)
    tag = []
    i=0
    for sen in sender_msg.Sender:
        if sen == 'response':
            i+=1
            tag.append(i)
        else:
            tag.append(i)
    sender_msg['Tag'] = tag
    # with open('recieve_response.pickle','w') as f:
    sender_msg.to_csv('recieve_response.csv',index=True)
    return sender_msg
# recieve_response()


def tokenize(recive_response):
    bag_of_words = []
    for msg in recieve_response.Msg:
        line_words =  nltk.tokenize.word_tokenize(msg)
        bag_of_words.extend(line_words)
    bag_of_words = sorted(set(bag_of_words))
    for w in bag_of_words[7501:]:
        bag_of_words.remove(w)
    print(bag_of_words)
    print(len(bag_of_words))

    with open('bag_of_words.txt','w',encoding='utf-8') as f:
        for w in bag_of_words:
            f.write(w+" ")
        f.close()
# recieve_response = recieve_response()
# tokenize(recieve_response)

def lemmetize():
    from nltk import WordNetLemmatizer
    lemmetizer = WordNetLemmatizer()
    with open('bag_of_words.txt','r',encoding='utf-8') as f:
        all_words = f.read().lower()
        f.close()
    all_words = all_words.split()
    lem_words = []
    for w in all_words:
        w = lemmetizer.lemmatize(w)
        lem_words.append(w)
    lem_words = sorted(set(lem_words))
    print(lem_words)
    with open('lem_words.txt','w',encoding='utf-8') as f:
        for w in lem_words:
            f.write(w+' ')
    # print(len(lem_words))
# lemmetize()

def training():
    data = recieve_response()
    with open('lem_words.txt','r',encoding='utf-8') as f:
        all_words = f.read().split()
        f.close()
    training = []
    output = []
    all_tags = set(data.Tag)
    # all_tags = np.array(all_tags)
    # print(word_array)
    for msg in data.Msg:
        word_array = np.zeros(len(all_words),dtype = int)
        line_words =  nltk.tokenize.word_tokenize(msg.lower())
        for idx,w in enumerate(all_words):
            if w in line_words:
                word_array[idx]=1
            else:
                word_array[idx] = 0
        training.append(np.array(word_array))
    for tag in data.Tag:
        # breakpoint()
        tag_array = np.zeros(len(all_tags))
        for idx,t in enumerate(all_tags):
            if t == tag:
                tag_array[idx] = 1
                break
        output.append(np.array(tag_array))
    with open('wp_data.pickle','wb') as f:
        pickle.dump((all_words,all_tags,training,output),f)

if __name__ == "__main__":
    fname = input("Input the file name of chat = ")
    remove_unwanted_char(fname)
    whatsapp_datapreprocess(fname)
    recieve_response()
    tokenize()
    lemmetize()
    training()
