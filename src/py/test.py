import json


with (open("C:/Users/pc/Desktop/code/diagnose_ai/data/dzs.json",'r') )as f:
    dzss=json.load(f)

for i in dzss['intents']:
    print(i['tag'])


print(len(dzss['intents']))

'''
from keras import layers , optimizers , models
from nltk.stem import WordNetLemmatizer



'''