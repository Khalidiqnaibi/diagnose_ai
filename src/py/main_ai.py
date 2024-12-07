import json
import pickle
import numpy as np
import pyttsx3
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras import models
import tensorflow as tf
import os
import time  # Added for measuring execution time
from dotenv import load_dotenv

#* based on Ash attempt num 4  

#! optimized dignose_ai : big o ==> O(n+p+klogk) (p is constant and k is the sort of dzs)

################
##? Immortal ?##
################

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for TensorFlow
tf.get_logger().setLevel('ERROR')  # Minimize TensorFlow logging


lemmatizer = WordNetLemmatizer()
with open("C:/Users/pc/Desktop/code/diagnose_ai/data/dzs.json", 'r') as f:
    comms = json.load(f)

commwords = pickle.load(open('C:/Users/pc/Desktop/code/diagnose_ai/src/models/dzswords.pkl', 'rb'))
commclasses = pickle.load(open('C:/Users/pc/Desktop/code/diagnose_ai/src/models/dzsclasses.pkl', 'rb'))
commmodel = models.load_model('C:/Users/pc/Desktop/code/diagnose_ai/src/models/chatbotdzs.h5')

# Preprocess word list for fast lookup
word_index_map = {word: i for i, word in enumerate(commwords)}  # O(f) one-time setup

# TTS setup
engine = pyttsx3.init()
engine.setProperty('voice', engine.getProperty('voices')[1].id)  # Select voice
engine.setProperty('rate', 225)  # Speed of speech

def clean_up_sentences(sentence):
    """tokenize and lemmatize input sentence"""
    sentence_words = word_tokenize(sentence)
    return [lemmatizer.lemmatize(word) for word in sentence_words]

def bag_of_words(sentence, word_index_map):
    """convert sentence to bow representation using hashmap"""
    sentence_words = clean_up_sentences(sentence)
    bag = np.zeros(len(word_index_map), dtype=np.float32)
    for word in sentence_words:
        if word in word_index_map:
            bag[word_index_map[word]] = 1.0
    return bag

def predict_class(sentence, model, word_index_map, classes):
    """predict intent using trained model"""
    bow = bag_of_words(sentence, word_index_map).reshape(1, -1)
    res = model.predict(bow, verbose=0)[0]
    ERROR_THRESHOLD = 0.06
    results = [{'intent': classes[i], 'probability': r} for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    return sorted(results, key=lambda x: x['probability'], reverse=True)

def get_type(intent_list, intents_data):
    """get the main intent type"""
    if not intent_list:
        return ''
    tag = intent_list[0]['intent']
    for intent in intents_data['intents']:
        if intent['tag'] == tag:
            return tag
    return ''

def process_sentence(message):
    """Process and print predictions for a single sentence"""
    start_time = time.time()  # Start timer
    predictions = predict_class(message, commmodel, word_index_map, commclasses)
    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    print("====================================")
    print(f"Processing Time: {elapsed_time:.2f} ms")
    if predictions:
        main_diagnose = get_type(predictions, comms)
        print(f"Main Diagnose: {main_diagnose} ==> Probability: {float(predictions[0]['probability'])*100:.2f}%")
        if len(predictions) > 1:
            for diagnose in predictions[1:]:
                print(f"Other Diagnose: {diagnose['intent']} ==> Probability: {float(diagnose['probability'])*100:.2f}%")
    else:
        print("No intents matched the message.")

def run():
    """Run the chatbot and wait for user input after processing"""
    # Process the hardcoded sentence
    message = "my chest is ouchy and I feel a little dizzy and I can't sleep"
    process_sentence(message)

    boo= True
    # Enter interactive mode for further sentences
    while boo:
        print("\nType a new sentence (or type 'exit' to quit):")
        message = input("> ").strip()
        if message.lower() == 'exit':
            print("Exiting... Goodbye!")
            boo = False
        else:
            process_sentence(message)

if __name__ == "__main__":
    run()


'''
# required modules
import random
import json
import pickle
import spacy
import numpy as np
import nltk
import string
import pyttsx3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier
from keras import models
import tensorflow as tf
from nltk.stem import WordNetLemmatizer 
from dotenv import load_dotenv
import os

# Ash attempt num 4  

user = "khalid afif sami iqnaibi"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Initialize the TTS engine
engine = pyttsx3.init()
voice = engine.getProperty('voices')[1]
engine.setProperty('voice', voice.id)
engine.setProperty('rate', 225)  # Speed of speech

# Load environment variables
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Set TensorFlow logging to only show errors
tf.get_logger().setLevel('ERROR')

# Load necessary resources
lemmatizer = WordNetLemmatizer()
with open("C:/Users/pc/Desktop/code/diagnose_ai/data/dzs.json", 'r') as f:
    comms = json.load(f)

commwords = pickle.load(open('C:/Users/pc/Desktop/code/diagnose_ai/src/models/dzswords.pkl', 'rb'))
commclasses = pickle.load(open('C:/Users/pc/Desktop/code/diagnose_ai/src/models/dzsclasses.pkl', 'rb'))
commmodel = models.load_model('C:/Users/pc/Desktop/code/diagnose_ai/src/models/chatbotdzs.h5')

nlp = spacy.load('en_core_web_md')

def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

def bag_of_words(sentence, words_list):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words_list)
    for w in sentence_words:
        if w in words_list:
            bag[words_list.index(w)] = 1
    # Ensure the output has the correct number of features
    if len(bag) != 6:
        print(f"Warning: Expected 6 features, got {len(bag)}")
    return np.array(bag)

def predict_class(sentence, wordspkl, classespkl, model):
    bow = bag_of_words(sentence, wordspkl)
    bow = np.array([bow], dtype=np.float32)  # Ensure input is 2D and float32
    #print(f"Input shape for prediction: {bow.shape}")
    #print("Bow:", bow)
    
    try:
        res = model.predict(bow, verbose=0)[0]
    except Exception as e:
        print("Error during prediction:", e)
        return []
    
    ERROR_THRESHOLD = 0.06
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Check if results are empty
    if not results:
        print("No results above the error threshold.")
        return []
    
    return [{'intent': classespkl[r[0]], 'probability': str(r[1])} for r in results]

def get_type(comm_list, comm_json):
    if not comm_list:
        print("Warning: comm_list is empty.")
        return ''  # Return an empty string or a default value
    tag = comm_list[0]['intent']
    for intent in comm_json['intents']:
        if intent['tag'] == tag:
            return tag
    return ''

def prnt_typess(comm_list):
    if not comm_list:
        print("Warning: comm_list is empty.")
    if len(comm_list)>1:
        for intent in comm_list[1:]:
            print(f"{intent['intent']} ==> probability: {float(intent['probability'])*100}%")

def run():
    message = 'my chest is ouchy and i feel a littel dizzy and i cant sleep'
    total_probability = 0 
    
    typclass = predict_class(message, commwords, commclasses, commmodel)
    typ = get_type(typclass, comms)
    total_probability += float(typclass[0]['probability'])
    print("====================================")
    print(f'{typ} ==> probability: {total_probability*100}%')
    prnt_typess(typclass)


if __name__ == "__main__":
    run()


'''
#sentences = sent_tokenize(message)
#results = []
#
#for sentence in sentences:
#    typclass = predict_class(sentence, commwords, commclasses, commmodel)
#    typ = get_type(typclass, comms)
#    total_probability += float(typclass[0]['probability'])
#    results.append([typ, total_probability])
#    print(typ, '\n', f'probability: {total_probability}%')
#
#print(results)
