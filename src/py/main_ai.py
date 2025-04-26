import json
import pickle
import numpy as np
import pyttsx3
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import os
import time  # added for measuring execution time
from dotenv import load_dotenv

#* based on Ash attempt num 4  

#! optimized dignose_ai : big o ==> O(n+p+klogk) (p is constant and k is the sort of dzs)

################
##? Immortal ?##
################

load_dotenv()
# disable GPU and logging for tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '../../data/dzs.json')
words_path = os.path.join(BASE_DIR, '../models/dzswords.pkl')
classes_path = os.path.join(BASE_DIR, '../models/dzsclasses.pkl')
model_path = os.path.join(BASE_DIR, '../models/chatbotdzs.h5')

lemmatizer = WordNetLemmatizer()
with open(data_path, 'r') as f:
    comms = json.load(f)

commwords = pickle.load(open(words_path, 'rb'))
commclasses = pickle.load(open(classes_path, 'rb'))
commmodel = load_model(model_path)

# preprocess word list
word_index_map = {word: i for i, word in enumerate(commwords)} 

# tts setup
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
    """predicts the intent using trained model"""
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
    start_time = time.time()
    predictions = predict_class(message, commmodel, word_index_map, commclasses)
    elapsed_time = (time.time() - start_time) * 1000  # convert to milliseconds

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
    """Run the model and wait for user input after processing"""
    # process an example sentence
    message = "my chest is ouchy and I feel a little dizzy and I can't sleep"
    process_sentence(message)

    boo= True
    # enter more sentences
    while boo:
        print("\nType a new sentence (or type 'exit' to quit):")
        message = input("> ").strip()
        if message.lower() == 'exit':
            print("exiting...")
            boo = False
        else:
            process_sentence(message)

if __name__ == "__main__":
    run()
