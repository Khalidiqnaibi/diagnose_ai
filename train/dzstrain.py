# importing the required modules
import random
import json
import pickle
import numpy as np
import nltk
import sys
import io
import os

from keras import layers, optimizers, models
from nltk.stem import WordNetLemmatizer

# Set default encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '../../data/dzs.json')
words_path = os.path.join(BASE_DIR, '../../src/models/dzswords.pkl')
classes_path = os.path.join(BASE_DIR, '../../src/models/dzsclasses.pkl')
model_path = os.path.join(BASE_DIR, '../../src/models/chatbotdzs.h5')

lemmatizer = WordNetLemmatizer()

with open(data_path, encoding='utf-8') as file:
    intents = json.load(file)

# lists to store words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# process each tag
for intent in intents['intents']:
    # add tag to classes
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

    for pattern in intent['patterns']:
        # tokenize each pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list) 
        
        # add pattern and associated tag to documents
        documents.append((word_list, intent['tag']))

# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words)) 

# sort classes
classes = sorted(set(classes))

with open(words_path, 'wb') as f:
    pickle.dump(words, f)

with open(classes_path, 'wb') as f:
    pickle.dump(classes, f)

# prepare training data
training = []
output_empty = [0] * len(classes)

# create the training set
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # make output array (0s with 1 for current tag)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# shuffle the training data and convert to numpy array
random.shuffle(training)

# separate training features and labels
train_x = np.array([np.array(i[0]) for i in training])  # Input features
train_y = np.array([np.array(i[1]) for i in training])  # Output labels

# create a sequential model
model = models.Sequential()
model.add(layers.Input(shape=(len(train_x[0]),)))  # Set input shape explicitly
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(train_y[0]), activation='softmax'))

# Compile the model with adam
adam = optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# train the model
hist = model.fit(train_x, train_y, epochs=1000, batch_size=5, verbose=1)

# save the model
model.save(model_path, hist)

print("yay! training completed successfully.")
