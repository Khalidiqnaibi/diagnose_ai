# importing the required modules
import random
import json
import pickle
import numpy as np
import nltk
import sys
import io

import tensorflow as tf
from keras import layers, optimizers, models
from nltk.stem import WordNetLemmatizer

# Set default encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from the JSON file
with open("C:/Users/pc/Desktop/code/diagnose_ai/data/dzs.json", encoding='utf-8') as file:
    intents = json.load(file)

# Lists to store words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# Process each intent
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # Add to words list
        
        # Add pattern and associated tag to documents
        documents.append((word_list, intent['tag']))
        
        # Add tag to classes if not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word, remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save the words and classes lists to binary files
with open('C:/Users/pc/Desktop/code/diagnose_ai/src/models/dzswords.pkl', 'wb') as f:
    pickle.dump(words, f)

with open('C:/Users/pc/Desktop/code/diagnose_ai/src/models/dzsclasses.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Prepare training data
training = []
output_empty = [0] * len(classes)

# Create the training set
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Create output array (0s with 1 for current tag)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle the training data and convert to numpy array
random.shuffle(training)

# Separate training features and labels
train_x = np.array([np.array(i[0]) for i in training])  # Input features
train_y = np.array([np.array(i[1]) for i in training])  # Output labels

# Create a models.Sequential model
model = models.Sequential()
model.add(layers.Input(shape=(len(train_x[0]),)))  # Set input shape explicitly
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save("C:/Users/pc/Desktop/code/diagnose_ai/src/models/chatbotdzs.h5", hist)

# Print successful completion
print("Yay! Training completed successfully.")
