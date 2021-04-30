# Imported Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import numpy as np
import pickle #For saving tokenizer


path_to_data = './Data/labeled_data_hate-speech-and-offensive-language.csv'
vocab_size = 10000
max_length = 100
trunc_type= 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000

embedding_dim = 16

all_sentences = [] #all data
labels = [] #the label

#Getting data from an csv_file
with open(path_to_data) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            all_sentences.append(row[6])
            if (row[5] == '0' or row[5] == '1'):
                labels.append(True)
            else:
                labels.append(False)

training_sentences = all_sentences[0:training_size]
testing_sentences = all_sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok) #Setup Tokenizer

tokenizer.fit_on_texts(training_sentences) #Process training data
word_index = tokenizer.word_index #Word indexsi

with open('./Model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# creating training sequences and padding them
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen = max_length, padding = padding_type, truncating=trunc_type)

# creating testing sequences and padding them
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen = max_length, padding = padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels= np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential(
        [tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
        #global average pooling
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24,activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

num_epochs = 10

history = model.fit(training_padded, training_labels, epochs = num_epochs, validation_data = (testing_padded, testing_labels))


model.save("./Model")

