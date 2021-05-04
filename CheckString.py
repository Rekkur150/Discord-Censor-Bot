import sys
import pickle #For loading tokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

Model_Location = "./Model"
Tokenizer_Location = "./Model/tokenizer.pickle"
Input_String = sys.argv[1]

max_length = 100
padding_type = 'post'
trunc_type = 'post'


with open(Tokenizer_Location, 'rb') as handle:
    tokenizer = pickle.load(handle)

Loaded_model = tf.keras.models.load_model(Model_Location)

new_sequence = tokenizer.texts_to_sequences([Input_String])

new_padded = pad_sequences(new_sequence, maxlen = max_length, padding = padding_type, truncating = trunc_type)

new_padded = np.array(new_padded)



print(Loaded_model.predict(new_padded))
sys.stdout.flush()
