# Imported Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ['I love my dog', 'I love my cat', 'Do you think my dog is amazing']

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>") #Setup Tokenizer
tokenizer.fit_on_texts(sentences) #Process words

word_index = tokenizer.word_index #Word indexsi

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences)

test_data = ['I love pizza', 'You are the best']

test_sequence = tokenizer.texts_to_sequences(test_data)

print(word_index)
print(sequences)
print(padded)

