import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class_names = {0: "<start>", 1: "<end>"}  # Example, to be replaced by real vocab

model = tf.keras.models.load_model("image_caption_model.h5")

def generate_caption(image_features, tokenizer, max_length=30):
    input_seq = [tokenizer.word_index['<start>']]
    for _ in range(max_length):
        seq = pad_sequences([input_seq], maxlen=max_length, padding='post')
        yhat = model.predict([image_features, seq])
        predicted_id = np.argmax(yhat[0, -1, :])
        word = tokenizer.index_word.get(predicted_id, '')
        if word == '<end>':
            break
        input_seq.append(predicted_id)
    return ' '.join([tokenizer.index_word.get(i, '') for i in input_seq])
