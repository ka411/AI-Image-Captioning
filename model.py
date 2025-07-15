import tensorflow as tf
from tensorflow.keras import layers

def create_caption_model(vocab_size=5000, embedding_dim=256, units=512):
    image_input = layers.Input(shape=(2048,))
    image_dense = layers.Dense(embedding_dim, activation='relu')(image_input)
    image_reshaped = layers.Reshape((1, embedding_dim))(image_dense)

    caption_input = layers.Input(shape=(None,))
    embedding = layers.Embedding(vocab_size, embedding_dim)(caption_input)
    lstm = layers.LSTM(units, return_sequences=True)(layers.concatenate([image_reshaped, embedding]))
    output = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(lstm)

    model = tf.keras.Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model
