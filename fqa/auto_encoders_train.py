from keras import Input, Model
from keras.layers import Dense


def auto_encoder(encoding_dim, input_image_shape, x_train, x_test):
    # this is the size of our encoded representations
    # this is our input placeholder
    input_img = Input(shape=(input_image_shape,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_image_shape, activation='sigmoid')(encoded)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    # intermediate result
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train,
                    x_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    autoencoder.save_weights('.')
