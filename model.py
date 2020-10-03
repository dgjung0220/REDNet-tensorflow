# Donggoo Jung (dgjung0220@gmail.com)
# https://dgjung.me


import tensorflow as tf

def REDNet(num_layers):

    conv_layers = []
    deconv_layers = []
    residual_layers = []

    inputs = tf.keras.Input(shape=(None, None, 3))

    conv_layers.append(tf.keras.layers.Conv2D(3, kernel_size=3, padding='same', activation='relu'))
    for i in range(num_layers-1):
        conv_layers.append(tf.keras.layers.Conv2D(64, kernel_size=64, padding='same', activation='relu'))
        deconv_layers.append(tf.keras.layers.Conv2DTranspose(64, kernel_size=3, padding='same'))
    deconv_layers.append(tf.keras.layers.Conv2DTranspose(3, kernel_size=3, padding='same'))

    x = conv_layers[0](inputs)
    for i in range(num_layers-1):
        x = conv_layers[i+1](x)
        if i % 2 == 0:
            residual_layers.append(x)

    for i in range(num_layers-1):
        if i % 2 == 1:
            x = tf.keras.layers.Add()([x, residual_layers.pop()])
            x = tf.keras.layers.Activation('relu')(x)

        x = deconv_layers[i](x)
    x = deconv_layers[-1](x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model