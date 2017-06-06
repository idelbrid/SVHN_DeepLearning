from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Reshape, Dropout, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2


def make_shallower_model_v1(input_shape):
    in0 = Input(shape=[input_shape[1], input_shape[2], input_shape[3]], name='X')
    conv1 = Conv2D(5, (3, 3), strides=(1, 1), padding='same', activation='relu')(in0)
    conv2 = Conv2D(5, (1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    max3 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv2)
    conv4 = Conv2D(15, (3, 3), strides=(1, 1), padding='same', activation='relu')(max3)
    conv5 = Conv2D(30, (1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
    max6 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv5)
    conv7 = Conv2D(15, (3, 3), strides=(1, 1), padding='same', activation='relu')(max6)
    max8 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv7)
    resh8 = Reshape([-1])(max8)
    dens9 = Dense(400, activation='relu', kernel_regularizer=l2(0.003))(resh8)
    dens9 = Dropout(0.7)(dens9)

    dig_10 = Dense(5 * 11, activation='linear', kernel_regularizer=l2(0.001))(dens9)
    resh10 = Reshape([5, 11])(dig_10)
    softmax10 = Activation('softmax')(resh10)

    model = Model(in0, outputs=[softmax10])
    opt = Adam(0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model


def make_shallower_model_v2(input_shape):
    in0 = Input(shape=[input_shape[1], input_shape[2], input_shape[3]], name='X')
    conv1 = Conv2D(15, (3, 3), strides=(1, 1), padding='same', activation='relu')(in0)  # 32 x 64 x 3
    conv2 = Conv2D(30, (1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)  # 32 x 64 x 15
    max3 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv2)  # 16 x 32 x 15
    conv4 = Conv2D(60, (3, 3), strides=(1, 1), padding='same', activation='relu')(max3)  # 16 x 32 x 45
    conv5 = Conv2D(90, (1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)  # 16 x 32 x 60
    max6 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv5)  # 8 x 16 x 90
    conv7 = Conv2D(45, (3, 3), strides=(1, 1), padding='same', activation='relu')(max6)  # 8 x 16 x 45
    max8 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv7)  # 4 x 8 x 45
    resh8 = Reshape([-1])(max8)  # 1440
    dens9 = Dense(400, activation='relu', kernel_regularizer=l2(0.003))(resh8)  # 400
    dens9 = Dropout(0.5)(dens9)  # 400

    dig_10 = Dense(5 * 11, activation='linear', kernel_regularizer=l2(0.001))(dens9)  # 55
    resh10 = Reshape([5, 11])(dig_10)
    softmax10 = Activation('softmax')(resh10)

    model = Model(in0, outputs=[softmax10])
    opt = Adam(0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model


def make_deeper_model_v1(input_shape):
    in0 = Input(shape=[input_shape[1], input_shape[2], input_shape[3]], name='X')
    conv1 = Conv2D(5, (3, 3), strides=(1, 1), padding='same', activation='relu')(in0)
    conv2 = Conv2D(25, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    max3 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv2)
    conv4 = Conv2D(100, (3, 3), strides=(1, 1), padding='same', activation='relu')(max3)
    conv5 = Conv2D(200, (5, 5), strides=(1, 1), padding='same', activation='relu')(conv4)
    max6 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv5)
    conv7 = Conv2D(200, (3, 3), strides=(1, 1), padding='same', activation='relu')(max6)
    conv8 = Conv2D(200, (1, 1), strides=(1, 1), padding='same', activation='relu')(conv7)
    max9 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv8)
    conv10 = Conv2D(100, (5, 5), strides=(1, 1), padding='same', activation='relu')(max9)
    conv11 = Conv2D(50, (1, 1), strides=(1, 1), padding='same', activation='relu')(conv10)
    max12 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv11)

    resh12 = Reshape([-1])(max12)
    dens13 = Dense(800, activation='relu', kernel_regularizer=l2(0.0001))(resh12)
    dens13 = Dropout(0.4)(dens13)
    dens14 = Dense(400, activation='relu', kernel_regularizer=l2(0.0001))(dens13)
    dens14 = Dropout(0.5)(dens14)

    dig_15 = Dense(5 * 11, activation='linear', kernel_regularizer=l2(0.0001))(dens14)
    resh15 = Reshape([5, 11])(dig_15)
    softmax15 = Activation('softmax')(resh15)

    model = Model(in0, outputs=[softmax15])
    opt = Adam(0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    return model


def make_deeper_model_v2(input_shape):
    in0 = Input(shape=[input_shape[1], input_shape[2], input_shape[3]], name='X')
    conv1 = Conv2D(5, (3, 3), strides=(1, 1), padding='same', activation='relu')(in0)
    conv2 = Conv2D(25, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    max3 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv2)
    conv4 = Conv2D(100, (3, 3), strides=(1, 1), padding='same', activation='relu')(max3)
    conv5 = Conv2D(200, (5, 5), strides=(1, 1), padding='same', activation='relu')(conv4)
    max6 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv5)
    conv7 = Conv2D(200, (3, 3), strides=(1, 1), padding='same', activation='relu')(max6)
    conv8 = Conv2D(200, (1, 1), strides=(1, 1), padding='same', activation='relu')(conv7)
    max9 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv8)
    conv10 = Conv2D(200, (5, 5), strides=(1, 1), padding='same', activation='relu')(max9)
    conv11 = Conv2D(100, (1, 1), strides=(1, 1), padding='same', activation='relu')(conv10)
    max12 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv11)

    resh12 = Reshape([-1])(max12)
    dens13 = Dense(800, activation='relu', kernel_regularizer=l2(0.0001))(resh12)
    dens13 = Dropout(0.4)(dens13)
    dens14 = Dense(300, activation='relu', kernel_regularizer=l2(0.0001))(dens13)
    dens14 = Dropout(0.5)(dens14)

    dig_15 = Dense(5 * 11, activation='linear', kernel_regularizer=l2(0.0001))(dens14)
    resh15 = Reshape([5, 11])(dig_15)
    softmax15 = Activation('softmax')(resh15)

    model = Model(in0, outputs=[softmax15])
    opt = Adam(0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    return model


