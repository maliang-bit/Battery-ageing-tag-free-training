'# -*- coding: utf-8 -*-'
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras import layers
from keras import backend as K

def custom_weighted_mse(y_true, y_pred, sample_weights):
    squared_diff = K.square(y_pred - y_true)
    weighted_squared_diff = squared_diff * sample_weights
    return K.mean(weighted_squared_diff)
def my_network(batch_size, epochs,train_x,train_y_lower,train_y_upper,y_ic,weight_all,model_path):
    weight_all = K.variable(weight_all)
    input = Input((None, train_x.shape[-1]))
    x_lstm = layers.LSTM(256)(input)
    x_lower_dense = layers.Dense(128)(x_lstm)
    x_upper_dense = layers.Dense(128)(x_lstm)
    x_ic = layers.Dense(128)(x_lstm)
    ds_lower = layers.Dense(train_y_lower.shape[-1])(x_lower_dense)
    ds_upper = layers.Dense(train_y_upper.shape[-1])(x_upper_dense)
    ds_ic = layers.Dense(y_ic.shape[-1])(x_ic)
    model = Model(input, [ds_lower, ds_upper,ds_ic])
    train_y = [train_y_lower, train_y_upper, y_ic]
    model.compile(optimizer='adam',
      loss=[lambda y_true, y_pred: custom_weighted_mse(y_true, y_pred, weight_all[:,0]),
            lambda y_true, y_pred: custom_weighted_mse(y_true, y_pred, weight_all[:,1]),
            lambda y_true, y_pred: custom_weighted_mse(y_true, y_pred, weight_all[:,2])])

    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=(model_path)
        )
    ]

    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, shuffle=True,
                        validation_split=0.1, callbacks=callbacks_list,verbose=True)
