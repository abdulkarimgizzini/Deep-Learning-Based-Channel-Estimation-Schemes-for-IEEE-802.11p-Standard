from scipy.io import loadmat 
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys
from sklearn.preprocessing import StandardScaler
import scipy.io
from scipy.io import loadmat
import warnings
warnings.filterwarnings("ignore")

# General Parameters
configuration_mode = len(sys.argv)
SNR_index = np.arange(0, 45, 5)


if configuration_mode == 13:
    # We are running the training phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    modulation_order = sys.argv[3]
    scheme = sys.argv[4]
    training_snr = sys.argv[5]
    dnn_input = sys.argv[6]
    hidden_layer1 = sys.argv[7]
    hidden_layer2 = sys.argv[8]
    hidden_layer3 = sys.argv[9]
    dnn_output = sys.argv[10]
    epoch = sys.argv[11]
    batch_size = sys.argv[12]

    mat = loadmat('./{}_{}_{}_{}_DNN_training_dataset_{}.mat'.format(mobility, channel_model, modulation_order, scheme, training_snr))
    Dataset = mat['DNN_Datasets']
    Dataset = Dataset[0, 0]
    X = Dataset['Train_X']
    Y = Dataset['Train_Y']
    print('Loaded Dataset Inputs: ', X.shape)  # Size: Training_Samples x 2Kon
    print('Loaded Dataset Outputs: ', Y.shape)  # Size: Training_Samples x 2Kon

    # Normalizing Datasets
    scalerx = StandardScaler()
    scalerx.fit(X)
    scalery = StandardScaler()
    scalery.fit(Y)
    XS = scalerx.transform(X)
    YS = scalery.transform(Y)

    # Build the model.
    init = TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    model = Sequential([
        Dense(units=int(hidden_layer1), activation='relu', input_dim=int(dnn_input), kernel_initializer=init,
              bias_initializer=init),
        Dense(units=int(hidden_layer2), activation='relu', kernel_initializer=init, bias_initializer=init),
        Dense(units=int(hidden_layer3), activation='relu', kernel_initializer=init, bias_initializer=init),
        Dense(units=int(dnn_output), kernel_initializer=init, bias_initializer=init)
    ])

    # Compile the model.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    print(model.summary())

    model_path = './{}_{}_{}_{}_DNN_{}.h5'.format(mobility, channel_model, modulation_order, scheme, training_snr)
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    RLearningRate = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=20, min_lr=0.000001, min_delta=0.002)
    es = EarlyStopping(monitor="val_loss", patience=20, verbose=2, mode="min")
    callbacks_list = [checkpoint, RLearningRate, es]

    model.fit(XS, YS, epochs=int(epoch), batch_size=int(1), verbose=1, validation_split=0.25, callbacks=callbacks_list)

else:
    # We are running the testing phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    modulation_order = sys.argv[3]
    scheme = sys.argv[4]
    testing_snr = sys.argv[5]

    for j in SNR_index:
        mat = loadmat('./{}_{}_{}_{}_DNN_testing_dataset_{}.mat'.format(mobility, channel_model, modulation_order, scheme, j))
        Dataset = mat['DNN_Datasets']
        Dataset = Dataset[0, 0]
        X = Dataset['Test_X']
        Y = Dataset['Test_Y']
        print('Loaded Dataset Inputs: ', X.shape)
        print('Loaded Dataset Outputs: ', Y.shape)

        # Normalizing Datasets
        scalerx = StandardScaler()
        scalerx.fit(X)
        scalery = StandardScaler()
        scalery.fit(Y)
        XS = scalerx.transform(X)
        YS = scalery.transform(Y)

        model = load_model('./{}_{}_{}_{}_DNN_{}.h5'.format(mobility, channel_model, modulation_order, scheme, testing_snr))
        # Testing the model
        Y_pred = model.predict(XS)

        Original_Testing_X = scalerx.inverse_transform(XS)
        Original_Testing_Y = scalery.inverse_transform(YS)
        Prediction_Y = scalery.inverse_transform(Y_pred)

        result_path = './{}_{}_{}_{}_DNN_Results_{}.pickle'.format(mobility, channel_model, modulation_order, scheme, j)
        dest_name = './{}_{}_{}_{}_DNN_Results_{}.mat'.format(mobility, channel_model, modulation_order, scheme, j)
        with open(result_path, 'wb') as f:
            pickle.dump([Original_Testing_X, Original_Testing_Y, Prediction_Y], f)

        a = pickle.load(open(result_path, "rb"))
        scipy.io.savemat(dest_name, {
            '{}_DNN_test_x_{}'.format(scheme, j): a[0],
            '{}_DNN_test_y_{}'.format(scheme, j): a[1],
            '{}_DNN_corrected_y_{}'.format(scheme, j): a[2]
        })
        print("Data successfully converted to .mat file ")
        os.remove(result_path)
