import sys
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import warnings
warnings.filterwarnings("ignore")
SNR_array = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

Ch = sys.argv[1]
mod = sys.argv[2]
scheme = sys.argv[3]
snr = sys.argv[4]
In = sys.argv[5]
hl1 = sys.argv[6]
hl2 = sys.argv[7]
hl3 = sys.argv[8]
Out = sys.argv[9]
epoch = sys.argv[10]
batch_size = sys.argv[11]


mat = loadmat('D:/{}_DNN_Dataset_{}.mat'.format(scheme, snr))
Training_Dataset = mat['DNN_Datasets']
Training_Dataset = Training_Dataset[0, 0]
X = Training_Dataset['Train_X']
Y = Training_Dataset['Train_Y']
print('Loaded Dataset Inputs: ', X.shape)
print('Loaded Dataset Outputs: ', Y.shape)

# Normalizing Datasets
scalerx = StandardScaler()
scalerx.fit(X)
scalery = StandardScaler()
scalery.fit(Y)
XS = scalerx.transform(X)
YS = scalery.transform(Y)
XS = XS.transpose()
YS = YS.transpose()

# Build the model.
init = glorot_uniform(seed=None)
model = Sequential([
    Dense(units= int(hl1), activation='relu', input_dim=int(In),
          kernel_initializer=init,
          bias_initializer=init),
    Dense(units= int(hl2), activation='relu',
          kernel_initializer=init,
          bias_initializer=init),
    Dense(units= int(hl3), activation='relu',
          kernel_initializer=init,
          bias_initializer=init),
    Dense(units=int(Out), kernel_initializer=init,
          bias_initializer=init)
])

# Compile the model.
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
print(model.summary())

model_path = 'D:/{}_DNN_{}{}{}_{}.h5'.format(scheme, hl1,hl2,hl3, SNR_array[int(snr)-1])


checkpoint = ModelCheckpoint(model_path, monitor='val_acc',
                             verbose=1, save_best_only=True,
                             mode='max')

callbacks_list = [checkpoint]

model.fit(XS, YS, epochs=int(epoch), batch_size=int(batch_size), verbose=2, validation_split=0.15,  callbacks=callbacks_list)
