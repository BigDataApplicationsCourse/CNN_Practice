# gaurdamos la version con mejor desempeño
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# cargar train y test dataset
def load_dataset():
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encoder para la salida
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convertir integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalizar en rango 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# retorne las imagenes normalizadas
	return train_norm, test_norm

# arquitectura cnn con 3 vggs
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.3))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	# compile modelo
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# run y evalue los modelos
def run_tests():
	# cargar dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare los pixels de las imagenes
	trainX, testX = prep_pixels(trainX, testX)
	# defina mejor modelo
	model = define_model()
	# generar la instancia del modelo
	datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	# prepare un iterador para las imagenes
	it_train = datagen.flow(trainX, trainY, batch_size=64)
	# entrenar el model
	steps = int(trainX.shape[0] / 64)
	model.fit_generator(it_train, steps_per_epoch=steps, epochs=400, validation_data=(testX, testY), verbose=0)
	# guardar el model
	model.save('final_model.h5')

# punto de entrada
run_tests()
