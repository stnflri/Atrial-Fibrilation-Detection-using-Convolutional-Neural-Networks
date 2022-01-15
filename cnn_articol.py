
from re import M
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from sklearn.metrics import confusion_matrix, classification_report
import joblib

import pickle
import scipy.io
import numpy as np
from os import walk
import os
import csv
import pdb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = '/home/iustin/Downloads/training2017/'

batch_size = 10
epochs = 20
num_classes = 10
no_samples = 18286
no_files = 8528
dim_maxima = 18286


def test ():

    test = np.load("/home/iustin/python code/testingset.npy")
    etichete_test = np.load ("/home/iustin/python code/testinglabels.npy")

    model = pickle.load(open("model.pkl", "rb"))
    model.predict(test)

    _, accuracy = model.evaluate(test, etichete_test, batch_size=batch_size, verbose = 1)
    print ("ACCURACY FOR TEST... ")
    print (accuracy)


def label_processing():

    # noua baza de date cu etichetele 1 si O
    d = []

    with open(path + 'REFERENCE.csv') as csvfile:
        content = csv.reader (csvfile, delimiter = ',')
        for c in content:
            if c[1] == 'A':
                c[1] = 1
            else :
                c[1] = 0
            d.append(c)

    with open('db.csv', 'w', newline='') as csvfile:
        s = csv.writer(csvfile, delimiter = ',')
        for r in d:
            s.writerow (r)


def citire_date():

    # citesc datele din fisier
    d = {}

    #csv contine nume imagine si label
    with open('/home/iustin/python code/db.csv', 'r', newline='') as csvfile:
        content = csv.reader (csvfile, delimiter = ',')
        for c in content:
            d [c[0]] = c[1]
    
    p1 = int (0.9 * no_files)
    p2 = int (0.7 * p1)

    dtrain = {}
    dtest = {}
    dvalidation = {}

    for cheie, val in d.items():
        if p2 > 0:
            dtrain [path + cheie] = val
            p2 -= 1
            p1 -= 1
        elif p1 > 0:
            dvalidation [path + cheie] = val
            p1 -= 1
        else :
            dtest [path + cheie] = val
    return dtrain, dvalidation, dtest


def preprocesare_train():

    dtrain, _, _ = citire_date() 
    train = []
    etichete_train = np.array ([])
    contor = 0
    print ("Incarc etichetele si datele in memorie ....")
    print ("Start training set")

    #cheia este calea catre imagine, val este labelul
    for cheie, val in dtrain.items():
        mat = scipy.io.loadmat(cheie)
        #incarcarea continutului fisierului din cheie
        #pentru semnalele cu lungimea mai mica de dim maxima, se adauga elemente de 0 pana ce dimensiunea este egala cu dim maxima
        if len(mat['val'][0]) < dim_maxima :
            pad = [float(0)] * (dim_maxima - len(mat['val'][0]))
            train.append([float(x) for x in mat['val'][0]] + pad)
        else :
            train.append([float(x) for x in mat['val'][0]])
        etichete_train = np.append(etichete_train, float(val))
        print ("Done processing Sample {} + Label {} ...".format(contor, contor))
        contor += 1

    print ("Done traning set")
    print ("START - saving train set")
    np.save("/home/iustin/python code/trainlabels", np.array(train))
    print ("END - saving train set")
    print ("START - saving train labels")
    np.save("/home/iustin/python code/trainlabels", etichete_train)
    print ("END - saving train labels")


def preprocesare_validation():

    _, dvalidation, _ =  citire_date() 
    validation = []
    etichete_validation = np.array ([])
    contor = 0
    print ("Start validation set")

    for cheie, val in dvalidation.items():
        mat = scipy.io.loadmat(cheie)
        if len(mat['val'][0]) < dim_maxima :
            pad = [0] * (dim_maxima - len(mat['val'][0]))
            validation.append(list(mat['val'][0]) + pad)
        else :
            validation.append(list(mat['val'][0]))
        etichete_validation = np.append(etichete_validation, float(val))
        print ("Done processing Sample {} + Label {} ...".format(contor, contor))
        contor += 1

    print ("Done validation set")
    print ("START - saving validation set")
    np.save("/home/iustin/python code/validationset", np.array(validation))
    print ("END - saving validation set")
    print ("START - saving validation labels")
    np.save("/home/iustin/python code/validationslabels", np.array(etichete_validation))
    print ("END - saving validation labels")


def preprocesare_test():

    _, _, dtest =  citire_date() 
    test = []
    etichete_test = np.array ([])
    contor = 0
    print ("Start test set")

    for cheie, val in dtest.items():
        mat = scipy.io.loadmat(cheie)
        if len(mat['val'][0]) < dim_maxima :
            pad = [0] * (dim_maxima - len(mat['val'][0]))
            test.append(list(mat['val'][0]) + pad)
        else :
            test.append(list(mat['val'][0]))
        etichete_test = np.append(etichete_test, float(val))
        print ("Done processing Sample {} + Label {} ...".format(contor, contor))
        contor += 1

    print ("Done test set")
    print ("START - saving testing set")
    np.save("/home/iustin/python code/testingset", np.array(test))
    print ("END - saving testing set")
    print ("START - saving testing labels")
    np.save("/home/iustin/python code/testinglabels", np.array(etichete_test))
    print ("END - saving testing labels")


def cnn ():

    model = Sequential()

    model.add(Conv1D(27, 1, padding='causal', input_shape=(no_samples, 1), activation="relu"))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Dropout(0.25))

    model.add(Conv1D(15, 1, padding='causal', activation="relu"))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Dropout(0.25))

    model.add(Conv1D(4, 1, padding='causal', activation="relu"))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Dropout(0.25))

    model.add(Conv1D(3, 1, padding='causal', activation="relu"))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(30, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation="softmax"))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation="softmax"))
    
    return model


if __name__ == '__main__':
 
    # preprocesare date
    # preprocesare_train()
    # preprocesare_validation()
    # preprocesare_test()

    print ("Creez si compilez cnn")
    model = cnn()
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )
    model.summary()
 

    train = np.load("/home/iustin/python code/trainset.npy")
    validation = np.load("/home/iustin/python code/validationset.npy")

    etichete_train = np.load ("/home/iustin/python code/trainlabels.npy")
    etichete_validation = np.load ("/home/iustin/python code/validationslabels.npy")

    print ("START TRAIN AND VALIDATION ... ")
    model.fit(train, etichete_train, validation_data = (validation, etichete_validation), epochs=epochs, batch_size=batch_size, verbose = 1)
    print ("END")

    model = pickle.dump(model, open("model.pkl", "wb"))
    joblib.dump(model, 'modelsalvat.pkl')

    # # test()
    