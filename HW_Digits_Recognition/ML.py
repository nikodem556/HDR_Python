import tensorflow as tf # wykorzystamy do pobrania bazy danych MNIST
import numpy as np
import matplotlib.pyplot as plt
import random

#Pytorch tez mozna, IRIS Database
# pobieramy dane MNIST z pomoca "tf", train 60k, test 10k
(x_train, lbl_train), (x_test, lbl_test) = tf.keras.datasets.mnist.load_data()
# ich rozmiar to 28x28 px

# ZMNIEJSAZMY LICZBE PROBEK DO SZYBKIEGO PISANIA KODU, 800 i 200
x_train = x_train[:800]
x_test = x_test[:200]

# normalizacja danych do zakresu [0, 1]
x_train = x_train/255.0
x_test = x_test/255.0

# wyświetlanie losowego obrazu
r1 = random.randint(0,800)
plt.imshow(x_train[r1], cmap="gray")
plt.title(f"Label: {lbl_train[r1]}")
plt.show()

# przygotowanie danych dla ML
x_train_flat = x_train.reshape(x_train.shape[0], -1)    # spłaszczanie do wektorów 1D
x_test_flat = x_test.reshape(x_test.shape[0], -1)   # -1 samo wylicza 784 czyli 28x28

# funkcja na one-hot encoding etykiet
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

lbl_train_encoded = one_hot_encode(lbl_train) # kazdy bedzie mial teraz 60k wektorow, po 10
lbl_test_encoded = one_hot_encode(lbl_test)   # zer z jedynka, gdzie jego lbl

print("rozmiar danych treningowych: ", x_train_flat.shape)
print("rozmiar danych testowych: ", x_test_flat.shape)

# ALGORYTM PCA/SVD - redukcja wielokrotności, pozwala zmniejszyć wymiarowośc danych, zmniejsza ale ciagle zachowując dane