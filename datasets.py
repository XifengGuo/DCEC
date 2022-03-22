import numpy as np
import reader.SequenceReader as sr
from keras.utils import to_categorical


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape(-1, 28, 28, 1).astype('float32')
    x = x/255.
    print('MNIST:', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float32')
    x /= 2.0
    x = x.reshape([-1, 16, 16, 1])
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y

def load_fasta():
    contigs = sr.readContigs("/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta")
    print(f'Parsed {len(contigs.keys())} contigs')
 
    # Will be translated into label y
    #first_key = list(contigs.keys())[0]
    #print(first_key)
    
    # will be the x dataset
    #first_val = list(contigs.values())[0]
    #t = bytes(first_val).decode()
    #print(t)
    
    s = map(myDecoder, list(contigs.values()))
    #print(list(s))
          
    data = list(s)
    x = np.array(data)
    #x = x.reshape(-1, 28, 28, 1).astype('float32')
    #x = x/255.
    print('FASTA:', x.shape)
    return x, None

def myMapCharsToInteger(data):
  # define universe of possible input values
  seq = 'ACTGN'
  # define a mapping of chars to integers
  char_to_int = dict((c, i) for i, c in enumerate(seq))
  # integer encode input data
  integer_encoded = [char_to_int[char] for char in data]
  return integer_encoded

def myDecoder(n):
  #return to_categorical(myMapCharsToInteger(bytes(n).decode()))
  decoded = bytes(n).decode()
  return to_categorical(myMapCharsToInteger(decoded))

