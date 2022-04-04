from base64 import decode
import numpy as np
import tensorflow

import reader.SequenceReader as sr
import matplotlib.pyplot as plt


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
    lst = list(contigs.values())
    # an attempt to display graph of seq. lengths, so that we can see the extreme values and delete them:
    # the maximum length is over 1 mil, the sequences legth is not balanced:
    
    # maxlength = max(data)
    # names = np.arange(19499)
    # plt.bar(data, names)
    # plt.show()
      
    s = map(myDecoder, lst)
              
    data = list(s)
    
    # maxlength = max(data)
    for i in data:
        if i.shape != (1000, 4):
            data.remove(i)
    
    x = np.array(data)
    print('FASTA:', x.shape)
    x = x.reshape(-1, 1000, 4, 1).astype('float32')
    print('FASTA:', x.shape)
    return x, None

def myMapCharsToInteger(data):
  # define universe of possible input values
  seq = 'ACTGO'
  # define a mapping of chars to integers
  char_to_int = dict((c, i) for i, c in enumerate(seq))
  #print("Chars to int")
  #print(char_to_int)
  # integer encode input data
  integer_encoded = [char_to_int[char] for char in data]
  return integer_encoded

def setCorrectSequenceLength(n, size):
    #an ugly fix - sequences that didn't contain O or N were one hot encoded in a shape (length, 4) or (length, 5) which was causing problems with reshaping
    #se we make sure there is at least one N and one O in every sequence
    if len(n) > size:
        return n[:size]
    elif len(n) < size:
        return n.ljust(size, "O")
    return n

def setSequenceLength(n, size):
    if len(n) > size:
        return n[:size]
    elif len(n) < size:
        padding = np.array([[-1., -1., -1., -1.]] *(size - len(n)))
        n = np.concatenate((n, padding), axis=0)
    return n


def myDecoder(n):
  """
  decoded = bytes(n).decode()
  most_common_nucleotide = max(set(decoded), key=decoded.count)
  decoded = setCorrectSequenceLength(decoded, 1000)
  decoded = [most_common_nucleotide if x == 'N' else x for x in decoded]
  print(decoded)
  return to_categorical(myMapCharsToInteger(decoded), num_classes=5)
  """
  decoded = bytes(n).decode()
  most_common_nucleotide = max(set(decoded), key=decoded.count)
  decoded = [most_common_nucleotide if x == 'N' else x for x in decoded]
  encodings = tensorflow.keras.utils.to_categorical(myMapCharsToInteger(decoded), num_classes=4)
  encodings = setSequenceLength(encodings, 1000)
  return encodings

 
def strLengths(n):
  decoded = bytes(n).decode()
  return len(decoded)