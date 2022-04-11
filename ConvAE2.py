from tensorflow import keras
import numpy as np

from reader.DataGenerator import DataGenerator


def CAE2(filters=[32, 64, 128, 10], contig_len=1000):
    input_shape = (contig_len, 4, 1)
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=-1., input_shape=input_shape))
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    model.add(keras.layers.Conv2D(filters[0], 5, strides=(2, 2), padding='same', activation='relu', name='conv1',
                                  input_shape=input_shape))

    model.add(keras.layers.Conv2D(filters[1], 5, strides=(2, 2), padding='same', activation='relu', name='conv2'))

    model.add(keras.layers.Conv2D(filters[2], 3, strides=(2, 1), padding=pad3, activation='relu', name='conv3'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=filters[3], name='embedding'))
    model.add(
        keras.layers.Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[1] / 4), activation='relu'))

    model.add(keras.layers.Reshape((int(input_shape[0] / 8), int(input_shape[1] / 4), filters[2])))
    model.add(
        keras.layers.Conv2DTranspose(filters[1], 3, strides=(2, 1), padding=pad3, activation='relu', name='deconv3'))

    model.add(
        keras.layers.Conv2DTranspose(filters[0], 5, strides=(2, 2), padding='same', activation='relu', name='deconv2'))

    model.add(keras.layers.Conv2DTranspose(input_shape[2], 5, strides=(2, 2), padding='same', name='deconv1'))
    model.summary()
    return model


def load_fasta():
    x = get_sequence_samples()
    return x, None


if __name__ == "__main__":
    from time import time

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--n_clusters', default=60, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_dir', default='results/temp1', type=str)
    args = parser.parse_args()
    print(args)

    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from datasets import get_sequence_samples

    x, y = load_fasta()

    # define the model
    model = CAE2(filters=[32, 64, 128, 10], contig_len=20000)
    keras.utils.plot_model(model, to_file=args.save_dir + '/fasta-pretrain-model.png', show_shapes=True)
    model.summary()

    # compile the model and callbacks
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='mse')
    from keras.callbacks import CSVLogger

    csv_logger = CSVLogger(args.save_dir + '/fasta-pretrain-log.csv')

    # begin training
    t0 = time()
    data_generator = DataGenerator(x, batch_size=args.batch_size, contig_len=20000)
    model.fit(x=data_generator, epochs=args.epochs, callbacks=[csv_logger])
    print('Training time: ', time() - t0)
    model.save(args.save_dir + '/fasta-pretrain-model-%d.h5' % args.epochs)

    # extract features
    feature_model = keras.models(inputs=model.input, outputs=model.get_layer(name='embedding').output)
    features = feature_model.predict(x)
    print('feature shape=', features.shape)

    # use features for clustering
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=args.n_clusters)

    features = np.reshape(features, newshape=(features.shape[0], -1))
    pred = km.fit_predict(features)
    from . import metrics

    print('acc=', metrics.acc(y, pred), 'nmi=', metrics.nmi(y, pred), 'ari=', metrics.ari(y, pred))
