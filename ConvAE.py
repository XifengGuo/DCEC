from tensorflow import keras
import numpy as np


def CAE(input_shape=(1000, 4, 1), filters=[32, 64, 128, 10]):
    model = keras.models.Sequential()

    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    model.add(keras.layers.Conv2D(filters[0], 5, strides=(2,2), padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(keras.layers.Conv2D(filters[1], 5, strides=(2,2), padding='same', activation='relu', name='conv2'))

    model.add(keras.layers.Conv2D(filters[2], 3, strides=(2,1), padding=pad3, activation='relu', name='conv3'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=filters[3], name='embedding'))
    model.add(keras.layers.Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[1]/4), activation='relu'))

    model.add(keras.layers.Reshape((int(input_shape[0]/8), int(input_shape[1]/4), filters[2])))
    model.add(keras.layers.Conv2DTranspose(filters[1], 3, strides=(2,1), padding=pad3, activation='relu', name='deconv3'))

    model.add(keras.layers.Conv2DTranspose(filters[0], 5, strides=(2,2), padding='same', activation='relu', name='deconv2'))

    model.add(keras.layers.Conv2DTranspose(input_shape[2], 5, strides=(2,2), padding='same', name='deconv1'))
    model.summary()
    return model


if __name__ == "__main__":
    from time import time

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='usps', choices=['mnist', 'usps', 'fasta'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_dir', default='results/temp', type=str)
    args = parser.parse_args()
    print(args)

    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from datasets import load_mnist, load_usps, load_fasta

    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'usps':
        x, y = load_usps('data/usps')
    elif args.dataset == 'fasta':
        x, y = load_fasta()
    
    # define the model
    model = CAE(input_shape=x.shape[1:], filters=[32, 64, 128, 10])
    keras.utils.plot_model(model, to_file=args.save_dir + '/%s-pretrain-model.png' % args.dataset, show_shapes=True)
    model.summary()

    # compile the model and callbacks
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='mse')
    from keras.callbacks import CSVLogger

    csv_logger = CSVLogger(args.save_dir + '/%s-pretrain-log.csv' % args.dataset)

    # begin training
    t0 = time()
    model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
    print('Training time: ', time() - t0)
    model.save(args.save_dir + '/%s-pretrain-model-%d.h5' % (args.dataset, args.epochs))

    # extract features
    feature_model = keras.models.Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
    features = feature_model.predict(x)
    print('feature shape=', features.shape)

    # use features for clustering
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=args.n_clusters)

    features = np.reshape(features, newshape=(features.shape[0], -1))
    pred = km.fit_predict(features)
    from . import metrics

    print('acc=', metrics.acc(y, pred), 'nmi=', metrics.nmi(y, pred), 'ari=', metrics.ari(y, pred))
