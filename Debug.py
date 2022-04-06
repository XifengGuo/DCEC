import reader.SequenceReader as sr
from DCEC import DCEC
from datasets import load_mnist, load_fasta
import metrics
import tensorflow as tf
from writer.BinWriter import writeBins, mapBinAndContigNames

contigs = sr.readContigs("/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta")
print(f'Parsed {len(contigs.keys())} contigs')
print(f'GPUs Available: {tf.config.list_physical_devices("GPU")}')

# x, y = load_fasta()
# dcec = DCEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=60)
# dcec.model.load_weights("results/fasta/dcec_model_final.h5")
# clusters = dcec.predict(x)
#
# fasta = "/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta"
# # fastaDict = sr.readContigs(fasta, numberOfSamples=10000, onlySequence=False)
# fastaDict = sr.readContigs(fasta, onlySequence=False)
# binsDict = mapBinAndContigNames(fastaDict, clusters)
# writeBins("results/fasta/bins2", bins=binsDict, fastadict=fastaDict)
# print(f'predict size: ', len(clusters))

# print('acc = %.4f, nmi = %.4f, ari = %.4f' % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)))