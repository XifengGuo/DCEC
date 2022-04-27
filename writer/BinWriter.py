import gzip
import os

from reader.SequenceReader import FastaEntry


def mapBinAndContigNames(fastaDict, clusters):
    global binsDict
    contigNames = list(fastaDict.keys())
    binsDict = {}
    for idx, val in enumerate(clusters):
        clusterName = str(val)
        if binsDict.get(clusterName) is None:
            binsDict[clusterName] = []
        contigName = contigNames[idx]
        binsDict[clusterName].append(contigName)
    return binsDict


def writeBins(directory, bins, fastadict, compressed=False, maxbins=250, minsize=0):
    """Writes bins as FASTA files in a directory, one file per bin.

    Inputs:
        directory: Directory to create or put files in
        bins: {'name': {set of contignames}} dictionary (can be loaded from
        clusters.tsv using vamb.cluster.read_clusters)
        fastadict: {contigname: FastaEntry} dict as made by `loadfasta`
        compressed: Sequences in dict are compressed [False]
        maxbins: None or else raise an error if trying to make more bins than this [250]
        minsize: Minimum number of nucleotides in cluster to be output [0]

    Output: None
    """

    # Safety measure so someone doesn't accidentally make 50000 tiny bins
    # If you do this on a compute cluster it can grind the entire cluster to
    # a halt and piss people off like you wouldn't believe.
    if maxbins is not None and len(bins) > maxbins:
        raise ValueError('{} bins exceed maxbins of {}'.format(len(bins), maxbins))

    # Check that the directory is not a non-directory file,
    # and that its parent directory indeed exists
    abspath = os.path.abspath(directory)
    parentdir = os.path.dirname(abspath)

    if parentdir != '' and not os.path.isdir(parentdir):
        raise NotADirectoryError(parentdir)

    if os.path.isfile(abspath):
        raise NotADirectoryError(abspath)

    if minsize < 0:
        raise ValueError("Minsize must be nonnegative")

    # Check that all contigs in all bins are in the fastadict
    allcontigs = set()

    for contigs in bins.values():
        allcontigs.update(set(contigs))

    allcontigs -= fastadict.keys()
    if allcontigs:
        nmissing = len(allcontigs)
        raise IndexError('{} contigs in bins missing from fastadict'.format(nmissing))

    # Make the directory if it does not exist - if it does, do nothing
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
    except:
        raise

    # Now actually print all the contigs to files
    for binname, contigs in bins.items():
        # Load bin into a list, decompress that bin if necessary
        bin = []
        for contig in contigs:
            entry = fastadict[contig]
            if compressed:
                uncompressed = bytearray(gzip.decompress(entry.sequence))
                entry = FastaEntry(entry.header, uncompressed)
            bin.append(entry)

        # Skip bin if it's too small
        if minsize > 0 and sum(len(entry) for entry in bin) < minsize:
            continue

        # Print bin to file
        filename = os.path.join(directory, binname + '.fna')
        with open(filename, 'w') as file:
            for entry in bin:
                print(entry.format(), file=file)