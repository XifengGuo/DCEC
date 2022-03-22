import reader.SequenceReader as sr

contigs = sr.readContigs("/share_data/cami_low/CAMI_low_RL_S001__insert_270_GoldStandardAssembly.fasta")
print(f'Parsed {len(contigs.keys())} contigs')
