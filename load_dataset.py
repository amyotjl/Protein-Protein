import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
from utils import one_hot_encode, aa_matrix_encoded, mul_random_matrix


# "data/AminoAcid.xlsx"
def load_amino_acid(path):
    #loading file
    amino_acid_df = pd.read_excel(path, header=None)
    amino_acid_df.columns = ['protein', 'sequence']

    amino_acid_df['sequence'] = amino_acid_df['sequence'].apply(lambda seq: list(seq))

    unique_amino_acid = list(set([item for sublist in amino_acid_df['sequence'] for item in sublist]))

    unique_amino_acid.sort()

    max_length_amino_acid = len(max(amino_acid_df['sequence'], key=len))

    amino_acid_df['seq_one_hot'] = amino_acid_df['sequence'].apply(lambda seq: one_hot_encode(seq, max_length_amino_acid, unique_amino_acid))
    return amino_acid_df

def read_fasta( fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    seqs = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                seqs[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq= ''.join( line.split() ).upper().replace("-","")
                # repl. all non-standard AAs and map them to unknown/X
                # seq = seq.replace('U','X').replace('Z','X').replace('O','X')
                seqs[ uniprot_id ] += seq 

    return seqs

def load_embeddings(path):
    return joblib.load(path)

def get_dataloader(path, AA_random_matrices, batch_size = 64, shuffle = True, train_size = 1.0, num_workers = 0):
    print(' Loading embeddings')
    embeddings_dict = load_embeddings(path)
    print(' Handling embeddings')
    sequences = [val['sequence'] for val in embeddings_dict.values()]
    print(' Sequences done')
    longest_sequence = len(max(sequences, key = len))
    print(' Found longest sequence: {}'.format(longest_sequence))

    one_hots = [mul_random_matrix(seq, longest_sequence, AA_random_matrices) for seq in sequences]
    print(' Created the label matrices')
    embbedings = [torch.from_numpy(val['emb']).unsqueeze(dim=0) for val in embeddings_dict.values()]
    print(' Embeddings done')

    if train_size != 1.0:
        train_embs, test_embs, train_labels, test_labels = train_test_split(embbedings,one_hots,train_size=train_size)
        train_dataset = list(zip(train_embs, train_labels))
        train = DataLoader(train_dataset, batch_size =  batch_size, shuffle=shuffle, num_workers=num_workers)
        
        test_dataset = list(zip(test_embs, test_labels))
        test = DataLoader(test_dataset, batch_size =  batch_size,shuffle=shuffle, num_workers=num_workers)
    else:

        train_dataset = list(zip(embbedings, one_hots))
        train = DataLoader(train_dataset, batch_size =  batch_size, shuffle=shuffle, num_workers=num_workers)
        test = None
    return train, test

def get_input_seqs_dataloader(path, batch_size = 64, shuffle = True, train_size = 0.8, num_workers = 0):

    data = joblib.load(path)
    inputs = data['inputs']
    sequences = data['sequences']

    if train_size != 1.0:
        train_inputs, test_inputs, train_seqs, test_seqs = train_test_split(
                                                inputs,
                                                sequences,
                                                train_size=train_size,
                                            )
        train_dataset = list(zip(train_inputs, train_seqs))
        train = DataLoader(train_dataset, batch_size =  batch_size, shuffle=shuffle, num_workers=num_workers)
        
        test_dataset = list(zip(test_inputs, test_seqs))
        test = DataLoader(test_dataset, batch_size =  batch_size,shuffle=shuffle, num_workers=num_workers)
    else:

        train_dataset = list(zip(inputs,sequences))
        train = DataLoader(train_dataset, batch_size =  batch_size, shuffle=shuffle, num_workers=num_workers)
        test = None
    return train, test


def create_orth_dataset(fasta_path, random_orth_matrices_path, longuest_seq, output_file):
    fasta = read_fasta(fasta_path)
    orth_matrix = joblib.load(random_orth_matrices_path)
    seqs = []
    inputs = []
    for seq in fasta.values():
        if len(seq) <= longuest_seq :
            inputs.append(aa_matrix_encoded(seq, longuest_seq, orth_matrix, '_'))
            seqs.append(seq)

    joblib.dump({"inputs":torch.stack(inputs), "sequences":seqs}, output_file)



