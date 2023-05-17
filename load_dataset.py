import pandas as pd
import numpy as np
import joblib



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
# Function to one-hot encode a sequence of amino acid.
# The output is a matrix of max_length_amino_acid (1965) x unique_amino_acid (21). Sequence shorter than max_length_amino_acid are filled with 0. 
def one_hot_encode(seq, max_length, unique):
    matrix = np.zeros((max_length, len(unique)))
    for idx, elem in enumerate(seq):
        matrix[idx][unique.index(elem)] = 1
    return matrix.astype(np.float32)


#@title Read in file in fasta format. { display-mode: "form" }
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
                seq = seq.replace('U','X').replace('Z','X').replace('O','X')
                seqs[ uniprot_id ] += seq 

    return seqs


def load_embeddings(path):
    return joblib.load(path)
    

