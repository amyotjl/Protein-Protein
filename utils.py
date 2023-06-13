import torch
import numpy as np
import pandas as pd
import itertools
from scipy import spatial as sp


def hardmax(matrix, device):
    max_indices = torch.argmax(matrix, axis=1)  # Find the indices of the maximum values along each row
    hardmax_matrix = torch.zeros_like(matrix)  # Create a matrix of zeros with the same shape as the input matrix
    row_indices = torch.arange(matrix.shape[0])  # Generate row indices

    hardmax_matrix[row_indices, max_indices] = 1  # Set the values at the maximum indices to 1

    return hardmax_matrix.to(device)

# Function to one-hot encode a sequence of amino acid.
# The output is a matrix of max_length_amino_acid (1965) x unique_amino_acid (21). Sequence shorter than max_length_amino_acid are filled with 0. 
def one_hot_encode(seq, max_length, unique, padding_char = None):
    if padding_char != None:
        seq = seq.ljust(max_length, padding_char)
        unique.append(padding_char)
    matrix = np.zeros((max_length, len(unique)))
    for idx, elem in enumerate(seq):
        matrix[idx][unique.index(elem)] = 1
    return matrix.astype(np.float32)

def aa_matrix_encoded(seq, max_length, aa_vectors, padding_char = None):
    if padding_char != None:
        seq = seq.ljust(max_length, padding_char)
        
    matrix = torch.empty((max_length, len(aa_vectors)))
    for idx, elem in enumerate(seq):
        matrix[idx] = aa_vectors[elem]
    return matrix


def generate_random_matrices(unique_aa):
    res = {}
    for aa in unique_aa:
        res[aa] = torch.randn(len(unique_aa))
    return res


def mul_random_matrix(seq, max_length, random_matrices):
    seq = seq.ljust(max_length, '_')
    matrix = torch.empty((max_length, len(random_matrices.keys())))
    for idx, elem in enumerate(seq):
        matrix[idx] = random_matrices[elem]
    return matrix

def reconstruct_sequence(AA_random_matrices, output):
    sequence = ''
    # vectors = np.array(list(AA_random_matrices.values()))
    for row in output:
        sequence += min(AA_random_matrices, key=lambda key: np.linalg.norm(row - AA_random_matrices[key]))
    return sequence


def get_combinations(dictionary):
    value_lists = dictionary.values()
    combinations = list(itertools.product(*value_lists))
    keys = dictionary.keys()
    
    result = []
    for combination in combinations:
        combined_dict = dict(zip(keys, combination))
        result.append(combined_dict)
    return result

def correct_reconstructed_amino_acid(sequence, output, AA_random_matrices):
    reconstructed = list(reconstruct_sequence(AA_random_matrices, output))
    ground_truth = list(sequence.ljust(1965,'_'))
    return sum(x == y for x, y in zip(reconstructed, ground_truth))

def batch_correct_reconstructed_amino_acid(sequences, output, AA_random_matrices, longest_sequence):
    tree = sp.KDTree([vec for vec in AA_random_matrices.values()])
    closest = tree.query(output)[1]
    aminoacid = list(AA_random_matrices.keys())
    correct_aa = 0
    reconstructed_pair = []
    for idx, seq in enumerate(sequences):
        reconstructed = [aminoacid[i] for i in closest[idx]]
        seq = list(seq.ljust(longest_sequence, '_'))
        reconstructed_pair.append((seq, reconstructed))
        correct_aa += sum(x == y for x, y in zip(reconstructed, seq))

    return correct_aa, reconstructed_pair
    