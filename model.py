import re
from flask import Markup
import numpy as np
import tensorflow as tf
import pandas as pd

model = tf.keras.models.load_model('model.h5')
go_obo_path = 'go.obo'
gp_terms_path = 'terms_file.pkl'

# Define function to convert amino acid sequence to one-hot encoding
def seq_to_onehot(seq):
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_dict = {aa: i for i, aa in enumerate(aa_list)}
    seq_onehot = np.zeros((1, 500, 21))
    for i, aa in enumerate(seq):
        if aa in aa_dict:
            seq_onehot[0, i, aa_dict[aa]] = 1
    return seq_onehot


# Define function to make predictions on a single sequence
def predict_GO(model, seq):
    seq_onehot = seq_to_onehot(seq)
    pred = model.predict(seq_onehot)
    return pred


def map_go_terms(go_obo_path, predictions):
    # Load the go.obo file and extract the relevant information
    with open(go_obo_path, 'r') as f:
        go_obo = f.read()
    go_Terms = pd.read_pickle(gp_terms_path)
    print(go_Terms)
    # Parse the file to extract the GO terms and their names
    go_terms = re.findall(r'\[Term\](.*?)\n\n', go_obo, re.DOTALL)
    term_dict = {}
    for term in go_terms:
        id = re.findall(r'^id:\s+(.*?)\n', term, re.MULTILINE)
        name = re.findall(r'^name:\s+(.*?)\n', term, re.MULTILINE)
        if id and name:
            term_dict[id[0]] = name[0]

    # Map the GO terms to their predicted values
    mapped_terms = {}
    for go_term, value in zip(go_Terms['terms'].tolist(), predictions[0]):
        if go_term in term_dict:
            mapped_terms[go_term +" - "+term_dict[go_term]] = value

    return mapped_terms


def get_results_above_threshold(mapped_terms, threshold):
    # Create an empty list to store the results
    results = []

    # Iterate over the keys and values in the mapped_dict
    for key, value in mapped_terms.items():
        # If the score for this key is above the threshold, add it to the results list
        if value > threshold:
            results.append((key, value))

    # Sort the results by descending score
    results.sort(key=lambda x: x[1], reverse=True)

    return results


seq = 'MALWMRLLPLLALLALWGPDPAAAFVNQHLEGSLCN'


def final_pred(sequence, threshold):
    pred = predict_GO(model, sequence)
    mapped_terms = map_go_terms(go_obo_path, pred)
    results = get_results_above_threshold(mapped_terms, threshold)
    results1 = beautify_predictions(results)
    return results


def beautify_predictions(predictions):
    result = []
    for prediction in predictions:
        label = prediction[0]
        confidence = prediction[1]
        formatted_string = f"Protein Function: {label} Confidence: {confidence:.2f}"
        result.append(formatted_string.replace("'", ""))
    return result


def contains_valid_amino_acids(sequence):
    # List of valid amino acids (replace with any additional valid amino acids)
    valid_amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    # Check if each character in the sequence is a valid amino acid
    for amino_acid in sequence:
        if amino_acid.upper() not in valid_amino_acids:
            return False

    # If all characters are valid amino acids, return True
    return True


def retrieve_sequence_from_fasta(fasta_string):
    # Remove leading and trailing whitespaces from the input string
    fasta_string = fasta_string.strip()

    # Check if the input contains at least two lines (header and sequence)
    if not fasta_string or '\n' not in fasta_string:
        return ""  # Return an empty string if the input is invalid

    # Split the fasta_string into lines
    lines = fasta_string.split('\n')

    # Initialize an empty sequence string
    sequence = ''

    # Check if the first line starts with the ">" symbol (header line)
    if lines[0].startswith('>'):
        # Loop through the remaining lines to extract the sequence
        for line in lines[1:]:
            # Skip any lines that start with ">" (header lines)
            if line.startswith('>'):
                break
            # Append the line (sequence) to the sequence string
            sequence += line.strip()

    return sequence




# Example Fasta format string




def extract_protein_sequence(fasta_sequence):
    protein_sequence = ""

    # Split the FASTA sequence by lines
    lines = fasta_sequence.strip().split('\n')

    # Skip the first line (header line starting with '>')
    header = lines[0]

    # Read the protein sequence line by line
    for line in lines[1:]:
        # Check if the line is not empty
        if line.strip():
            protein_sequence += line.strip()

    return protein_sequence





