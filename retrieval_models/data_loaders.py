
import os


class Index:
    def __init__(self, inverted_index_positions=None, vocab=None, doc_lengths=None)->None:
        """
        inverted_index_positions: dict
            A dictionary where each key is a token and the value is a dictionary with:
            - 'df': The document frequency (number of documents in which the token occurs)
            - 'doc_info': A list of tuples, where each tuple contains:
                - [docno]: The document number
                - [positions]: list with number of times the token appears in the document
        """
        self.inverted_index_positions = {}
        self.vocab = set()
        self.doc_lengths = {}

        self.load_inverted_index_positions()
        print("done loading index positions")
        self.find_vocab()
        print("done finding vocab")
        self.load_doc_lengths()
        print("done loading doc lengths")

    def load_inverted_index_positions(self):
        """
        Load the inverted index from the index.txt file (temporary solution)
        """
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
        index_path = os.path.join(BASE_DIR, "data", "index.txt")
        with open(index_path, 'r', encoding='utf-8', errors='replace') as file:
            current_term = None
            for line in file:
                line = line.rstrip()  # remove trailing space
        
                if not line:  # in case of blank
                    continue
                
                if not line.startswith("\t"):  # this will be for new terms
                    # if the line contains a valid term:df pair
                    if ":" in line:
                        term, df = line.split(":", 1)  # Split only on the first ":"
                        current_term = term.strip()
                        try:
                            self.inverted_index_positions[current_term] = {'df': int(df.strip()), 'doc_info': []}
                        except ValueError:
                            #print(f"Skipping invalid term line: {line}")
                            current_term = None
                elif current_term:  # doc info 
                    # check if valid
                    if ":" in line.strip("\t"):
                        doc_info = line.strip("\t")  # remove tab
                        try:
                            docno, positions_str = doc_info.split(":", 1)
                            positions = list(map(int, positions_str.split(",")))  # Convert positions to integers
                            self.inverted_index_positions[current_term]['doc_info'].append((docno.strip(), positions))
                        except ValueError:
                            print(f"Skipping invalid doc_info line: {line}") #some errors with tokens

    def find_vocab(self):
        """
        Find the vocabulary from the inverted index
        """
        self.vocab = list(self.inverted_index_positions.keys())
    
    def load_doc_lengths(self):
        """
        Load the document lengths from the doc_metadata.txt file (temporary solution)
        """
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

        # Construct the correct path to doc_metadata.txt
        metadata_path = os.path.join(BASE_DIR, "data", "doc_metadata.txt")
        print(metadata_path)

        # Open the file safely
        with open(metadata_path, 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                docno, length = line.split(":")
                self.doc_lengths[docno] = int(length)
            
    def get_index(self):
        """Returns the inverted index as a dictionary."""
        return self.inverted_index_positions


    
