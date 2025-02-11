from spellchecker import SpellChecker
from textblob import TextBlob

def load_dictionary():
  
    """Loads a dictionary using the spellchecker library. This will be done in advance."""
  
    return SpellChecker()

def find_closest_words(query, spell):
    """Finds the closest words for each word in the query using the spellchecker library."""
    words = query.lower().split()
    corrected_words = [spell.correction(word) if word not in spell else word for word in words]
    return " ".join(corrected_words)

def correct_with_textblob(query):
    """Uses TextBlob for spell correction."""
    return str(TextBlob(query).correct())

if __name__ == "__main__":
    spell = load_dictionary()
    
    query = input("Enter a query: ")
    corrected_phrase_spellchecker = find_closest_words(query, spell)
    corrected_phrase_textblob = correct_with_textblob(query)
    
    if corrected_phrase_spellchecker == corrected_phrase_textblob:
        print(f"Suggested correction: {corrected_phrase_spellchecker}")
    else:
        print(f"SpellChecker suggests: {corrected_phrase_spellchecker}")
        print(f"TextBlob suggests: {corrected_phrase_textblob}")
