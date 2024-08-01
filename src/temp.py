import spacy

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    # Process the text with spacy
    doc = nlp(text)
    # Extract sentences
    return [sent.text for sent in doc.sents]

text = "Dr. Smith is an expert in AI. He works at OpenAI. Mr. A. B. Johnson joined him in 2022. They both research NLP."
sentences = split_sentences(text)

for i, sentence in enumerate(sentences):
    print(f"Sentence {i+1}: {sentence}")

