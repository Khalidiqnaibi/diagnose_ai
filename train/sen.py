import pprint
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree


def extract_symptoms(text):
    stop_words = set(stopwords.words('english'))

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Extract symptoms from each sentence
    symptoms = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        # Filter out stopwords
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # POS tagging (Part of Speech tagging)
        tagged = pos_tag(filtered_words)

        # Use chunking to find noun phrases (possible symptoms)
        chunked = ne_chunk(tagged, binary=False)
        
        for chunk in chunked:
            if type(chunk) == Tree and chunk.label() == 'NP':
                symptom = " ".join([token for token, pos in chunk.leaves()])
                symptoms.append(symptom)

    return symptoms

def extract_disease_name(text):
    # Assuming disease name is mentioned in the first sentence
    first_sentence = sent_tokenize(text)[0]
    
    # Extract proper nouns as potential disease names
    words = word_tokenize(first_sentence)
    tagged = pos_tag(words)
    
    disease_name = ""
    for word, tag in tagged:
        if tag == 'NNP':  # Proper noun
            disease_name += word + " "
    
    return disease_name.strip()

def process_input(input_text):
    # Split the question from the text (e.g., "Symptoms of ...?")
    lines = input_text.split("?")
    
    # Extract disease name from the question part
    disease_name = extract_disease_name(lines[0])

    # Extract symptoms from the description part
    symptoms = extract_symptoms(lines[1])
    
    # Format the result
    result = {
        "tag": disease_name,
        "patterns": symptoms
    }
    
    return result

# Example input
input_text = """Hurler syndrome?
Symptoms of Hurler syndrome range in severity and are unique to each person diagnosed with the condition. Symptoms begin in early childhood and continue through adolescence.

A symptom of Hurler syndrome that sets it apart from other levels of mucopolysaccharidosis type I (MPS 1) is early childhood developmental delays and a progressive decline in how your child can learn and retain information. Mild cases of MPS 1 don’t affect a child’s intelligence.

Symptoms of Hurler syndrome could include:

Heart valve problems (cardiomyopathy).
Hearing loss.
Buildup of cerebrospinal fluid around your child's brain (hydrocephalus).
Enlarged organs like connective tissues, tonsils, muscles, heart, liver and spleen.
Vision problems (glaucoma).
Joint problems (tight muscles, carpal tunnel and joint disease).
Respiratory infections, sleep apnea and difficulty breathing.
Hernias.
Physical characteristics
During a child’s first year, physical symptoms of Hurler syndrome will appear. These characteristics include:

Short stature.
Bones forming incorrectly (dysostosis).
Rounding curve of your child's upper back (thoracic-lumbar kyphosis).
Excessive hair growth.
What causes Hurler syndrome?
A mutation of the IDUA gene causes Hurler syndrome. The IDUA gene is responsible for creating lysosomal enzymes, which break down waste in cells. When the IDUA gene doesn't create enough enzymes, toxic waste collects in cells, causing them to die or not function properly. 
When your cells can’t get rid of waste, symptoms of Hurler syndrome occur."""

# Process the input
result = process_input(input_text)

# Print the result
pprint.pprint(result)
