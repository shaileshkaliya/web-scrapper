from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.tokenize import word_tokenize
from nltk import CFG
from nltk.corpus import stopwords
import string
from mangum import Mangum

# Initialize FastAPI app
app = FastAPI()

handler = Mangum(app);
# Function to download NLTK resources
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')


# Download NLTK resources
download_nltk_resources()


# Define request body model
class TextRequest(BaseModel):
    text: str

# Tokenize text function
def tokenize_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and token not in string.punctuation]
    return filtered_tokens

# POS tagging function
def pos_tags(tokens):
    pos_tags = nltk.pos_tag(tokens)
    determiners = []
    nouns = []
    verbs = []
    prepositions = []
    pronouns = []
    for token, pos_tag in pos_tags:
        if pos_tag.startswith('DT'):
            determiners.append(token.lower())
        elif pos_tag.startswith('NN'):
            nouns.append(token.lower())
        elif pos_tag.startswith('VB'):
            verbs.append(token.lower())
        elif pos_tag.startswith('IN'):
            prepositions.append(token.lower())
        elif pos_tag.startswith('PRP'):
            pronouns.append(token.lower())
    determiners = sorted(set(determiners))
    nouns = sorted(set(nouns))
    verbs = sorted(set(verbs))
    prepositions = sorted(set(prepositions))
    pronouns = sorted(set(pronouns))
    return determiners, nouns, verbs, prepositions, pronouns

# Generate grammar function
def generate_grammar(determiners, nouns, verbs, prepositions):
    grammar_rules = ""
    for det in determiners:
        for noun in nouns:
            det = det.replace("'", "")
            noun = noun.replace("'", "")
            grammar_rules += f"NP -> '{det}' '{noun}'\n"
    for verb in verbs:
        for noun in nouns:
            verb = verb.replace("'", "")
            noun = noun.replace("'", "")
            grammar_rules += f"VP -> '{verb}' 'NP'\n"
    for prep in prepositions:
        for noun in nouns:
            prep = prep.replace("'", "")
            noun = noun.replace("'", "")
            grammar_rules += f"PP -> '{prep}' 'NP'\n"
    for word in determiners + nouns + verbs + prepositions:
        if word.isalpha():
            word = word.replace("'", "")
            grammar_rules += f"{word.upper()} -> '{word}'\n"
    grammar_rules += "S -> NP VP\n"
    grammar = CFG.fromstring(grammar_rules)
    return grammar

# Encode and decode text function
def encode_decode_text(text):
    encoded_text = text.encode('utf-8')
    decoded_text = encoded_text.decode('utf-8')
    return encoded_text, decoded_text, text == decoded_text

# Analyze frequency distribution function
def analyze_frequency_distribution(tokens):
    word_freq_dist = nltk.FreqDist(tokens)
    most_common = word_freq_dist.most_common(20)
    return most_common

# Define routes
@app.post("/process-text/")
async def process_text(request: TextRequest):
    try:
        text = request.text
        tokens = tokenize_text(text)
        determiners, nouns, verbs, prepositions, pronouns = pos_tags(tokens)
        grammar = generate_grammar(determiners, nouns, verbs, prepositions)
        encoded_text, decoded_text, is_same_text = encode_decode_text(text)
        word_freq_dist = analyze_frequency_distribution(tokens)
        item = {
            "text": text,
            "tokens": tokens,
            "pos_tags": {
                "determiners": determiners,
                "nouns": nouns,
                "verbs": verbs,
                "prepositions": prepositions,
                "pronouns": pronouns
            },
            "grammar": str(grammar),  # Convert CFG object to string
            "encoded_text": encoded_text.decode('utf-8'),  # Convert bytes to string
            "decoded_text": decoded_text,
            "is_same_text": is_same_text,
            "word_freq_dist": dict(word_freq_dist)  # Convert FreqDist object to dictionary
        }
        return {"message": "SUCCESSFULLY PROCESSED", "response": item}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {e}")

# Documentation for the API endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to the Text Processing API!"}

@app.get("/docs")
async def get_docs():
    return {"message": "API documentation"}

# If you want to run the FastAPI application, you can add the following block:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)


