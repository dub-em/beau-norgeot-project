from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch, spacy, re

#python -m spacy download en_core_web_sm, to download the en_core_web_sm dictionary.

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Load BioBERT tokenizer and model into the transformer object from local repository
tokenizer = AutoTokenizer.from_pretrained("./biobert_model/", local_files_only=True)
model = AutoModel.from_pretrained("./biobert_model/", local_files_only=True)


#Cleaning sentence of puntuations
def clean_sentence(sentence):
    """Function to clean tweet by removing punctuations
    Parameters:
        sentence (string): The sentence
    """
    
    sentence = re.sub(r"[^\w\s\@]","",sentence) # removes punctuation
    sentence = sentence.strip()
    return sentence


#Lemmatize sentence to improve uniformity to create a more accurate similarity
def lemmatize(sentence, nlp):
    # Tokenize and lemmatize the sentence
    lemmatized_words = [token.lemma_ for token in nlp(sentence)]
    return lemmatized_words


# Function to calculate Jaccard similarity
def jaccard_similarity(str1, str2):
    if ('list' in str(type(str1))) & ('list' in str(type(str2))):
        set1 = set(str1)
        set2 = set(str2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        precision = len(intersection)/len(set2)
        recall = len(intersection)/len(set1)
        if (precision == 0) & (recall == 0):
            f1_score = 'Nan'
        else:
            f1_score = 2*((precision*recall)/(precision+recall))
        jacc_sim = len(intersection)/len(union) #Same as acuracy
        return jacc_sim, precision, recall, f1_score 
    elif ('str' in str(type(str1))) & ('atr' in str(type(str2))):
        set1 = set(str1.split())
        set2 = set(str2.split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        precision = len(intersection)/len(set2)
        recall = len(intersection)/len(set1)
        if (precision == 0) & (recall == 0):
            f1_score = 'Nan'
        else:
            f1_score = 2*((precision*recall)/(precision+recall))
        jacc_sim = len(intersection)/len(union) #Same as accuracy
        return jacc_sim, precision, recall, f1_score
    else:
        print('Please make sure the two inputs to be compared are both either lists or strings.')
        return None
    

def biobert_similarity(text1, text2, model, tokenizer):
    # Tokenize the sentence
    tokens1 = tokenizer(text1, return_tensors="pt")
    tokens2 = tokenizer(text2, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs1 = model(**tokens1)
        outputs2 = model(**tokens2)

    #Extract the sentence embeddings (last hidden states of all tokens)
    sentence_embedding1 = outputs1.last_hidden_state.mean(dim=1)  # Mean pooling
    sentence_embedding2 = outputs2.last_hidden_state.mean(dim=1)

    cos_sim = cosine_similarity(sentence_embedding1, sentence_embedding2)[0][0]
    return cos_sim


def ensemble_similarity(text1, text2, nlp, model, tokenizer):
    '''This ensemble combines two evaluation methods (Jaccard Similarity and BioBERT-Cosine Similarity) to score similarity
    between two texts.
    
    -Jaccard: Important because accuracy of details is important given the sensitivity of detail being represented, even though jaccard can be strict regardles of accuracy of details and context
    -BioBERT: Important because even in instance where details are correct and context is aptly represented, slight variations in words generated or the way they are phrased can lead to negative score by Jaccard.
    -More testing and comaprison to human evaluation needed to aptly determine the best weight distribution.'''
    
    nlp = nlp
    model = model 
    tokenizer = tokenizer
    
    #Clean sentences using regex
    cleantext1 = clean_sentence(text1)
    cleantext2 = clean_sentence(text2)
    
    #Lematize the words in the sentences for more uniformity
    lemmatext1 = lemmatize(cleantext1, nlp)
    lemmatext2 = lemmatize(cleantext2, nlp)
    
    #Similarity scores
    jacc_sim, precision, recall, f1_score = jaccard_similarity(lemmatext1, lemmatext2) #Jaccard Similarity is calculated on the transformed sentences
    biobert_sim = biobert_similarity(text1, text2, model, tokenizer) #BioBERT-Cosine Similarity is calculated on the untransformed sentence
    
    #Ensemble (Weighted Average) of the Jaccard and BioBERT-Cosine Similarities is calculated with more weight on teh Jaccard similarity
    #More weight is given to Jaccard similarity because even though exactness of words can be a limited measure, it is still more important than just representing the general concept
    ensemble_sim = 0.65*jacc_sim + 0.35*biobert_sim
    return jacc_sim, biobert_sim, ensemble_sim, precision, recall, f1_score