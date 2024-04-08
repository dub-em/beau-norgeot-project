from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch, spacy, re, time

#"python -m spacy download en_core_web_sm": to download the en_core_web_sm dictionary.

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Load BioBERT tokenizer and model into the transformer object from local repository
tokenizer = AutoTokenizer.from_pretrained("./biobert_model/", local_files_only=True)
model = AutoModel.from_pretrained("./biobert_model/", local_files_only=True)


# Cleaning sentence of puntuations
def clean_sentence(sentence):
    '''Function to clean tweet by removing punctuations.
    Parameters:
        sentence (string): The sentence
    '''
    
    sentence = re.sub(r"[^\w\s\@]","",sentence) # removes punctuation
    sentence = sentence.strip()
    return sentence


# Lemmatize sentence to improve uniformity to create a more accurate similarity
def lemmatize(sentence, nlp):
    '''Function to lemmatize words in the sentence. Lemmatizing a word reduces it to its base
    word. Lemmatizing is preferable to Stemming in this case because sensitivity and accuracy of
    information is important.

    Parameters:
        sentence (string): The sentence
        nlp (object): The spacy dictionary for lemmatizing
    '''

    # Tokenize and lemmatize the sentence
    lemmatized_words = [token.lemma_ for token in nlp(sentence)]
    return lemmatized_words


# Function to calculate Jaccard similarity
def jaccard_similarity(str1, str2):
    '''Function to compute the Jaccard Similarity between two texts.

    Parameters:
        str1 (string/list): The sentence or list of words in the sentence
        str2 (string/list): The sentence or list of words in the sentence
    '''

    if ('list' in str(type(str1))) & ('list' in str(type(str2))):
        # If input is a list, then the list of words in the strings are converted to unique list of words
        set1 = set(str1)
        set2 = set(str2)

        intersection = set1.intersection(set2) # Intersection of the two sets
        union = set1.union(set2) # Union of the two sets
        disjoint = set1 ^ set2
        precision = len(intersection)/len(set2) # Precision of the LLM's prediction
        recall = len(intersection)/len(set1) # Recall of the LLM's prediction
        # F1-Score of the LLM's prediction
        if (precision == 0) & (recall == 0):
            f1_score = 'Nan'
        else:
            f1_score = 2*((precision*recall)/(precision+recall))
        onlyin_original = list(disjoint.intersection(set1))
        onlyin_generated = list(disjoint.intersection(set2))

        jacc_sim = len(intersection)/len(union) # Same as acuracy
        return jacc_sim, precision, recall, f1_score, onlyin_original, onlyin_generated
    elif ('str' in str(type(str1))) & ('str' in str(type(str2))):
        # If input is a string, then the string is first converted to a list of word, and then to a unique list of words
        set1 = set(str1.split())
        set2 = set(str2.split())

        intersection = set1.intersection(set2)
        union = set1.union(set2)
        disjoint = set1 ^ set2
        precision = len(intersection)/len(set2)
        recall = len(intersection)/len(set1)
        if (precision == 0) & (recall == 0):
            f1_score = 'Nan'
        else:
            f1_score = 2*((precision*recall)/(precision+recall))
        onlyin_original = list(disjoint.intersection(set1))
        onlyin_generated = list(disjoint.intersection(set2))

        jacc_sim = len(intersection)/len(union)
        return jacc_sim, precision, recall, f1_score, onlyin_original, onlyin_generated
    else:
        print('Please make sure the two inputs to be compared are both either lists or strings.')
        return None, None, None, None, None, None
    

# Function to calculate BioBERT-Cosine similarity
def biobert_similarity(text1, text2, model, tokenizer):
    '''Function to compute the BioBERT-Cosine Similarity between two texts. The function first converts
    the text to embeddings using the BioBERT model, and then calculates the cosine similarity of the vector
    embeddings.

    Parameters:
        text1 (string): The sentence
        text2 (string): The sentence
        model (object): The transformers object containing the model
        tokenizer (object): The transformers object containing the tokenizer
    '''

    # Tokenize the sentence
    # start_3 = time.perf_counter()
    tokens1 = tokenizer(text1, return_tensors="pt")
    tokens2 = tokenizer(text2, return_tensors="pt")
    # finish_3 = time.perf_counter()
    # print(f"Finished tokenizing the sentences in {round((finish_3-start_3), 2)} second(s)")

    # Forward pass through the model to convert the token representations of the texts to vector embeddings using the pretrained model
    # start_3 = time.perf_counter()
    with torch.no_grad():
        outputs1 = model(**tokens1)
        outputs2 = model(**tokens2)
    # finish_3 = time.perf_counter()
    # print(f"Finished forward pass in {round((finish_3-start_3), 2)} second(s)")

    # Extract the sentence embeddings (last hidden states of all tokens)
    # start_3 = time.perf_counter()
    sentence_embedding1 = outputs1.last_hidden_state.mean(dim=1)  # Mean pooling
    sentence_embedding2 = outputs2.last_hidden_state.mean(dim=1)
    # finish_3 = time.perf_counter()
    # print(f"Finished extracting the sentence embedding from the model in {round((finish_3-start_3), 2)} second(s)")

    # Computes the cosine similarity of the two sentence embeddings
    biobert_sim = cosine_similarity(sentence_embedding1, sentence_embedding2)[0][0]
    return biobert_sim


# Function to calculate Ensemble similarity
def ensemble_similarity(text1, text2, nlp, model, tokenizer):
    '''This ensemble combines two evaluation methods (Jaccard Similarity and BioBERT-Cosine Similarity) to score similarity
    between two texts.
    
    -Jaccard: Important because accuracy of details is important given the sensitivity of detail being represented, even though jaccard can be strict regardles of accuracy of details and context
    -BioBERT: Important because even in instance where details are correct and context is aptly represented, slight variations in words generated or the way they are phrased can lead to negative score by Jaccard.
    -More testing and comaprison to human evaluation needed to aptly determine the best weight distribution.
    
    Parameters:
        text1 (string): The sentence
        text2 (string): The sentence
        nlp (object): The spacy dictionary for lemmatizing
        model (object): The transformers object containing the model
        tokenizer (object): The transformers object containing the tokenizer
    '''
    
    # Assigns the dicitonary, model and tokenizer to be used by the Jaccars and BioBERT-Cosine similarity to their respective variables
    nlp = nlp
    model = model 
    tokenizer = tokenizer
    
    #Clean sentences using regex
    # start_2 = time.perf_counter()
    cleantext1 = clean_sentence(text1)
    cleantext2 = clean_sentence(text2)
    # finish_2 = time.perf_counter()
    # print(f"Finished cleaning texts in {round((finish_2-start_2), 2)} second(s)")
    
    #Lematize the words in the sentences for more uniformity
    # start_2 = time.perf_counter()
    lemmatext1 = lemmatize(cleantext1, nlp)
    lemmatext2 = lemmatize(cleantext2, nlp)
    # finish_2 = time.perf_counter()
    # print(f"Finished lemmatizing texts in {round((finish_2-start_2), 2)} second(s)")
    
    #Similarity scores
    # start_2 = time.perf_counter()
    jacc_sim, precision, recall, f1_score, onlyin_original, onlyin_generated = jaccard_similarity(lemmatext1, lemmatext2) #Jaccard Similarity is calculated on the transformed sentences
    # finish_2 = time.perf_counter()
    # print(f"Finished calculating Jaccard similarity in {round((finish_2-start_2), 2)} second(s)")
    
    # start_2 = time.perf_counter()
    biobert_sim = biobert_similarity(text1, text2, model, tokenizer) #BioBERT-Cosine Similarity is calculated on the untransformed sentence
    # finish_2 = time.perf_counter()
    # print(f"Finished calculating BioBERT-Cosine similarity in {round((finish_2-start_2), 2)} second(s)")

    #Ensemble (Weighted Average) of the Jaccard and BioBERT-Cosine Similarities is calculated with more weight on teh Jaccard similarity
    #More weight is given to Jaccard similarity because even though exactness of words can be a limited measure, it is still more important than just representing the general concept
    ensemble_sim = 0.65*jacc_sim + 0.35*biobert_sim

    return jacc_sim, biobert_sim, ensemble_sim, precision, recall, f1_score, onlyin_original, onlyin_generated