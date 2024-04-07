import json, utilities, variables


def dictionary_scoring(dictionary_1, dictionary_2, nlp, model, tokenizer):
    '''This funciton computes the Jaccard and BioBERT-Cosine similarities and an Ensemble of these two methods for the data in two dictionaries. 
    The Ensemble is a weighted average of the two methods. It also calculates other metrics like the precision, recall and f1-score for each key 
    in the dictionary. Finally it also computes the overall score of these two dictionaries using a weighted average of the scores of each key in
    the dictionaries.
    
    Parameters:
        dictionary_1 (dictionary): The original dictionary containing the data of the patient
        dictionary_2 (dictionary): The generated dictionary containing the name-entities extracted from the LLM-generate patient report.
        nlp (object): The spacy dictionary for lemmatizing
        model (object): The transformers object containing the model
        tokenizer (object): The transformers object containing the tokenizer
    '''
    
    #Extract all the keys in the two dictionaries to be computed and get the unique list of keys
    key_set = set(list(dictionary_1.keys())).union(set(list(dictionary_2.keys())))
    score_dict_total = {} #Dictionary to store the scores of each key in both dictionaries to be compared.

    #Loops through each key to compare its value in the original dictionary to th eone in the generated dictionary
    for key in key_set:
        if (key in list(dictionary_1.keys())) & (key in list(dictionary_2.keys())):
            #Extracts the value of a given key from both dictionaries
            text1 = str(dictionary_1[key])
            text2 = str(dictionary_2[key])
            avg_textlength = sum([len(text1.split()), len(text2.split())])/2 #Computes the average length of both values to be used for the weighted average
            
            #Computes the Jaccard, BioBERT-Cosine and Ensemble Similarity of both texts
            jacc_sim, biobert_sim, ensemble_sim, precision, recall, f1_score = utilities.ensemble_similarity(text1, text2, nlp, model, tokenizer)
            if f1_score == 'Nan':
                score_list = [float(jacc_sim), float(biobert_sim), float(ensemble_sim), float(precision), float(recall), f1_score, float(avg_textlength)]
            else:
                score_list = [float(jacc_sim), float(biobert_sim), float(ensemble_sim), float(precision), float(recall), float(f1_score), float(avg_textlength)]
            key_list = ['Jacc_Sim', 'BioBERT_Sim', 'Ensemble_Sim', 'Precision', 'Recall', 'F1_Score', 'Avg_TextLength']
            score_dict = dict(zip(key_list, score_list)) #Creates a dictionary of scores using the computed values
            
            #Assigns the dictionary of scores to the key is was calculated for in the overall dictionary
            score_dict_total[key] = score_dict
        elif (key in list(dictionary_1.keys())) & ~(key in list(dictionary_2.keys())):
            #If a certain key is present in the original dictionary but not in the generated dictionary then this value is assinged for later calculation
            score_dict_total[key]['Status'] = 'FN'
        elif ~(key in list(dictionary_1.keys())) & (key in list(dictionary_2.keys())):
            #If a certain key is present in the generated dictionary but not in the original dictionary then this value is assinged for later calculation
            score_dict_total[key]['Status'] = 'FP'
    
    #Weights of each key in the overall score dicitonary is calculated using the ratio of its average text length to the total sum of average text lengths
    avg_lengths = [float(score_dict_total[key]['Avg_TextLength']) for key in score_dict_total.keys()]
    total_textlengths = sum(avg_lengths)
    perc_lengths = [length/total_textlengths for length in avg_lengths]

    #Total average scores and the weighted average scores of keys in the dictionaries are calculated
    jacc_average = [float(score_dict_total[key]['Jacc_Sim']) for key in score_dict_total.keys()]
    biobert_average = [float(score_dict_total[key]['BioBERT_Sim']) for key in score_dict_total.keys()]
    ensem_average = [float(score_dict_total[key]['Ensemble_Sim']) for key in score_dict_total.keys()]
    weighted_average = [ensem_average[i]*perc_lengths[i] for i in range(len(ensem_average))] #Weighted Ensemble average

    total_list = [sum(jacc_average)/len(jacc_average), sum(biobert_average)/len(biobert_average), sum(ensem_average)/len(ensem_average), sum(weighted_average)]
    total_listtitle = ['Jaccard Average', 'BioBERT Average', 'Ensemble Average', 'Weighted Ensemble Average']

    #These scores for the entire dictionary are added to the overall dictionary
    total_dict = dict(zip(total_listtitle, total_list))     
    score_dict_total['Total Average Score'] = total_dict
    
    return score_dict_total



if __name__ == "__main__":
    #Calculates the similarity scores and metrics of the LLM generated dictionary to the original dictionary
    score_dictionary = dictionary_scoring(variables.data_1, variables.data_2, utilities.nlp, utilities.model, utilities.tokenizer)
    list_of_column = list(variables.data_1.keys())
    list_of_column.append('Total Average Score')
    score_dictionary = {key:score_dictionary[key] for key in list_of_column}
    
    # Converts the resulting dictionary to a json string and saves the json string to a file
    scores_json = json.dumps(score_dictionary, indent=4)
    with open(f"dictionary_scores.json", "w") as json_file:
        json_file.write(scores_json)