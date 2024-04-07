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
    
    key_set = set(list(dictionary_1.keys())).union(set(list(dictionary_2.keys())))
    score_dict_total = {}
    for key in key_set:
        if (key in list(dictionary_1.keys())) & (key in list(dictionary_2.keys())):
            text1 = str(dictionary_1[key])
            text2 = str(dictionary_2[key])
            avg_textlength = sum([len(text1.split()), len(text2.split())])/2
            
            jacc_sim, biobert_sim, ensemble_sim, precision, recall, f1_score = utilities.ensemble_similarity(text1, text2, nlp, model, tokenizer)
            if f1_score == 'Nan':
                score_list = [float(jacc_sim), float(biobert_sim), float(ensemble_sim), float(precision), float(recall), f1_score, float(avg_textlength)]
            else:
                score_list = [float(jacc_sim), float(biobert_sim), float(ensemble_sim), float(precision), float(recall), float(f1_score), float(avg_textlength)]
            key_list = ['Jacc_Sim', 'BioBERT_Sim', 'Ensemble_Sim', 'Precision', 'Recall', 'F1_Score', 'Avg_TextLength']
            score_dict = dict(zip(key_list, score_list))
            
            score_dict_total[key] = score_dict
        elif (key in list(dictionary_1.keys())) & ~(key in list(dictionary_2.keys())):
            score_dict_total[key]['Status'] = 'FN'
        elif ~(key in list(dictionary_1.keys())) & (key in list(dictionary_2.keys())):
            score_dict_total[key]['Status'] = 'FP'
    
    avg_lengths = [float(score_dict_total[key]['Avg_TextLength']) for key in score_dict_total.keys()]
    total_textlengths = sum(avg_lengths)
    perc_lengths = [length/total_textlengths for length in avg_lengths]

    jacc_average = [float(score_dict_total[key]['Jacc_Sim']) for key in score_dict_total.keys()]
    biobert_average = [float(score_dict_total[key]['BioBERT_Sim']) for key in score_dict_total.keys()]
    ensem_average = [float(score_dict_total[key]['Ensemble_Sim']) for key in score_dict_total.keys()]
    weighted_average = [ensem_average[i]*perc_lengths[i] for i in range(len(ensem_average))]

    total_list = [sum(jacc_average)/len(jacc_average), sum(biobert_average)/len(biobert_average), sum(ensem_average)/len(ensem_average), sum(weighted_average)]
    total_listtitle = ['Jaccard Average', 'BioBERT Average', 'Ensemble Average', 'Weighted Ensemble Average']

    total_dict = dict(zip(total_listtitle, total_list))     
    score_dict_total['Total Average Score'] = total_dict
    
    return score_dict_total



if __name__ == "__main__":
    score_dictionary = dictionary_scoring(variables.data_1, variables.data_2, utilities.nlp, utilities.model, utilities.tokenizer)
    list_of_column = list(variables.data_1.keys())
    list_of_column.append('Total Average Score')
    score_dictionary = {key:score_dictionary[key] for key in list_of_column}
    
    # Save JSON string to a file
    scores_json = json.dumps(score_dictionary, indent=4)
    with open(f"dictionary_scores.json", "w") as json_file:
        json_file.write(scores_json)