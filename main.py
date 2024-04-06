from transformers import AutoTokenizer, AutoModel
import spacy, re, json
import utilities, variables


def dictionary_scoring(dictionary_1, dictionary_2, nlp, model, tokenizer):
    '''This funciton computes the Jaccard and BioBERT-Cosine similarities and an Ensemble of these two methods. The Ensemble is a weighted average
    of the two methods. It also calculates other metrics like the precision, recall and f1-score.'''
    
    key_set = set(list(dictionary_1.keys())).union(set(list(dictionary_2.keys())))
    score_dict_total = {}
    for key in key_set:
        if (key in list(dictionary_1.keys())) & (key in list(dictionary_2.keys())):
            text1 = str(dictionary_1[key])
            text2 = str(dictionary_2[key])
            
            jacc_sim, biobert_sim, ensemble_sim, precision, recall, f1_score = utilities.ensemble_similarity(text1, text2, nlp, model, tokenizer)
            score_list = [str(jacc_sim), str(biobert_sim), str(ensemble_sim), str(precision), str(recall), str(f1_score)]
            key_list = ['jacc_sim', 'biobert_sim', 'ensemble_sim', 'precision', 'recall', 'f1_score']
            score_dict = dict(zip(key_list, score_list))
            
            score_dict_total[key] = score_dict
        elif (key in list(dictionary_1.keys())) & ~(key in list(dictionary_2.keys())):
            score_dict_total[key]['Status'] = 'FN'
        elif ~(key in list(dictionary_1.keys())) & (key in list(dictionary_2.keys())):
            score_dict_total[key]['Status'] = 'FP'
    return score_dict_total



if __name__ == "__main__":
    score_dictionary = dictionary_scoring(variables.data_1, variables.data_2, utilities.nlp, utilities.model, utilities.tokenizer)
    score_dictionary = {key:score_dictionary[key] for key in list(variables.data_1.keys())}
    
    # Save JSON string to a file
    scores_json = json.dumps(score_dictionary, indent=4)
    with open(f"dictionary_scores.json", "w") as json_file:
        json_file.write(scores_json)

    # arranged_keys = list(variables.data_1.keys())
    # for key in arranged_keys:
    #     print(f"{key}: {score_dictionary[key]}")