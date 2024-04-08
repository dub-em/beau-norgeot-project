import pytest, main, utilities, variables, warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_dictionary_scoring_1():
    #This checks if the categories in the dictionaries are present in the result of the scores
    dictionary_1 = {"Title": "Referral Order",
                    "Date": "05/10/2024",
                    "To Provider": "Cardiology Partners",
                    "Phone": "(555) 789-1234",}
    dictionary_2 = {"Title": "Referral Order",
                    "Date": "04/02/2024",
                    "To Provider": "Heartcare Cardiology Group",
                    "Phone": "(555) 555-1234",}
    list_of_columns = ['Title','Date','To Provider','Phone','Total Average Score']
    score_dictionary = main.dictionary_scoring(dictionary_1, dictionary_2, utilities.nlp, utilities.model, utilities.tokenizer)
    check_1 = [key in list_of_columns for key in list(score_dictionary.keys())]
    check_2 = [key in list(score_dictionary.keys()) for key in list_of_columns]
    assert str(type(score_dictionary)) == "<class 'dict'>"
    assert False not in check_1
    assert False not in check_2

    #This checks if the score metrics listed are all present in the score results for each category in the dictionaries
    key_list = [['Jacc_Sim', 'BioBERT_Sim', 'Ensemble_Sim', 'Precision', 'Recall', 'F1_Score', 'Avg_TextLength'], 
                ['Jaccard Average', 'BioBERT Average', 'Ensemble Average', 'Weighted Ensemble Average']]
    check_1 = [str(type(score_dictionary[key])) == "<class 'dict'>" for key in list(score_dictionary.keys())]
    check_2 = [list(score_dictionary[key].keys()) in key_list for key in list(score_dictionary.keys())]
    assert False not in check_1
    assert False not in check_2

    #This checks if the value type is as expected for all metrics calculated in the score results for each category in the dictionaries
    check_1 = [[str(type(score_dictionary[key_1][key_2])) in ["<class 'float'>", "<class 'str'>"]
                for key_2 in list(score_dictionary[key_1].keys())] for key_1 in list(score_dictionary.keys())]
    check_2 = [False not in check for check in check_1]
    assert False not in check_2


def test_dictionary_scoring_2():
    dictionary_1 = {"Title": "Referral Order",
                    "Date": "05/10/2024",
                    "To Provider": "Cardiology Partners",
                    "Phone": "(555) 789-1234",
                    "Fax": "(555) 789-5678",}
    dictionary_2 = {"Title": "Referral Order",
                    "Date": "04/02/2024",
                    "To Provider": "Heartcare Cardiology Group",
                    "Phone": "(555) 555-1234",
                    "Name": "Dr. Emily Chen",}
    list_of_columns = ['Title','Date','To Provider','Phone','Fax','Name','Total Average Score']
    score_dictionary = main.dictionary_scoring(dictionary_1, dictionary_2, utilities.nlp, utilities.model, utilities.tokenizer)
    check_1 = [key in list_of_columns for key in list(score_dictionary.keys())]
    check_2 = [key in list(score_dictionary.keys()) for key in list_of_columns]
    assert str(type(score_dictionary)) == "<class 'dict'>"
    assert False not in check_1
    assert False not in check_2

    #This checks if the score metrics listed are all present in the score results for each category in the dictionaries
    key_list = [['Jacc_Sim', 'BioBERT_Sim', 'Ensemble_Sim', 'Precision', 'Recall', 'F1_Score', 'Avg_TextLength'], 
                ['Jaccard Average', 'BioBERT Average', 'Ensemble Average', 'Weighted Ensemble Average']]
    check_1 = [str(type(score_dictionary[key])) == "<class 'dict'>" for key in list(score_dictionary.keys())]
    check_2 = [list(score_dictionary[key].keys()) in key_list for key in list(score_dictionary.keys())]
    assert False not in check_1
    assert False not in check_2

    #This checks if the value type is as expected for all metrics calculated in the score results for each category in the dictionaries
    check_1 = [[str(type(score_dictionary[key_1][key_2])) in ["<class 'float'>", "<class 'str'>"]
                for key_2 in list(score_dictionary[key_1].keys())] for key_1 in list(score_dictionary.keys())]
    check_2 = [False not in check for check in check_1]
    assert False not in check_2

    

    
