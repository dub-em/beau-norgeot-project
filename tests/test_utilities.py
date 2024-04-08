import pytest, utilities, variables, warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_clean_sentence():
    text = "Please evaluate for potential cardiac causes of syncope, and provide recommendations for further testing and management."
    expected_text = "Please evaluate for potential cardiac causes of syncope and provide recommendations for further testing and management"
    cleaned_text = utilities.clean_sentence(text)
    comp_list = ['True' for symbol in [',','.',"'",'"',':',';','!','?'] if symbol in cleaned_text]
    assert str(type(cleaned_text)) == "<class 'str'>"
    assert len(comp_list) == 0
    assert cleaned_text == expected_text


def test_lemmatize():
    text = "Please evaluate for potential cardiac causes of syncope and provide recommendations for further testing and management"
    expected_output = ['please', 'evaluate', 'for', 'potential',
                       'cardiac', 'cause', 'of', 'syncope', 'and',
                       'provide', 'recommendation', 'for', 'further',
                       'testing', 'and', 'management']
    lemmatized_text = utilities.lemmatize(text, utilities.nlp)
    assert str(type(lemmatized_text)) == "<class 'list'>"
    assert lemmatized_text == expected_output


def test_jaccard_similarity():
    text_1 = 'please evaluate for potential cardiac cause of syncope and provide recommendations for further testing and management'
    text_2 = 'please evaluate for potential arrhythmia and provide recommendations for further testing and management'
    text_list1 = ['please', 'evaluate', 'for', 'potential', 'cardiac', 'cause', 'of', 'syncope', 'and', 'provide', 'recommendation', 'for', 'further', 'testing', 'and', 'management']
    text_list2 = ['please', 'evaluate', 'for', 'potential', 'arrhythmia', 'and', 'provide', 'recommendation', 'for', 'further', 'testing', 'and', 'management']
    expected_list_1 = [0.6666666666666666, 0.9090909090909091, 0.7142857142857143, 0.8]
    expected_list_2 = [['cardiac', 'of', 'syncope', 'cause'], ['arrhythmia']]
    jacc_sim, precision, recall, f1_score, onlyin_original, onlyin_generated = utilities.jaccard_similarity(text_1, text_2)
    result_list_1 = [jacc_sim, precision, recall, f1_score]
    result_list_2 = [onlyin_original, onlyin_generated]
    check_1 = [result_list_2[0][i] in expected_list_2[0] for i in range(len(result_list_2[0]))] + [expected_list_2[0][i] in result_list_2[0] for i in range(len(expected_list_2[0]))]
    check_2 = [result_list_2[1][i] in expected_list_2[1] for i in range(len(result_list_2[1]))] + [expected_list_2[1][i] in result_list_2[1] for i in range(len(expected_list_2[1]))]
    assert result_list_1 == expected_list_1
    assert False not in check_1
    assert False not in check_2
    jacc_sim, precision, recall, f1_score, onlyin_original, onlyin_generated = utilities.jaccard_similarity(text_list1, text_list2)
    result_list_1 = [jacc_sim, precision, recall, f1_score]
    result_list_2 = [onlyin_original, onlyin_generated]
    check_1 = [result_list_2[0][i] in expected_list_2[0] for i in range(len(result_list_2[0]))] + [expected_list_2[0][i] in result_list_2[0] for i in range(len(expected_list_2[0]))]
    check_2 = [result_list_2[1][i] in expected_list_2[1] for i in range(len(result_list_2[1]))] + [expected_list_2[1][i] in result_list_2[1] for i in range(len(expected_list_2[1]))]
    assert result_list_1 == expected_list_1
    assert False not in check_1
    assert False not in check_2

    text_1 = 'R55: Syncope and collapse'
    text_2 = 'R00.2: Palpitations'
    expected_list = [0.0, 0.0, 0.0, 'Nan']
    jacc_sim, precision, recall, f1_score, onlyin_original, onlyin_generated = utilities.jaccard_similarity(text_1, text_2)
    result_list = [jacc_sim, precision, recall, f1_score]
    assert result_list == expected_list

    jacc_sim, precision, recall, f1_score, onlyin_original, onlyin_generated = utilities.jaccard_similarity(text_1, text_list2)
    result_list = [jacc_sim, precision, recall, f1_score, onlyin_original, onlyin_generated]
    assert result_list == [None, None, None, None, None, None]


def test_biobert_similarity():
    text1 = "Please evaluate for potential cardiac causes of syncope and provide recommendations for further testing and management."
    text2 = "Please evaluate for potential arrhythmia and provide recommendations for further testing and management."

    biobert_sim = utilities.biobert_similarity(text1, text2, utilities.model, utilities.tokenizer)
    assert round(float(biobert_sim), 5) == round(float(0.9759095907211304), 5)


def test_ensemble_similarity():
    text1 = "Please evaluate for potential cardiac causes of syncope and provide recommendations for further testing and management."
    text2 = "Please evaluate for potential arrhythmia and provide recommendations for further testing and management."
    expected_list_1 = [round(float(0.6666666666666666), 5), round(float(0.9759095907211304), 5), round(float(0.774901690085729), 5), round(float(0.9090909090909091), 5), round(float(0.7142857142857143), 5), 0.8]
    expected_list_2 = [['cardiac', 'of', 'syncope', 'cause'], ['arrhythmia']]

    jacc_sim, biobert_sim, ensemble_sim, precision, recall, f1_score, onlyin_original, onlyin_generated = utilities.ensemble_similarity(text1, text2, utilities.nlp, utilities.model, utilities.tokenizer)
    result_list_1 = [round(float(jacc_sim), 5), round(float(biobert_sim), 5), round(float(ensemble_sim), 5), round(float(precision), 5), round(float(recall), 5), f1_score]
    result_list_2 = [onlyin_original, onlyin_generated]
    check_1 = [result_list_2[0][i] in expected_list_2[0] for i in range(len(result_list_2[0]))] + [expected_list_2[0][i] in result_list_2[0] for i in range(len(expected_list_2[0]))]
    check_2 = [result_list_2[1][i] in expected_list_2[1] for i in range(len(result_list_2[1]))] + [expected_list_2[1][i] in result_list_2[1] for i in range(len(expected_list_2[1]))]
    assert result_list_1 == expected_list_1
    assert False not in check_1
    assert False not in check_2
