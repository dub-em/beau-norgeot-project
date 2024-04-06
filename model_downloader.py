from transformers import AutoTokenizer, AutoModel


def download_model(tokenizer_hub_path:str, model_hub_path:str, tokenizer_local_path:str, model_local_path:str):
    '''This funciton takes in paths to a specific model and its token in the hugging face hub,
    downloads the model and tokenizer and saves them in a specifiied local path.'''

    #Downloads the tokenizer and model from the huggingface hub
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_hub_path)
    model = AutoModel.from_pretrained(model_hub_path)

    #Saves the dowloaded tokenizer and model to a local directory
    tokenizer.save_pretrained(tokenizer_local_path)
    model.save_pretrained(model_local_path)

    print(f"Successfully downloaded and stored tokenizer and model")



if __name__ == "__main__":

    tokenizer_hub_path = "monologg/biobert_v1.1_pubmed"
    model_hub_path = "monologg/biobert_v1.1_pubmed"
    
    tokenizer_local_path = "./biobert_model/"
    model_local_path = "./biobert_model/"

    download_model(tokenizer_hub_path, model_hub_path, tokenizer_local_path, model_local_path)