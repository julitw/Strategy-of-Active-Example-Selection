from Amodels.GPT4oMiniLLM import GPT4oMiniLLM
from Amodels.CohereLLM import CohereLLM
from Amodels.Llama3_1LLM import Llama3_1LLM
import sys
import os

def get_model_by_tag(tag, token, temperature = 0):

    model_mapping = {
        'gpt4omini': GPT4oMiniLLM,
        'llama3': Llama3_1LLM,
        'cohere': CohereLLM
    }

    if tag not in model_mapping:
        raise ValueError(f"Nieznany tag: {tag}. Dozwolone wartości to: {list(model_mapping.keys())}")

    model_class = model_mapping[tag]

    return model_class(token=token, temperature = temperature)


import pandas as pd

def get_dataset(dataset_tag):
    if(dataset_tag == 'ag_news'):
        return pd.read_csv('../Adata/ag_news/train_subset.csv')
    if(dataset_tag == '20_newsgroups'):
        return pd.read_csv('../Adata/20_newsgroups/train_subset.csv')
    if(dataset_tag == 'yahoo'):
        return pd.read_csv('../Adata/yahoo/train_subset.csv')
    if(dataset_tag == 'social_media'):
        return pd.read_csv('../Adata/social_media/train_subset.csv')
    if(dataset_tag == 'sst5'):
        return pd.read_csv('../Adata/sst5/train_subset.csv')
    if(dataset_tag == 'go_emotions'):
        return pd.read_csv('../Adata/go_emotions/train_subset.csv')
    if(dataset_tag == '20_newsgroups'):
        return pd.read_csv('../Adata/20_newsgroups/train_subset.csv')
    
    
def get_prompt_template(dataset_tag, prompt_name):
    path = f'prompts/{dataset_tag}/{prompt_name}.txt'
    with open(path, "r") as file:
        content = file.read()
        return  content


# 

import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def evaluate_predictions_from_folders(folder_paths, filter, true_col='y_true', pred_col='y_pred', filter_col='was_in_selected_samples'):
    """
    Wczytuje wszystkie pliki CSV z podanych folderów, porównuje kolumny true_col i pred_col,
    i oblicza metryki klasyfikacyjne tylko dla tych wierszy, które mają wartość False w kolumnie filter_col.

    Parameters:
        folder_paths (list): Lista ścieżek do folderów.
        true_col (str): Nazwa kolumny z rzeczywistymi etykietami.
        pred_col (str): Nazwa kolumny z przewidywaniami.
        filter_col (str): Nazwa kolumny, która zawiera wartość True/False (do filtrowania wierszy).

    Returns:
        pd.DataFrame: DataFrame z wynikami dla każdego pliku.
    """
    results = []

    for folder in folder_paths:
        for file_name in os.listdir(folder):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder, file_name)
                try:
                    df = pd.read_csv(file_path)

                    # Filtrowanie tylko tych wierszy, które mają False w kolumnie filter_col
                    if filter:
                        df_filtered = df[df[filter_col] == False]
                    else:
                        df_filtered = df

                    if df_filtered.empty:
                        continue
                    
                    y_true = df_filtered[true_col].astype(str).str.lower()
                    y_pred = df_filtered[pred_col].astype(str).str.lower()

                    y_true = df_filtered[true_col].apply(
                        lambda x: str(int(x)) if pd.api.types.is_numeric_dtype(type(x)) and not pd.isna(x) else str(x)
                    )
                    y_pred = df_filtered[pred_col].apply(
                        lambda x: str(int(x)) if pd.api.types.is_numeric_dtype(type(x)) and not pd.isna(x) else str(x)
                    )

                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average='macro')
                    recall = recall_score(y_true, y_pred, average='macro')
                    precision = precision_score(y_true, y_pred, average='macro')

                    results.append({
                        'file': file_name.lower(),
                        'accuracy': acc,
                        'f1_score': f1,
                        'recall': recall,
                        'precision': precision
                    })
                except Exception as e:
                    print(f"błąd przetwarzania pliku {file_path}: {e}")

    return pd.DataFrame(results)






import os
import pandas as pd

def count_missing_in_column_from_folders(folder_paths, target_col='output'):
    """
    Wczytuje wszystkie pliki CSV z podanych folderów i zlicza brakujące
    lub puste wartości w zadanej kolumnie.

    Parameters:
        folder_paths (list): Lista ścieżek do folderów z plikami CSV.
        target_col (str): Nazwa kolumny, w której szukamy braków.

    Returns:
        pd.DataFrame: DataFrame z informacją o brakujących wartościach w kolumnie.
    """
    results = []

    for folder in folder_paths:
        for file_name in os.listdir(folder):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder, file_name)
                try:
                    df = pd.read_csv(file_path)

                    if target_col not in df.columns:
                        missing_count = 'kolumna nie istnieje'
                    else:
                        missing_count = df[target_col].isna().sum() + (df[target_col] == '').sum()

                    results.append({
                        'file': file_path,
                        'missing_count': missing_count
                    })
                except Exception as e:
                    print(f"Błąd przetwarzania pliku {file_path}: {e}")

    return pd.DataFrame(results)
