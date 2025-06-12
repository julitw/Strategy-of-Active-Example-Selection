import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langsmith import traceable
import re
import concurrent.futures
import multiprocessing
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def call_chain_process(chain, text, return_dict):
    try:
        result = chain.invoke({"text": text})
        content = result[0] if isinstance(result, tuple) else result

        # Próba pobrania logprobs jeśli są dostępne
        logprobs = getattr(chain.llm, 'logprobs', None)
        top_logprobs = None
        if logprobs and isinstance(logprobs, dict):
            top_logprobs = logprobs.get('content', [{}])[0].get('top_logprobs', None)

        return_dict["result"] = content
        return_dict["logprobs"] = logprobs
        return_dict["top_logprobs"] = top_logprobs

    except Exception as e:
        return_dict["error"] = str(e)


class LLMAnnotator:
    def __init__(self, model, dataset, examples_for_prompt, prompt_template, column_text = 'text', column_label='label', column_output='predicted_label'
                 ):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # dodane
        self.model = model
        self.dataset_for_annotation = dataset
        self.examples_for_prompt = examples_for_prompt
        self.prompt_template = prompt_template
        self.column_text = column_text
        self.column_label = column_label
        self.output_column = column_output
        self.prompt = None
        self.chain = None
        self.results = None

    def _build_prompt(self):
        """
        Tworzy prompt dla modelu LLM na podstawie podanego przykładu.
        """
        try:
            if not self.examples_for_prompt.empty:
                examples = []
                for idx, row in self.examples_for_prompt.iterrows():
                    label_name = row['label']
                    example_str = (
                        f"Example {idx + 1}:\n"
                        f"    Text: \"{row['text']}\"\n"
                        f"    Annotation:  {label_name}\n"
                    )
                    examples.append(example_str)
                result_string_examples = "\n\n".join(examples)
                prompt_template_with_examples = self.prompt_template.replace("{examples}", result_string_examples)
          
            else:
                prompt_template_with_examples = self.prompt_template
            
          

            self.prompt = PromptTemplate(
                input_variables=["text"],
                template=prompt_template_with_examples
            )
        except Exception as e:
            raise ValueError(f"Error while building prompt: {e}")

    def _build_chain(self):
        if not self.prompt:
            raise ValueError("Prompt has not been built. Call _build_prompt() first.")
        self.chain = LLMChain(llm=self.model, prompt=self.prompt)



    @traceable(name="traceable_annotation")
    def fetch_answer(self, texts, original_labels, max_retries=30, timeout_seconds=60, is_save = False, output_path=None):
        annotations = []

        for i, text in enumerate(texts):
            attempt = 0
            while attempt < max_retries:
                manager = multiprocessing.Manager()
                return_dict = manager.dict()

                process = multiprocessing.Process(target=call_chain_process, args=(self.chain, text, return_dict))
                process.start()
                process.join(timeout_seconds)

                if process.is_alive():
                    print(f"Nr: {i}, próba {attempt + 1}/{max_retries} – przekroczono timeout {timeout_seconds}s, przerywam")
                    process.terminate()
                    process.join()
                    attempt += 1
                elif "error" in return_dict:
                    print(f"Nr: {i}, próba {attempt + 1}/{max_retries} – błąd: {return_dict['error']}")
                    attempt += 1
                else:
                    content = return_dict.get("result", None)
                    logprobs = return_dict.get("logprobs", None)
                    top_logprobs = return_dict.get("top_logprobs", None)

                    print('Nr:', i, "Predicted label:", content['text'])

                    annotations.append({
                        "text": text,
                        self.output_column: content['text'],
                        "logprobs": logprobs,
                        "top_logprobs": top_logprobs,
                        "original_label": original_labels[i]
                    })

                    break

            if attempt == max_retries:
                annotations.append({"text": text, "error": f"Max retries reached: {max_retries}"})
                
                        # <- ZMIANA: co 50 zapis
            if is_save and output_path and (i + 1) % 50 == 0:
                print(f"Zapis tymczasowy po {i+1} przykładach do {output_path}")
                pd.DataFrame(annotations).to_csv(output_path, index=False, encoding='utf-8')


        return annotations




    def get_results(self, is_save=False, output_path=None):
        try:
            data = self.dataset_for_annotation
            if self.column_label not in data.columns or self.column_text not in data.columns:
                raise ValueError("Dataset must contain 'text' and 'label' columns.")
            
            texts = data[self.column_text].tolist()
            labels = data[self.column_label].tolist()
            
            self._build_prompt()
            self._build_chain()
            
            self.results = self.fetch_answer(texts, labels,max_retries=30, timeout_seconds=60, is_save= is_save, output_path = output_path)
            return self.results  
        except Exception as e:
            raise ValueError(f"Error in get_results: {e}")

    def save_results(self, output_path, if_extract=False):
        if self.results is None:
            raise ValueError("No results available. Run get_results() first.")
        try:
            results_df = pd.DataFrame(self.results)
            if(if_extract):
                results_df['original_text'] = self.dataset_for_annotation['text']
            results_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Results saved successfully to {output_path}")
        except Exception as e:
            raise ValueError(f"Error while saving results: {e}")



