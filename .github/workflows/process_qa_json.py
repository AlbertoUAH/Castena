# -- Python script to process Questions & Answer datasets and store it as JSON file
import pandas as pd
import itertools
import json
import re
import os

BASE_DIR = '../../'

# -- Auxiliar function to elaborate a cleaned dictionary with all questions and answers
def elaborate_qa_dict(text):
    pattern = r'(.*?)\? (.*?)((?:\?|$))'
    results = re.findall(pattern, text)
    question_answer_dictionary = {}
    for question, answer, _ in results:
        question_answer_dictionary[question] = answer
    return question_answer_dictionary

list_original_qa_json_files = os.listdir(BASE_DIR + 'data/eval/')
list_original_cleaned_qa_json_files = os.listdir(BASE_DIR + 'data/eval/eval_cleaned/')

for original_qa_json_file in list_original_qa_json_files:
  if original_qa_json_file not in list_original_cleaned_qa_json_files:
    df = pd.read_json(BASE_DIR + 'data/eval/' + original_qa_json_file)
    # Extract Q&A's (if exist)
    df['qa'] = df['annotations'].apply(lambda x: list(itertools.chain.from_iterable([[annotation_aux['value']['text'] \
                                                                                      for annotation_aux in annotation['result']] \
                                                                                     for annotation in x])
                                                     )[0][0]
                                      )
    df['qa_dict'] = df['qa'].apply(lambda x: elaborate_qa_dict(x))
    final_json_file = list(df['qa_dict'])
    with open(BASE_DIR + 'data/eval/eval_cleaned/' + original_qa_json_file, 'w') as f:
      json.dump(data, final_json_file)

print("Processing finished successfully!")