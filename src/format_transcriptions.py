# -- Format transcriptions + translation into English (if necessary)
from   googletrans import Translator
import pandas as pd
import itertools
import spacy
import json
import re
import os

BASE_DIR = '/home/runner/work/Castena/Castena'

list_original_transcriptions   = [file for file in os.listdir(BASE_DIR + '/data/original_spanish_transcriptions/') if '.txt' in file]
list_translated_transcriptions = [file for file in os.listdir(BASE_DIR + '/data/translated_transcriptions/') if '.txt' in file]

translator = Translator(service_urls=['translate.googleapis.com'])
nlp = spacy.load('es_core_news_sm')

# -- Define auxiliar functions
def translate_text(text, target_lang='en'):
    # Tokenize text and find proper nouns (omit translation in these cases)
    doc = nlp(text)
    named_entities = [ent.text for ent in doc if ent.pos_ == 'PROPN' and ent.dep_ in ['NNP', 'NN']]
    named_entities_list = []
    # Replace proper nouns for temporal __identifier__
    for entity in named_entities:
        text = text.replace(entity, f'__{entity}__')
        named_entities_list.append(entity)

    translated_text = translator.translate(text, dest=target_lang).text
    final_translated_text = []

    i = 0
    for text in translated_text.split(' '):
      if '__' in text and len(named_entities_list):
        final_translated_text.append(named_entities_list[i])
        i+=1
      else:
        final_translated_text.append(text)
    return ' '.join(final_translated_text)

def capitalize_proper_nouns(text):
    doc = nlp(text)
    modified_text = []

    for token in doc:
        if token.pos_ != "PROPN":
            modified_text.append(token.text)
        else:
            modified_text.append(token.text.capitalize())

    modified_text = " ".join(modified_text)
    # Correct punctuation signs
    modified_text = modified_text.replace(" .. ", "...")\
                                 .replace(" , ", ",")\
                                 .replace(" .... ", "...")\
                                 .replace(" ... ", "...")\
                                 .replace(",¿", " ¿")\
                                 .replace(",.", ",")\
                                 .replace("?.", "? ")

    return modified_text

# -- For loop to format every original transcription (if not already processed)
for original_transcription in list_original_transcriptions:
	if original_transcription not in list_translated_transcriptions:
		transcription_df = pd.read_table(BASE_DIR + '/data/original_spanish_transcriptions/' + original_transcription, sep='|', header=None)
		transcription_df.rename(columns={0: 'time', 1: 'speaker', 2: 'transcript'}, inplace=True)
		
		transcription_df['time'] = pd.to_timedelta(transcription_df['time'])
		transcription_df['speaker_change'] = transcription_df['speaker'] != transcription_df['speaker'].shift()
		
		result = transcription_df.groupby(['speaker', transcription_df['speaker_change'].cumsum()]).agg({\
	                                                                                                 'time': ['min', 'max'],
	                                                                                                 'transcript': lambda x: '.'.join(x)
	                                                                                                })
		result.columns = result.columns.droplevel()
		result.columns = ['min_time', 'max_time', 'transcript']
		result.reset_index(inplace=True)
		result['min_time'] = result['min_time'].apply(lambda x: str(x).replace('0 days ', ''))
		result['max_time'] = result['max_time'].apply(lambda x: str(x).replace('0 days ', ''))

		# -- Preprocess transcript
		result['transcript'] = result['transcript'].apply(capitalize_proper_nouns)
		
		result['literal_transcript'] = 'Desde el instante ' + result['min_time'] + ' hasta ' + result['max_time'] +\
						' ' + result['speaker'] + ' dice: \"' + result['transcript'] + '\"'
		result['literal_transcript'] = result['literal_transcript'].apply(translate_text)
		
		result = result.sort_values('min_time')
		result_text = '\n\n'.join(result['literal_transcript'])
		
		# -- Save text as .txt file
		with open(BASE_DIR + '/data/translated_transcriptions/' + original_transcription, 'w') as f:
			json.dump(result_text, f)

print("Processing finished successfully!")
