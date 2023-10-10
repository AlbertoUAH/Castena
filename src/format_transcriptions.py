# -- Format transcriptions + translation into English (if necessary)
import pandas as pd
import itertools
import json
import re
import os

BASE_DIR = '/home/runner/work/Castena/Castena'

list_original_transcriptions   = [file for file in os.listdir(BASE_DIR + '/data/original_spanish_transcriptions/') if '.txt' in file]
list_translated_transcriptions = [file for file in os.listdir(BASE_DIR + '/data/translated_transcriptions/') if '.txt' in file]

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
	result['literal_transcript'] = 'Desde el instante ' + result['min_time'] + ' hasta ' + result['max_time'] +\
										 ' ' + result['speaker'] + ' dice: \"' + result['transcript'] + '\"'
	result = result.sort_values('min_time')

	result_text = '\n\n'.join(result['literal_transcript'])
	# -- Save text as .txt file
	with open(BASE_DIR + '/data/translated_transcriptions/' + original_transcription, 'w') as f:
      json.dump(result_text, f)

print("Processing finished successfully!")