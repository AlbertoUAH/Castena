# -- Utils .py file
# -- Libraries
from   typing                      import Any, Dict, List, Mapping, Optional
from   pydantic                    import Extra, Field, root_validator
from   langchain.llms.base         import LLM
from   langchain.utils             import get_from_dict_or_env
from   langchain.vectorstores      import Chroma
from   langchain.text_splitter     import RecursiveCharacterTextSplitter
from   langchain.chains            import RetrievalQA
from   langchain.document_loaders  import TextLoader
from   langchain.embeddings        import SelfHostedHuggingFaceEmbeddings
from   googletrans                 import Translator
import streamlit as st
import together
import textwrap
import spacy
import os
import re

os.environ["TOGETHER_API_KEY"] = "6101599d6e33e3bda336b8d007ca22e35a64c72cfd52c2d8197f663389fc50c5"

# -- LLM class
class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    original_transcription: str = ""
    """Original transcription"""

    class Config:
        extra = Extra.forbid

    #@root_validator(skip_on_failure=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def clean_duplicates(self, transcription: str) -> str:
      transcription = transcription.strip().replace('/n/n ', """
""")
      new_transcription_aux = []
      for text in transcription.split('\n\n'):
          if text not in new_transcription_aux:
            is_substring = any(transcription_aux.replace('"', '').lower() in text.replace('"', '').lower()\
                               for transcription_aux in new_transcription_aux)
            if not is_substring:
                new_transcription_aux.append(text)
      return '\n\n'.join(new_transcription_aux)

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        regex_transcription = r'CONTEXTO:(\n.*)+PREGUNTA'
        regex_init_transcription = r"Desde el instante [0-9]+:[0-9]+:[0-9]+(?:\.[0-9]+)? hasta [0-9]+:[0-9]+:[0-9]+(?:\.[0-9]+)? [a-zA-Záéíóú ]+ dice: ?"

        # -- Extract transcription
        together.api_key = self.together_api_key
        cleaned_prompt   = self.clean_duplicates(prompt)
        resultado        = re.search(regex_transcription, cleaned_prompt, re.DOTALL)

        resultado        = re.sub(regex_init_transcription, "", resultado.group(1).strip()).replace('\"', '')
        resultado_alpha_num = [re.sub(r'\W+', ' ', resultado_aux).strip().lower() for resultado_aux in resultado.split('\n\n')]

        # -- Setup new transcription format, without duplicates and with its correspondent speaker
        new_transcription = []
        for transcription in self.original_transcription.split('\n\n'):
          transcription_cleaned = re.sub(regex_init_transcription, "", transcription.strip()).replace('\"', '')
          transcription_cleaned = re.sub(r'\W+', ' ', transcription_cleaned).strip().lower()
          for resultado_aux in resultado_alpha_num:
            if resultado_aux in transcription_cleaned:
              init_transcription = re.search(regex_init_transcription, transcription).group(0)
              new_transcription.append(init_transcription + '\"' + resultado_aux + '\"')
        # -- Merge with original transcription
        new_transcription = '\n\n'.join(list(set(new_transcription)))
        new_cleaned_prompt = re.sub(regex_transcription, f"""CONTEXTO:
{new_transcription}
PREGUNTA:""", cleaned_prompt, re.DOTALL)
        print(new_cleaned_prompt)
        output = together.Complete.create(new_cleaned_prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        text = self.clean_duplicates(text)
        return text

# -- Python function to setup basic features: translator, SpaCy pipeline and LLM model
@st.cache_resource
def setup_app(transcription_path, emb_model, model, _logger):
    # -- Setup enviroment and features
    print("Setup googletranslator")
    translator = Translator(service_urls=['translate.googleapis.com'])
    nlp        = spacy.load('es_core_news_lg')

    _logger.info('Setup environment and features...')

    # -- Setup LLM
    together.api_key = os.environ["TOGETHER_API_KEY"]
    # List available models and descriptons
    print("Setup models")
    models = together.Models.list()
    # Set llama2 7b LLM
    #together.Models.start(model)
    _logger.info('Setup environment and features - FINISHED!')

    # -- Read translated transcription
    _logger.info('Loading transcription...')
    print("Load text loader")
    loader = TextLoader('./src/prod/' + transcription_path)
    documents = loader.load()
    # Splitting the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    _logger.info('Loading transcription - FINISHED!')

    # -- Load embedding
    _logger.info('Loading embedding...')
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    print("Load HuggingFace embeddings: {}".format(emb_model))
    model_norm = SelfHostedHuggingFaceEmbeddings(
        model_name=emb_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs=encode_kwargs
    )
    print("Finished loading embedding")
    _logger.info('Loading embedding - FINISHED!')

    # -- Create document database
    print("Before log")
    _logger.info('Creating document database...')
    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = 'db'
    ## Here is the nmew embeddings being used
    print("embedding = model_norm")
    embedding = model_norm
    print("Load vector db")
    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)

    # -- Make a retreiver
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    _logger.info('Creating document database - FINISHED!')
    _logger.info('Setup finished!')
    return translator, nlp, retriever

# -- Function to get prompt template
def get_prompt(instruction, system_prompt, b_sys, e_sys, b_inst, e_inst, _logger):
    new_system_prompt = b_sys + system_prompt + e_sys
    prompt_template   =  b_inst + new_system_prompt + instruction + e_inst
    _logger.info('Prompt template created: {}'.format(instruction))
    return prompt_template

# -- Function to create the chain to answer questions
@st.cache_resource
def create_llm_chain(model, _retriever, _chain_type_kwargs, _logger, transcription_path):
    _logger.info('Creating LLM chain...')
    # -- Keep original transcription
    with open(transcription_path, 'r') as f:
        formatted_transcription = f.read()
    
    llm = TogetherLLM(
        model= model,
        temperature = 0.0,
        max_tokens = 1024,
        original_transcription = formatted_transcription
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=_retriever,
                                           chain_type_kwargs=_chain_type_kwargs,
                                           return_source_documents=True)
    _logger.info('Creating LLM chain - FINISHED!')
    return qa_chain

# -------------------------------------------
# -- Auxiliar functions
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response, nlp):
  response = llm_response['result']
  return wrap_text_preserve_newlines(response)


def time_to_seconds(time_str):
    parts = time_str.split(':')
    hours, minutes, seconds = map(float, parts)
    return int((hours * 3600) + (minutes * 60) + seconds)

# -- Extract seconds from transcription
def add_hyperlink_and_convert_to_seconds(text):
    time_pattern = r'(\d{2}:\d{2}:\d{2}(?:.\d{6})?)'
    
    def get_seconds(match):
        start_time_str, end_time_str = match[0], match[1]
        start_time_seconds = time_to_seconds(start_time_str)
        end_time_seconds   = time_to_seconds(end_time_str)
        return start_time_str, start_time_seconds, end_time_str, end_time_seconds
    start_time_str, start_time_seconds, end_time_str, end_time_seconds = get_seconds(re.findall(time_pattern, text))
    return start_time_str, start_time_seconds, end_time_str, end_time_seconds

# -- Streamlit HTML template
def typewrite(youtube_video_url, i=0):
    youtube_video_url = youtube_video_url.replace("?enablejsapi=1", "")
    margin = "{margin: 0;}"
    html = f"""
        <html>
        <style>
          p {margin}
        </style>
        <body>
          <script src="https://www.youtube.com/player_api"></script>
          <p align="center">
              <iframe id="player_{i}" src="{youtube_video_url}" width="600" height="450"></iframe>
          </p>
        </body>
        </html>
    """
    return html
