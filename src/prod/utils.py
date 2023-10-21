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
from   langchain.embeddings        import HuggingFaceEmbeddings
from   langchain.prompts           import PromptTemplate
from   googletrans                 import Translator
import streamlit as st
import together
import textwrap
import spacy
import os
import re

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

    class Config:
        extra = Extra.forbid

    @root_validator()
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
      lines = transcription.strip().split('\n')

      new_transcription = []

      for linea in lines:
          if linea.replace('CONTEXT:/n/n ', '').replace('/n', '') not in new_transcription and linea != '':
              new_transcription.append(linea.replace('CONTEXT:/n/n ', '').replace('/n', ''))
      # Create new transcription without duplicates
      new_transcription = '\n\n'.join(new_transcription).replace("""<</SYS>>
      """, """<</SYS>>
      CONTEXT: """)
      return new_transcription

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        cleaned_text = self.clean_duplicates(text)
        return cleaned_text


# -- Python function to setup basic features: translator, SpaCy pipeline and LLM model
@st.cache_resource
def setup_app(transcription_path, emb_model, model, _logger):
    # -- Setup enviroment and features
    translator = Translator(service_urls=['translate.googleapis.com'])
    nlp        = spacy.load('es_core_news_lg')

    _logger.info('Setup environment and features...')

    # -- Setup LLM
    together.api_key = os.environ["TOGETHER_API_KEY"]
    # List available models and descriptons
    models = together.Models.list()
    # Set llama2 7b LLM
    together.Models.start(model)
    _logger.info('Setup environment and features - FINISHED!')

    # -- Read translated transcription
    _logger.info('Loading transcription...')
    loader = TextLoader(transcription_path)
    documents = loader.load()
    # Splitting the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    _logger.info('Loading transcription - FINISHED!')

    # -- Load embedding
    _logger.info('Loading embedding...')
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    model_norm = HuggingFaceEmbeddings(
        model_name=emb_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs=encode_kwargs
    )
    _logger.info('Loading embedding - FINISHED!')

    # -- Create document database
    _logger.info('Creating document database...')
    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = 'db'
    ## Here is the nmew embeddings being used
    embedding = model_norm

    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)

    # -- Make a retreiver
    retriever = vectordb.as_retriever(search_type="similarity_score_threshold",
                                      search_kwargs={"k": 7, "score_threshold": 0.5})
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
def create_llm_chain(model, _retriever, _chain_type_kwargs, _logger):
    _logger.info('Creating LLM chain...')
    llm = TogetherLLM(
        model= model,
        temperature = 0.0,
        max_tokens = 1024
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
def translate_text(text, nlp, target_lang='en'):
    # Traducir el texto sin los nombres propios
    translator = Translator()
    # Tokenizar el texto y encontrar nombres propios
    doc = nlp(text)
    named_entities = [ent.text for ent in doc if ent.pos_ == 'PROPN' and ent.dep_ in ['NNP', 'NN']]
    named_entities_list = []
    # Reemplazar los nombres propios con marcadores temporales
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
  return wrap_text_preserve_newlines(translate_text(response, nlp, target_lang='es'))


def time_to_seconds(time_str):
    parts = time_str.split(':')
    hours, minutes, seconds = map(float, parts)
    return int((hours * 3600) + (minutes * 60) + seconds)

def add_hyperlink_and_convert_to_seconds(text):
    time_pattern = r'(\d{2}:\d{2}:\d{2}.\d{6})'
    
    def replace_with_hyperlink(match):
        time_str = match.group(1)
        seconds = time_to_seconds(time_str)
        link = f'<button type="button" class="button" onclick="seek({seconds})">{time_str}</button>'
        return link
    
    modified_text = re.sub(time_pattern, replace_with_hyperlink, text)
    return modified_text

# -- Streamlit HTML template
def typewrite(text:str):
    js = """var player, seconds = 0;
            function onYouTubeIframeAPIReady() {
                console.log("player");
                player = new YT.Player('player', {
                    events: {
                      'onReady': onPlayerReady
                    }
                  });
            }

            function onPlayerReady(event) {
                event.target.playVideo();
            }


            function seek(sec){
                if(player){
                    player.seekTo(sec, true);
                }
            }
        """

    css = """
    .button {
      background-color: transparent;
      font-family: "Tahoma sans-serif;", monospace;
      color: red;
      font-weight: bold;
      border: none;
      text-align: center;
      text-decoration: none;
      display: inline-block;
      font-size: 16px;
      cursor: pointer;
    }
    body {
      color: white;
      font-family: "Tahoma sans-serif;", monospace;
      font-weight: 450;
    }
    """

    html = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <title>Modificar iframe</title>
        </head>
        <style>
            {css}
        </style>
        <body>
          <script src="https://www.youtube.com/player_api"></script>
          <p>{text}</p>
          <br/>
          <iframe id="player" type="text/html" src="https://www.youtube.com/embed/4sXT1tHVbjE?enablejsapi=1" scrolling="yes" frameborder="0" width="600" height="450"></iframe>
          <script>
            {js}
           </script>
        </body>
        </html>
    """
    return html
