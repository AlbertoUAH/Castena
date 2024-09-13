# -- Utils .py file
# -- Libraries
from   typing                      import Any, Dict, List, Mapping, Optional
from   pydantic                    import Extra, Field, root_validator
from   langchain_community.vectorstores import FAISS
from   langchain_core.runnables    import RunnablePassthrough
from   langchain.llms.base         import LLM
from   langchain.chat_models       import ChatOpenAI
from   langchain_core.prompts      import ChatPromptTemplate
from   langchain.prompts           import PromptTemplate
from   langchain.schema            import StrOutputParser
from   langchain.utils             import get_from_dict_or_env
from   langchain.vectorstores      import Chroma
from   langchain.text_splitter     import RecursiveCharacterTextSplitter, CharacterTextSplitter
from   langchain.chains            import RetrievalQA, MapReduceDocumentsChain, ReduceDocumentsChain
from   langchain.document_loaders  import TextLoader
from   langchain.embeddings        import HuggingFaceEmbeddings, OpenAIEmbeddings
from   langchain.chains            import LLMChain
from   langchain.evaluation        import StringEvaluator
from   typing                      import Any, Optional
from   langsmith.client            import Client
from   langchain.smith             import RunEvalConfig, run_on_dataset
from   langchain.chains.combine_documents.stuff import StuffDocumentsChain
import streamlit as st
import together
import textwrap
import getpass
import spacy
import os
import re

#os.environ["TOGETHER_API_KEY"] = "6101599d6e33e3bda336b8d007ca22e35a64c72cfd52c2d8197f663389fc50c5"
#os.environ["OPENAI_API_KEY"]   = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

client = Client()

# -- LLM class
class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.0
    """What sampling temperature to use."""

    max_tokens: int = 1024
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

# -- Langchain evaluator
class RelevanceEvaluator(StringEvaluator):
    """An LLM-based relevance evaluator."""

    def __init__(self):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        template = """En una escala del 0 al 100, ¿Como de relevante es la siguiente salida con respecto a la siguiente entrada?
        --------
        ENTRADA: {input}
        --------
        SALIDA: {prediction}
        --------
        Razona paso a paso porqué el score que has elegido es apropiado y despues muestra la puntuacion al final."""

        self.eval_chain = LLMChain.from_string(llm=llm, template=template)

    @property
    def requires_input(self) -> bool:
        return True

    @property
    def requires_reference(self) -> bool:
        return False

    @property
    def evaluation_name(self) -> str:
        return "scored_relevance"

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any
    ) -> dict:
        evaluator_result = self.eval_chain(
            dict(input=input, prediction=prediction), **kwargs
        )
        reasoning, score = evaluator_result["text"].split("\n", maxsplit=1)
        score = re.search(r"\d+", score).group(0)
        if score is not None:
            score = float(score.strip()) / 100.0
        return {"score": score, "reasoning": reasoning.strip()}

# -- Get GPT response
def get_gpt_response(transcription_path, query, logger):
    template = """Eres un asistente. Su misión es proporcionar respuestas precisas a preguntas relacionadas con la transcripción de una entrevista de YouTube.
    No saludes en tu respuesta. No repita la pregunta en su respuesta. Sea conciso y omita las exenciones de responsabilidad o los mensajes predeterminados.
    Solo responda la pregunta, no agregue texto adicional. No des tu opinión personal ni tu conclusión personal. No haga conjeturas ni suposiciones.
    Si no sabe la respuesta de la pregunta o el contexto está vacío, responda cortésmente por qué no sabe la respuesta. Por favor no comparta información falsa.
    {context}
    Pregunta: {question}
    Respuesta:"""
    
    rag_prompt_custom = PromptTemplate.from_template(template)
    loader = TextLoader(transcription_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=250, length_function=len)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)    
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )
    llm_output = rag_chain.invoke(query)
    return llm_output

def get_character_info_gpt(text, character):
    vectorstore = FAISS.from_texts(
        [text], embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()
    
    template = """Responde a la siguiente pregunta basandote unicamente en el siguiente contexto:
    {context}
    
    Pregunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    model = ChatOpenAI()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain.invoke("¿Quien es {}?".format(character))

    
# -- Text summarisation with OpenAI (map-reduce technique)
def summarise_doc(transcription_path, model_name, model=None):
    if model_name == 'gpt':
        llm = ChatOpenAI(temperature=0, max_tokens=1024)
        
        # -- Map
        loader = TextLoader(transcription_path)
        docs   = loader.load()
        map_template = """Lo siguiente es listado de fragmentos de una conversacion:
        {docs}
        En base a este listado, por favor identifica los temas/topics principales.
        Respuesta:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=llm, prompt=map_prompt)
    
        # -- Reduce
        reduce_template = """A continuacion se muestra un conjunto de resumenes:
        {docs}
        Usalos para crear un unico resumen consolidado de todos los temas/topics principales. 
        Respuesta:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
    
        # Run chain
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
        
        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )
        
        # Combines and iteravely reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=3000,
        )
    
        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )
        
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=3000, chunk_overlap=0
        )
        split_docs  = text_splitter.split_documents(docs)
        doc_summary = map_reduce_chain.run(split_docs)
    else:
        # -- Keep original transcription
        with open(transcription_path, 'r') as f:
            docs = f.read()

        print("READ DOCUMENT")
        print(docs)
        
        llm = TogetherLLM(
            model= model,
            temperature = 0.0,
            max_tokens = 1024,
            original_transcription = docs
        )
        
        # Map
        map_template = """Lo siguiente es un extracto de una conversación entre dos hablantes en español.
{docs}
Por favor resuma la conversación en español.
Resumen:"""
        map_prompt = PromptTemplate(template=map_template, input_variables=["docs"])
        map_chain  = LLMChain(llm=llm, prompt=map_prompt)
        
        # Reduce
        reduce_template = """Lo siguiente es una lista de resumenes en español:
{doc_summaries}
Tómelos y descríbalos en un resumen final consolidado en español. Además, enumera los temas principales de la conversación en español.

Resumen:"""
        reduce_prompt   = PromptTemplate(template=reduce_template, input_variables=["doc_summaries"])
        
        # Run chain
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
        
        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="doc_summaries"
        )
        
        # Combines and iteravely reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            verbose=True,
            token_max=1024
        )
        
        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
            verbose=True
        )
        text_splitter = CharacterTextSplitter(
            separator = "\n\n",
            chunk_size = 2000,
            chunk_overlap  = 50,
            length_function = len,
            is_separator_regex = True,
        )
        split_docs  = text_splitter.create_documents([docs])
        doc_summary = map_reduce_chain.run(split_docs)
    
    return doc_summary

# -- Python function to setup basic features: SpaCy pipeline and LLM model
@st.cache_resource
def setup_app(transcription_path, emb_model, model, _logger):
    # -- Setup enviroment and features
    nlp        = spacy.load('es_core_news_lg')

    _logger.info('Setup environment and features...')

    # -- Setup LLM
    together.api_key = os.environ["TOGETHER_API_KEY"]
    # List available models and descriptons
    models = together.Models.list()
    # Set llama2 7b LLM
    #together.Models.start(model)
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
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    _logger.info('Creating document database - FINISHED!')
    _logger.info('Setup finished!')
    return nlp, retriever

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

def process_llm_response(llm_response):
  return wrap_text_preserve_newlines(llm_response)


def time_to_seconds(time_str):
    parts = time_str.split(':')
    hours, minutes, seconds = map(float, parts)
    return int((hours * 3600) + (minutes * 60) + seconds)

# -- Extract seconds from transcription
def add_hyperlink_and_convert_to_seconds(text):
    time_pattern = r'(\d{2}:\d{2}:\d{2}(?:.\d{6})?)'
    
    def get_seconds(match):
        if len(match) == 2:
            start_time_str, end_time_str = match[0], match[1]
        else:
            start_time_str = match[0]
            end_time_str   = re.findall(r"Desde el instante {} hasta {}".format(start_time_str, time_pattern))[0].split('hasta ')[-1]
            
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
