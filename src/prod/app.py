# -- Import libraries
from   streamlit.logger            import get_logger
import pandas                      as pd
import streamlit                   as st
import urllib.request
import logging
import time
import os

# -- 1. Setup env and logger
os.environ["TOGETHER_API_KEY"] = "6101599d6e33e3bda336b8d007ca22e35a64c72cfd52c2d8197f663389fc50c5"
logger = get_logger(__name__)

# -- 2. Setup constants
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS   = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT_LINK = "https://raw.githubusercontent.com/AlbertoUAH/Castena/main/prompts/default_system_prompt.txt"
TRANSCRIPTION_PATH = 'worldcast_roberto_vaquero_transcription.txt'
MODEL = 'togethercomputer/llama-2-7b-chat'
EMB_MODEL = "BAAI/bge-base-en-v1.5"

# -- 3. Setup request for system prompt
f = urllib.request.urlopen(DEFAULT_SYSTEM_PROMPT_LINK)
DEFAULT_SYSTEM_PROMPT = str(f.read(), 'UTF-8')

# -- 4. Setup app
from utils import *
translator, nlp, retriever = setup_app(TRANSCRIPTION_PATH, EMB_MODEL, MODEL, logger)


# -- 5. Setup prompt template + llm chain
instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""
prompt_template = get_prompt(instruction, DEFAULT_SYSTEM_PROMPT, B_SYS, E_SYS, B_INST, E_INST, logger)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": llama_prompt}

qa_chain = create_llm_chain(MODEL, retriever, chain_type_kwargs, logger)

# ---------------------------------------------------------------------
# -- 6. Setup Streamlit app
st.title("Entrevista Roberto Vaquero - Worldca$t")
st.image("https://raw.githubusercontent.com/AlbertoUAH/autexTification/main/media/worldcast_roberto_vaquero.jpeg")

original_input_text = st.text_input("Pregunta")
if st.button("Consultar") or original_input_text:
    translated_input_text = translate_text(original_input_text, nlp, target_lang='en')
    logger.info('A query has been launched. Query: {}'.format(original_input_text))
    logger.info('Waiting for response...')
    llm_response = qa_chain(translated_input_text)
    llm_response = process_llm_response(llm_response, nlp)
    logger.info('Response recieved successfully! {}'.format(llm_response))
    typewrited_llm_response = typewrite(add_hyperlink_and_convert_to_seconds(llm_response))
    st.components.v1.html(typewrited_llm_response, width=800, height=750, scrolling=True)