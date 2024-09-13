# -- Import libraries
from   langchain.prompts             import PromptTemplate
from   PIL                           import Image
from   streamlit.logger              import get_logger
from   streamlit_player              import st_player
from   langchain.tools               import DuckDuckGoSearchRun
import pandas                        as pd
import streamlit                     as st
import urllib.request
import argparse
import together
import logging
import requests
import utils
import spacy
import time
import os
import re
st.set_page_config(layout="wide")

@st.cache_data
def get_args():
    # -- 1. Setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--DEFAULT_SYSTEM_PROMPT_LINK', type=str, default="https://raw.githubusercontent.com/AlbertoUAH/Castena/main/prompts/default_system_prompt.txt", help='Valor para DEFAULT_SYSTEM_PROMPT_LINK')
    parser.add_argument('--PODCAST_URL_VIDEO_PATH', type=str, default="https://raw.githubusercontent.com/AlbertoUAH/Castena/main/data/podcast_youtube_video.csv", help='Valor para PODCAST_URL_VIDEO_PATH')
    parser.add_argument('--TRANSCRIPTION', type=str, default='worldcast_roberto_vaquero', help='Name of the trascription')
    parser.add_argument('--MODEL', type=str, default='togethercomputer/llama-2-13b-chat', help='Model name')
    parser.add_argument('--EMB_MODEL', type=str, default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', help='Embedding model name')
    os.system("python -m spacy download es_core_news_lg")

    # -- 2. Setup env and logger
    logger = get_logger(__name__)

    # -- 3. Setup constants
    args = parser.parse_args()
    return args, logger

@st.cache_data
def get_podcast_data(path):
    podcast_url_video_df = pd.read_csv(path, sep=';')
    return podcast_url_video_df

@st.cache_resource(experimental_allow_widgets=True)
def get_basics_comp(emb_model, model, default_system_prompt_link, _logger, podcast_url_video_df, img_size=100):
    r    = requests.get("https://raw.githubusercontent.com/AlbertoUAH/Castena/main/media/castena-animated-icon.gif", stream=True)
    icon = Image.open(r.raw)
    icon = icon.resize((img_size, img_size))

    with st.sidebar.container():
        st.markdown(
            """
            <head>
                <style>
                    .footer1 {
                        text-align: center;
                    }
                </style>
            </head>
            <body>
                <div class="footer1">
                    <img src=https://raw.githubusercontent.com/AlbertoUAH/Castena/main/media/castena-animated-icon.gif width="150" height="150">
                </div>
                <br>
            </body>
            """,
            unsafe_allow_html=True,
    )


    genre = st.sidebar.radio(
                        "Seleccione el LLM",
                        ["LLAMA", "GPT"]
                    )
    st.sidebar.info('Modelo LLAMA: ' + str(model).split('/')[-1] + '\nModelo GPT: gpt-3.5-turbo', icon="ℹ️")
    podcast_list = list(podcast_url_video_df['podcast_name_lit'].apply(lambda x: x.replace("'", "")))
    video_option = st.sidebar.selectbox(
        "Seleccione el podcast",
        podcast_list,
        on_change=clean_chat
    )

    # -- Add icons
    with st.sidebar.container():
        st.markdown(
            """
            <head>
                <style>
                    .footer2 {
                        position: fixed;
                        bottom: 2%;
                        left: 6.5%;
                    }

                    .footer2 a {
                        margin: 10px;
                        text-decoration: none;
                    }
                </style>
            </head>
            <body>
                <div class="footer2">
                    <a href="https://www.linkedin.com/in/alberto-fernandez-hernandez-3a3474136">
                        <img src="https://cdn-icons-png.flaticon.com/128/3536/3536505.png" width="32" height="32">
                    </a>
                    <a href="https://github.com/AlbertoUAH/Castena">
                        <img src="https://cdn-icons-png.flaticon.com/128/733/733553.png" width="32" height="32">
                    </a>
                    <a href="https://www.buymeacoffee.com/castena">
                        <img src="https://cdn-icons-png.flaticon.com/128/761/761767.png" width="32" height="32">
                    </a>
                </div>
            </body>
            """,
            unsafe_allow_html=True,
    )


    video_option_joined = '_'.join(video_option.replace(': Entrevista a ', ' ').lower().split(' ')).replace("\'", "")
    video_option_joined_path = "{}_transcription.txt".format(video_option_joined)
    youtube_video_url   = list(podcast_url_video_df[podcast_url_video_df['podcast_name'].str.contains(video_option_joined)]['youtube_video_url'])[0].replace("\'", "")
    st.title("[Podcast: {}]({})".format(video_option.replace("'", "").title(), youtube_video_url))

    # -- 4. Setup request for system prompt
    f = urllib.request.urlopen(default_system_prompt_link)
    default_system_prompt = str(f.read(), 'UTF-8')

    # -- 5. Setup app
    nlp, retriever = utils.setup_app(video_option_joined_path, emb_model, model, _logger)

    # -- 6. Setup model
    together.api_key = os.environ["TOGETHER_API_KEY"]
    #together.Models.start(model)
    return together, nlp, retriever, video_option, video_option_joined_path, default_system_prompt, youtube_video_url, genre

def clean_chat():
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state.messages     = [{'role': 'assistant', 'content': 'Nuevo chat creado'}]

def main():
    args, logger = get_args()
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS   = "<<SYS>>\n", "\n<</SYS>>\n\n"

    PODCAST_URL_VIDEO_PATH     = args.PODCAST_URL_VIDEO_PATH
    DEFAULT_SYSTEM_PROMPT_LINK = args.DEFAULT_SYSTEM_PROMPT_LINK
    TRANSCRIPTION              = args.TRANSCRIPTION
    TRANSCRIPTION_PATH         = '{}_transcription.txt'.format(TRANSCRIPTION)
    MODEL                      = args.MODEL
    EMB_MODEL                  = args.EMB_MODEL
    WIDTH                      = 50
    SIDE                       = (100 - WIDTH) / 2

    
    podcast_url_video_df = get_podcast_data(PODCAST_URL_VIDEO_PATH)

    together, nlp, retriever, video_option, video_option_joined_path, default_system_prompt, youtube_video_url, genre = get_basics_comp(EMB_MODEL, MODEL, DEFAULT_SYSTEM_PROMPT_LINK, logger, 
                                                                                                podcast_url_video_df, img_size=100)

    # -- 6. Setup prompt template + llm chain
    instruction = """CONTEXTO:/n/n {context}/n

PREGUNTA: {question}

RESPUESTA: """
    prompt_template = utils.get_prompt(instruction, default_system_prompt, B_SYS, E_SYS, B_INST, E_INST, logger)

    llama_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": llama_prompt}

    qa_chain = utils.create_llm_chain(MODEL, retriever, chain_type_kwargs, logger, video_option_joined_path)

    # ---------------------------------------------------------------------
    if st.button('Info.'):
        search = DuckDuckGoSearchRun()
        character_name = video_option.replace("'", "").title().split("Entrevista A ")[-1]
        info = search.run("¿Quien es {}?".format(character_name))
        character_info = utils.get_character_info_gpt(info, character=character_name)
        st.info(character_info)
    
    _, container, _ = st.columns([SIDE, WIDTH, SIDE])
    with container:
        st_player(utils.typewrite(youtube_video_url))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("¡Pregunta lo que quieras!"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            if 'GPT' not in genre:
                if prompt.lower() == 'resume':
                    llm_response = utils.summarise_doc(video_option_joined_path, model_name='llama', model=MODEL)
                    st.markdown(llm_response)
                else:
                    llm_response = qa_chain(prompt)['result']
                    llm_response = utils.process_llm_response(llm_response)
                    st.markdown(llm_response)
        
                    start_time_str_list = []; start_time_seconds_list = []; end_time_seconds_list = []
                    for response in llm_response.split('\n'):
                        if re.search(r'(\d{2}:\d{2}:\d{2}(.\d{6})?)', response) != None:
                            start_time_str, start_time_seconds, _, end_time_seconds = utils.add_hyperlink_and_convert_to_seconds(response)
                            start_time_str_list.append(start_time_str)
                            start_time_seconds_list.append(start_time_seconds)
                            end_time_seconds_list.append(end_time_seconds)
        
                    if start_time_str_list:
                        for start_time_seconds, start_time_str, end_time_seconds in zip(start_time_seconds_list, start_time_str_list, end_time_seconds_list):
                            st.markdown("__Fragmento: " + start_time_str + "__")
                            _, container, _ = st.columns([SIDE, WIDTH, SIDE])
                            with container:
                                st_player(youtube_video_url.replace("?enablejsapi=1", "") + f'?start={start_time_seconds}&end={end_time_seconds}')
            else:
                if prompt.lower() == 'resume':
                    llm_response = utils.summarise_doc(video_option_joined_path, model_name='gpt')
                    st.markdown(llm_response)
                else:
                    llm_response = utils.get_gpt_response(video_option_joined_path, prompt, logger)
                    llm_response = utils.process_llm_response(llm_response)
                    st.markdown(llm_response)
        
                    start_time_str_list = []; start_time_seconds_list = []; end_time_seconds_list = []
                    for response in llm_response.split('\n'):
                        if re.search(r'(\d{2}:\d{2}:\d{2}(.\d{6})?)', response) != None:
                            start_time_str, start_time_seconds, _, end_time_seconds = utils.add_hyperlink_and_convert_to_seconds(response)
                            start_time_str_list.append(start_time_str)
                            start_time_seconds_list.append(start_time_seconds)
                            end_time_seconds_list.append(end_time_seconds)
        
                    if start_time_str_list:
                        for start_time_seconds, start_time_str, end_time_seconds in zip(start_time_seconds_list, start_time_str_list, end_time_seconds_list):
                            st.markdown("__Fragmento: " + start_time_str + "__")
                            _, container, _ = st.columns([SIDE, WIDTH, SIDE])
                            with container:
                                st_player(youtube_video_url.replace("?enablejsapi=1", "") + f'?start={start_time_seconds}&end={end_time_seconds}')
                        
        st.session_state.messages.append({"role": "assistant", "content": llm_response})
# -- Sample: streamlit run app.py -- --DEFAULT_SYSTEM_PROMPT_LINK=https://raw.githubusercontent.com/AlbertoUAH/Castena/main/prompts/default_system_prompt.txt --PODCAST_URL_VIDEO_PATH=https://raw.githubusercontent.com/AlbertoUAH/Castena/main/data/podcast_youtube_video.csv --TRANSCRIPTION=worldcast_roberto_vaquero --MODEL=togethercomputer/llama-2-7b-chat --EMB_MODEL=BAAI/bge-base-en-v1.5
if __name__ == '__main__':
    main()
