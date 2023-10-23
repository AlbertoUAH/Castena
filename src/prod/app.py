# -- Import libraries
from   langchain.prompts           import PromptTemplate
from   streamlit.logger            import get_logger
import pandas                      as pd
import streamlit                   as st
import urllib.request
import argparse
import together
import logging
import utils
import spacy
import time
import os

def main():
    # -- 1. Setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--DEFAULT_SYSTEM_PROMPT_LINK', type=str, default="https://raw.githubusercontent.com/AlbertoUAH/Castena/main/prompts/default_system_prompt.txt", help='Valor para DEFAULT_SYSTEM_PROMPT_LINK')
    parser.add_argument('--PODCAST_URL_VIDEO_PATH', type=str, default="https://raw.githubusercontent.com/AlbertoUAH/Castena/main/data/podcast_youtube_video.csv", help='Valor para PODCAST_URL_VIDEO_PATH')
    parser.add_argument('--TRANSCRIPTION', type=str, default='worldcast_roberto_vaquero', help='Name of the trascription')
    parser.add_argument('--MODEL', type=str, default='togethercomputer/llama-2-7b-chat', help='Model name')
    parser.add_argument('--EMB_MODEL', type=str, default='BAAI/bge-base-en-v1.5', help='Embedding model name')
    os.system("python -m spacy download es_core_news_lg")

    # -- 2. Setup env and logger
    os.environ["TOGETHER_API_KEY"] = "6101599d6e33e3bda336b8d007ca22e35a64c72cfd52c2d8197f663389fc50c5"
    logger = get_logger(__name__)

    # -- 3. Setup constants
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS   = "<<SYS>>\n", "\n<</SYS>>\n\n"
    args = parser.parse_args()
    PODCAST_URL_VIDEO_PATH     = args.PODCAST_URL_VIDEO_PATH
    DEFAULT_SYSTEM_PROMPT_LINK = args.DEFAULT_SYSTEM_PROMPT_LINK
    TRANSCRIPTION              = args.TRANSCRIPTION
    TRANSCRIPTION_PATH         = '{}_transcription.txt'.format(TRANSCRIPTION)
    MODEL                      = args.MODEL
    EMB_MODEL                  = args.EMB_MODEL
    SOCIAL_ICONS               = {
        "LinkedIn": ["https://www.linkedin.com/in/alberto-fernandez-hernandez-3a3474136/", "https://icon.signature.email/social/linkedin-circle-medium-0077b5-FFFFFF.png"],
        "GitHub": ["https://github.com/AlbertoUAH", "https://icon.signature.email/social/github-circle-medium-24292e-FFFFFF.png"]
    }
    social_icons_html = [f"<a href='{SOCIAL_ICONS[platform][0]}' target='_blank' style='margin-right: 10px;'><img class='social-icon' src='{SOCIAL_ICONS[platform][1]}'' alt='{platform}''></a>" for platform in SOCIAL_ICONS]

    together.api_key = os.environ["TOGETHER_API_KEY"]
    together.Models.start(MODEL)
    podcast_url_video_df = pd.read_csv(PODCAST_URL_VIDEO_PATH, sep=';')
    youtube_video_url    = list(podcast_url_video_df[podcast_url_video_df['podcast_name'] == "\'" + TRANSCRIPTION + "\'"]['youtube_video_url'])[0].replace("\'", "")

    # -- 4. Setup request for system prompt
    f = urllib.request.urlopen(DEFAULT_SYSTEM_PROMPT_LINK)
    DEFAULT_SYSTEM_PROMPT = str(f.read(), 'UTF-8')

    # -- 5. Setup app
    translator, nlp, retriever = utils.setup_app(TRANSCRIPTION_PATH, EMB_MODEL, MODEL, logger)


    # -- 6. Setup prompt template + llm chain
    instruction = """CONTEXT:/n/n {context}/n

    Question: {question}"""
    prompt_template = utils.get_prompt(instruction, DEFAULT_SYSTEM_PROMPT, B_SYS, E_SYS, B_INST, E_INST, logger)

    llama_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": llama_prompt}

    qa_chain = utils.create_llm_chain(MODEL, retriever, chain_type_kwargs, logger)

    # ---------------------------------------------------------------------
    # -- 7. Setup Streamlit app
    st.title("Podcast: {}".format(' '.join(x.capitalize() for x in TRANSCRIPTION.split('_'))))
    st.image("https://raw.githubusercontent.com/AlbertoUAH/autexTification/main/media/{}.jpeg".format(TRANSCRIPTION))

    original_input_text = st.text_input("Pregunta")
    if st.button("Consultar") or original_input_text:
        translated_input_text = utils.translate_text(original_input_text, nlp, target_lang='en')
        logger.info('A query has been launched. Query: {}'.format(original_input_text))
        logger.info('Waiting for response...')
        llm_response = qa_chain(translated_input_text)
        llm_response = utils.process_llm_response(llm_response, nlp).replace(': ', ':<br>').replace('. ', '.<br>').replace('" ', '"<br>')
        logger.info('Response recieved successfully! {}'.format(llm_response))
        typewrited_llm_response = utils.typewrite(utils.add_hyperlink_and_convert_to_seconds(llm_response), youtube_video_url)
        st.components.v1.html(typewrited_llm_response, width=800, height=750, scrolling=True)

    st.write(f"""<div class="subtitle" style="text-align: center;">Información de contacto</div>""", unsafe_allow_html=True)
    st.write(f"""
        <div style="display: flex; justify-content: center; margin-bottom: 10px;">
            {''.join(social_icons_html)}
        </div>""", 
        unsafe_allow_html=True
    )

# -- Sample: streamlit run app.py -- --DEFAULT_SYSTEM_PROMPT_LINK=https://raw.githubusercontent.com/AlbertoUAH/Castena/main/prompts/default_system_prompt.txt --PODCAST_URL_VIDEO_PATH=https://raw.githubusercontent.com/AlbertoUAH/Castena/main/data/podcast_youtube_video.csv --TRANSCRIPTION=worldcast_roberto_vaquero --MODEL=togethercomputer/llama-2-7b-chat --EMB_MODEL=BAAI/bge-base-en-v1.5
if __name__ == '__main__':
    main()
