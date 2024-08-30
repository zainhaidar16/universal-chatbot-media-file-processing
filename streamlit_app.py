import os
import time
import base64
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from pypdf import PdfReader, PdfWriter, PdfMerger
from google.auth import credentials

# Path to your credentials file
CREDENTIALS_PATH = './cred.json'

# Initialize Google Cloud Vertex AI with the credentials file
vertexai.init(project='real-time-rag', location='us-central1', credentials=CREDENTIALS_PATH)

def page_setup():
    st.header("Chat with different types of media/files!", anchor=False, divider="blue")
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def get_typeofmedia():
    st.sidebar.header("Select type of Media", divider='orange')
    media_type = st.sidebar.radio("Choose one:",
                                  ("PDF files", "Images", "Video, mp4 file", "Audio files"))
    return media_type

def get_llminfo():
    st.sidebar.header("Options", divider='rainbow')
    tip1 = "Select a model you want to use."
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("gemini-1.5-flash", "gemini-1.5-pro"), help=tip1)
    tip2 = "Lower temperatures are good for prompts that require a less open-ended or creative response, while higher temperatures can lead to more diverse or creative results. A temperature of 0 means that the highest probability tokens are always selected."
    temp = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=1.0, step=0.25, help=tip2)
    tip3 = "Used for nucleus sampling. Specify a lower value for less random responses and a higher value for more random responses."
    topp = st.sidebar.slider("Top P:", min_value=0.0, max_value=1.0, value=0.94, step=0.01, help=tip3)
    tip4 = "Number of response tokens, 8194 is limit."
    maxtokens = st.sidebar.slider("Maximum Tokens:", min_value=100, max_value=5000, value=2000, step=100, help=tip4)
    return model_name, temp, topp, maxtokens

def handle_pdf_files(uploaded_files, model_name, temperature, top_p, max_tokens):
    path_to_files = './data/'
    if not os.path.exists(path_to_files):
        os.makedirs(path_to_files)
    
    merger = PdfMerger()
    for file in uploaded_files:
        file_name = file.name
        file_path = os.path.join(path_to_files, file_name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        merger.append(file_path)

    merged_file = os.path.join(path_to_files, "merged_all_pages.pdf")
    merger.write(merged_file)
    merger.close()
    
    with open(merged_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    file_1 = Part.from_data(
        mime_type="application/pdf",
        data=base64.b64decode(encoded_string)
    )

    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_tokens,
    }
    try:
        gen_model = GenerativeModel(model_name=model_name, generation_config=generation_config)
        st.write(f"Total tokens submitted: {gen_model.count_tokens(file_1)}")
        question = st.text_input("Enter your question and hit return.")
        if question:
            response = gen_model.generate_content([question, file_1])
            st.markdown(response.text)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def handle_image_files(image_file, model_name, temperature, top_p, max_tokens):
    if image_file:
        path_to_files = './media/'
        if not os.path.exists(path_to_files):
            os.makedirs(path_to_files)
        
        file_path = os.path.join(path_to_files, image_file.name)
        with open(file_path, "wb") as f:
            f.write(image_file.read())
        
        try:
            image_file = genai.upload_file(path=file_path)
            while image_file.state.name == "PROCESSING":
                time.sleep(10)
                image_file = genai.get_file(image_file.name)
            if image_file.state.name == "FAILED":
                raise ValueError(image_file.state.name)
            
            prompt2 = st.text_input("Enter your prompt.")
            if prompt2:
                generation_config = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_tokens,
                }
                model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
                response = model.generate_content([image_file, prompt2], request_options={"timeout": 600})
                st.markdown(response.text)
                genai.delete_file(image_file.name)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def handle_video_files(video_file, model_name):
    if video_file:
        path_to_files = './media/'
        if not os.path.exists(path_to_files):
            os.makedirs(path_to_files)
        
        file_path = os.path.join(path_to_files, video_file.name)
        with open(file_path, "wb") as f:
            f.write(video_file.read())
        
        try:
            video_file = genai.upload_file(path=file_path)
            while video_file.state.name == "PROCESSING":
                time.sleep(10)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)
            
            prompt3 = st.text_input("Enter your prompt.")
            if prompt3:
                model = genai.GenerativeModel(model_name=model_name)
                st.write("Making LLM inference request...")
                response = model.generate_content([video_file, prompt3], request_options={"timeout": 600})
                st.markdown(response.text)
                genai.delete_file(video_file.name)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def handle_audio_files(audio_file, model_name):
    if audio_file:
        path_to_files = './media/'
        if not os.path.exists(path_to_files):
            os.makedirs(path_to_files)
        
        file_path = os.path.join(path_to_files, audio_file.name)
        with open(file_path, "wb") as f:
            f.write(audio_file.read())
        
        try:
            audio_file = genai.upload_file(path=file_path)
            while audio_file.state.name == "PROCESSING":
                time.sleep(10)
                audio_file = genai.get_file(audio_file.name)
            if audio_file.state.name == "FAILED":
                raise ValueError(audio_file.state.name)
            
            prompt3 = st.text_input("Enter your prompt.")
            if prompt3:
                model = genai.GenerativeModel(model_name=model_name)
                response = model.generate_content([audio_file, prompt3], request_options={"timeout": 600})
                st.markdown(response.text)
                genai.delete_file(audio_file.name)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def main():
    page_setup()
    media_type = get_typeofmedia()
    model_name, temperature, top_p, max_tokens = get_llminfo()
    
    if media_type == "PDF files":
        uploaded_files = st.file_uploader("Choose 1 or more files", accept_multiple_files=True)
        if uploaded_files:
            handle_pdf_files(uploaded_files, model_name, temperature, top_p, max_tokens)
    
    elif media_type == "Images":
        image_file = st.file_uploader("Upload your image file.")
        if image_file:
            handle_image_files(image_file, model_name, temperature, top_p, max_tokens)
    
    elif media_type == "Video, mp4 file":
        video_file = st.file_uploader("Upload your video")
        if video_file:
            handle_video_files(video_file, model_name)
    
    elif media_type == "Audio files":
        audio_file = st.file_uploader("Upload your audio")
        if audio_file:
            handle_audio_files(audio_file, model_name)

if __name__ == '__main__':
    load_dotenv()  # Load environment variables from .env file
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        st.error("GOOGLE_API_KEY is not set. Please check your .env file.")
    
    main()
