import os
import time
import base64
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2.service_account import Credentials
from PyPDF2 import PdfMerger  # Updated import for PdfMerger

# Load credentials from Streamlit secrets
def load_credentials():
    credentials_info = {
        'type': st.secrets["google"]["type"],
        'project_id': st.secrets["google"]["project_id"],
        'private_key_id': st.secrets["google"]["private_key_id"],
        'private_key': st.secrets["google"]["private_key"].replace('\\n', '\n'),
        'client_email': st.secrets["google"]["client_email"],
        'client_id': st.secrets["google"]["client_id"],
        'auth_uri': st.secrets["google"]["auth_uri"],
        'token_uri': st.secrets["google"]["token_uri"],
        'auth_provider_x509_cert_url': st.secrets["google"]["auth_provider_x509_cert_url"],
        'client_x509_cert_url': st.secrets["google"]["client_x509_cert_url"],
    }
    credentials = Credentials.from_service_account_info(credentials_info)
    return credentials, credentials_info['project_id']

# Initialize Google Cloud Vertex AI with the credentials
credentials, project = load_credentials()
vertexai.init(project=project, location='us-central1', credentials=credentials)

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #fff1f2;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #f43f5e;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #e11d48;
        color: white;
    }
    .stHeader {
        color: #e11d48;
        font-weight: 700;
    }
    .dark-header {
        color: #e11d48; /* Dark color for the header */
    }
    .stSidebar .sidebar-content {
        background-color: #f7f7f7;
    }
    .chat-bubble {
        background-color: #f43f5e;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #e11d48;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
    }
    .stFileUploader>div>div>div>input {
        background-color: #fda4af;
        color: #333;
    }
    .stSubheader {
        color: #e11d48; /* Match subheader color with header */
        font-weight: 600;
        font-size: 1.5em;
    }
    </style>
""", unsafe_allow_html=True)

# Function to set up the page with a dark header
def page_setup():
    st.markdown("<h1 class='dark-header'>🧠 Universal Chatbot for Media File Processing</h1>", unsafe_allow_html=True)
    st.markdown("This application allows you to upload and process various media files using advanced AI models. Please select the media type from the sidebar to get started.")

# Sidebar for selecting the type of media
def get_typeofmedia():
    st.sidebar.header("Select type of Media")
    media_type = st.sidebar.radio("Choose one:",
                                  ("PDF files", "Images", "Video, mp4 file", "Audio files"))
    return media_type

# Sidebar for LLM configuration options
def get_llminfo():
    st.sidebar.header("LLM Configuration")
    st.sidebar.markdown("Configure the language model settings to customize the AI's responses.")
    
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("gemini-1.5-flash", "gemini-1.5-pro"))
    temp = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=1.0, step=0.25)
    topp = st.sidebar.slider("Top P:", min_value=0.0, max_value=1.0, value=0.94, step=0.01)
    maxtokens = st.sidebar.slider("Maximum Tokens:", min_value=100, max_value=5000, value=2000, step=100)
    
    return model_name, temp, topp, maxtokens

# Function to handle PDF files
def handle_pdf_files(uploaded_files, model_name, temperature, top_p, max_tokens):
    st.markdown("<h2 class='stSubheader'>📄 PDF File Processing</h2>", unsafe_allow_html=True)
    st.write("You can upload multiple PDF files. The files will be merged and processed together.")

    # Create a PdfMerger object
    merger = PdfMerger()

    # Save and merge uploaded files
    for file in uploaded_files:
        file_name = file.name
        file_path = f'/tmp/{file_name}'
        with open(file_path, "wb") as f:
            f.write(file.read())
        merger.append(file_path)

    merged_file = '/tmp/merged_all_pages.pdf'
    merger.write(merged_file)
    merger.close()

    with open(merged_file, "rb") as pdf_file:
        encoded_string = base64.b64encode(pdf_file.read()).decode()

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
        question = st.text_input("Enter your question and hit return.", help="Ask any question about the content of the PDF file.")
        if question:
            response = gen_model.generate_content([question, file_1])
            st.markdown(response.text)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Function to handle image files
def handle_image_files(image_file, model_name, temperature, top_p, max_tokens):
    st.markdown("<h2 class='stSubheader'>🖼️ Image File Processing</h2>", unsafe_allow_html=True)
    if image_file:
        file_path = f'/tmp/{image_file.name}'
        with open(file_path, "wb") as f:
            f.write(image_file.read())
        
        try:
            image_file = genai.upload_file(path=file_path)
            while image_file.state.name == "PROCESSING":
                time.sleep(10)
                image_file = genai.get_file(image_file.name)
            if image_file.state.name == "FAILED":
                raise ValueError(image_file.state.name)
            
            prompt2 = st.text_input("Enter your prompt.", help="Describe what you want the AI to do with the image.")
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

# Function to handle video files
def handle_video_files(video_file, model_name):
    st.markdown("<h2 class='stSubheader'>🎥 Video File Processing</h2>", unsafe_allow_html=True)
    if video_file:
        file_path = f'/tmp/{video_file.name}'
        with open(file_path, "wb") as f:
            f.write(video_file.read())
        
        try:
            video_file = genai.upload_file(path=file_path)
            while video_file.state.name == "PROCESSING":
                time.sleep(10)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)
            
            prompt3 = st.text_input("Enter your prompt.", help="Describe what you want the AI to do with the video.")
            if prompt3:
                model = genai.GenerativeModel(model_name=model_name)
                st.write("Making LLM inference request...")
                response = model.generate_content([video_file, prompt3], request_options={"timeout": 600})
                st.markdown(response.text)
                genai.delete_file(video_file.name)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Function to handle audio files
def handle_audio_files(audio_file, model_name):
    st.markdown("<h2 class='stSubheader'>🎵 Audio File Processing</h2>", unsafe_allow_html=True)
    if audio_file:
        file_path = f'/tmp/{audio_file.name}'
        with open(file_path, "wb") as f:
            f.write(audio_file.read())
        
        try:
            audio_file = genai.upload_file(path=file_path)
            while audio_file.state.name == "PROCESSING":
                time.sleep(10)
                audio_file = genai.get_file(audio_file.name)
            if audio_file.state.name == "FAILED":
                raise ValueError(audio_file.state.name)
            
            prompt3 = st.text_input("Enter your prompt.", help="Describe what you want the AI to do with the audio.")
            if prompt3:
                model = genai.GenerativeModel(model_name=model_name)
                response = model.generate_content([audio_file, prompt3], request_options={"timeout": 600})
                st.markdown(response.text)
                genai.delete_file(audio_file.name)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Main function to run the app
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
    main()
