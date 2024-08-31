import os
import time
import base64
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from pypdf import PdfMerger
from google.auth import load_credentials_from_file

# Path to your credentials file
CREDENTIALS_PATH = 'cred.json'

# Load credentials from the file
credentials, project = load_credentials_from_file(CREDENTIALS_PATH)

# Initialize Google Cloud Vertex AI with the credentials
vertexai.init(project=project, location='us-central1', credentials=credentials)

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #fff1f2; /* rose-50 */
        color: #333;
        font-family: 'Arial', sans-serif;
        padding: 20px;
    }
    .stButton>button {
        background-color: #f43f5e; /* rose-500 */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #e11d48; /* rose-600 */
    }
    .stHeader {
        color: #be123c; /* rose-700 */
        font-weight: 700;
        border-bottom: 2px solid #f43f5e; /* rose-500 */
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .dark-header {
        color: #9f1239; /* rose-800 */
    }
    .stSidebar .sidebar-content {
        background-color: #ffe4e6; /* rose-100 */
        padding: 20px;
        border-radius: 10px;
    }
    .stFileUploader {
        background-color: #fda4af; /* rose-300 */
        border: 2px dashed #f43f5e; /* rose-500 */
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    h2, h3 {
        color: #e11d48; /* rose-600 */
    }
    .chat-bubble {
        background-color: #fb7185; /* rose-400 */
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #fda4af; /* rose-300 */
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

# Function to set up the page with a dark header
def page_setup():
    st.markdown("<h1 class='dark-header'>🧠 Universal Chatbot for Media File Processing</h1>", unsafe_allow_html=True)
    st.markdown("This application allows you to upload and process various media files using advanced AI models. Please select the media type from the sidebar to get started.")
    
# Sidebar for selecting the type of media
def get_typeofmedia():
    st.sidebar.header("Select Type of Media")
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
    st.subheader("📄 PDF File Processing")
    st.write("You can upload multiple PDF files. The files will be merged and processed together.")
    
    path_to_files = './data/'
    if not os.path.exists(path_to_files):
        os.makedirs(path_to_files)

    # Create a PdfMerger object
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
    st.subheader("🖼️ Image File Processing")
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
    st.subheader("🎥 Video File Processing")
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
    st.subheader("🎵 Audio File Processing")
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
    load_dotenv()  # Load environment variables from .env file
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        st.error("GOOGLE_API_KEY is not set. Please check your .env file.")
    
    main()
