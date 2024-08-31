import os
import base64
import requests
import streamlit as st
from PIL import Image
from io import BytesIO
from pypdf import PdfMerger
from dotenv import load_dotenv  # Import the load_dotenv function

# Load environment variables from .env file
load_dotenv()

# Hugging Face API token
HF_API_TOKEN = os.getenv('HF_API_TOKEN')  # Make sure to set this environment variable

# Hugging Face API endpoints
ENDPOINTS = {
    'text-generation': 'https://api-inference.huggingface.co/models/gpt2',
    'summarization': 'https://api-inference.huggingface.co/models/facebook/bart-large-cnn',
    'image-classification': 'https://api-inference.huggingface.co/models/google/vit-base-patch16-224-in21k',
    'audio-transcription': 'https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h'
}

# Helper function to call Hugging Face Inference API
def call_hf_api(endpoint, data, headers):
    response = requests.post(endpoint, headers=headers, files=data)
    return response.json()

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
    
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=1.0, step=0.25)
    topp = st.sidebar.slider("Top P:", min_value=0.0, max_value=1.0, value=0.94, step=0.01)
    maxtokens = st.sidebar.slider("Maximum Tokens:", min_value=100, max_value=5000, value=2000, step=100)
    
    return temperature, topp, maxtokens

# Function to handle PDF files
def handle_pdf_files(uploaded_files, temperature, top_p, max_tokens):
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

    st.write(f"Merged PDF file created at: {merged_file}")
    
    # Optionally display the file or provide a download link
    with open(merged_file, "rb") as pdf_file:
        st.download_button(
            label="Download Merged PDF",
            data=pdf_file,
            file_name="merged_all_pages.pdf",
            mime="application/pdf"
        )
    
    question = st.text_input("Enter your question and hit return.", help="Ask any question about the content of the PDF file.")
    if question:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        response = requests.post(
            ENDPOINTS['text-generation'],
            headers=headers,
            json={"inputs": question}
        )
        if response.status_code == 200:
            result = response.json()
            st.markdown(result[0]['generated_text'])
        else:
            st.error("Error with API request. Please check the API and your request.")


# Function to handle image files
def handle_image_files(image_file, temperature, top_p, max_tokens):
    st.subheader("🖼️ Image File Processing")
    if image_file:
        image = Image.open(image_file)
        
        # Check if image has an alpha channel and convert to RGB if necessary
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Save image to a buffer in JPEG format
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        response = call_hf_api(
            ENDPOINTS['image-classification'],
            data={"file": buffer},
            headers=headers
        )
        
        st.write("Image Classification Results:")
        for label in response:
            st.write(f"Label: {label['label']}, Confidence: {label['score']:.2f}")


# Function to handle video files
def handle_video_files(video_file):
    st.subheader("🎥 Video File Processing")
    st.write("Video file processing is not implemented in this lightweight example.")

# Function to handle audio files
def handle_audio_files(audio_file):
    st.subheader("🎵 Audio File Processing")
    if audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
        
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        response = requests.post(
            ENDPOINTS['audio-transcription'],
            headers=headers,
            files={"file": audio_bytes}
        )
        result = response.json()
        st.write("Audio Transcription:")
        st.write(result['text'])

# Main function to run the app
def main():
    page_setup()
    media_type = get_typeofmedia()
    temperature, top_p, max_tokens = get_llminfo()
    
    if media_type == "PDF files":
        uploaded_files = st.file_uploader("Choose 1 or more files", accept_multiple_files=True)
        if uploaded_files:
            handle_pdf_files(uploaded_files, temperature, top_p, max_tokens)
    
    elif media_type == "Images":
        image_file = st.file_uploader("Upload your image file.", type=["jpg", "png", "jpeg"])
        if image_file:
            handle_image_files(image_file, temperature, top_p, max_tokens)
    
    elif media_type == "Video, mp4 file":
        st.write("Video file processing is not implemented in this lightweight example.")
    
    elif media_type == "Audio files":
        audio_file = st.file_uploader("Upload your audio", type=["wav", "mp3"])
        if audio_file:
            handle_audio_files(audio_file)

if __name__ == '__main__':
    main()
