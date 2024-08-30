import streamlit as st
import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from pypdf import PdfMerger
from dotenv import load_dotenv
import os
import time
import base64

def page_setup():
    st.header("Chat with different types of media/files!", anchor=False, divider="blue")
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def get_type_of_media():
    st.sidebar.header("Select type of Media", divider='orange')
    media_type = st.sidebar.radio("Choose one:",
                                  ("PDF files", "Images", "Video, mp4 file", "Audio files"))
    return media_type

def get_llm_info():
    st.sidebar.header("Options", divider='rainbow')
    model = st.sidebar.radio("Choose LLM:",
                             ("gemini-1.5-flash", "gemini-1.5-pro"),
                             help="Select a model you want to use.")
    temp = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=1.0, step=0.25,
                             help="Lower temperatures are less creative; higher temperatures are more creative.")
    topp = st.sidebar.slider("Top P:", min_value=0.0, max_value=1.0, value=0.94, step=0.01,
                             help="Lower values make responses less random; higher values make responses more random.")
    max_tokens = st.sidebar.slider("Maximum Tokens:", min_value=100, max_value=5000, value=2000, step=100,
                                   help="Number of response tokens, 8194 is limit.")
    return model, temp, topp, max_tokens

def handle_pdf_files(uploaded_files, model, temperature, top_p, max_tokens):
    path_to_files = './data'  # Adjust the path if needed
    merger = PdfMerger()

    for file in uploaded_files:
        file_name = file.name
        with open(os.path.join(path_to_files, file_name), "wb") as f:
            f.write(file.getbuffer())
        merger.append(os.path.join(path_to_files, file_name))

    merged_file = os.path.join(path_to_files, "merged_all_pages.pdf")
    merger.write(merged_file)
    merger.close()

    st.write(f"Merged PDF file: {merged_file}")

    with open(merged_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()

    file_1 = Part.from_data(mime_type="application/pdf", data=base64.b64decode(encoded_string))

    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_tokens,
    }

    gen_model = GenerativeModel(model_name=model, generation_config=generation_config)
    st.write(f"Total tokens submitted: {gen_model.count_tokens(file_1)}")
    question = st.text_input("Enter your question and hit return.")
    if question:
        response = gen_model.generate_content([question, file_1])
        st.markdown(response.text)

def handle_media_files(media_type, model, temperature, top_p, max_tokens):
    file = st.file_uploader(f"Upload your {media_type.lower()} file")
    if file:
        path = './media'  # Adjust the path if needed
        file_path = os.path.join(path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        uploaded_file = genai.upload_file(path=file_path)
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(10)
            uploaded_file = genai.get_file(uploaded_file.name)
        if uploaded_file.state.name == "FAILED":
            st.error("File processing failed.")
            return

        prompt = st.text_input("Enter your prompt.")
        if prompt:
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_tokens,
            }
            gen_model = genai.GenerativeModel(model_name=model, generation_config=generation_config)
            response = gen_model.generate_content([uploaded_file, prompt], request_options={"timeout": 600})
            st.markdown(response.text)

            genai.delete_file(uploaded_file.name)

def main():
    page_setup()
    media_type = get_type_of_media()
    model, temperature, top_p, max_tokens = get_llm_info()

    # Ensure vertexai.init() is called with the correct project_id
    vertexai.init(project="real-time-rag", location="us-central1")

    if media_type == "PDF files":
        uploaded_files = st.file_uploader("Choose 1 or more files", accept_multiple_files=True)
        if uploaded_files:
            handle_pdf_files(uploaded_files, model, temperature, top_p, max_tokens)
    elif media_type in ["Images", "Video, mp4 file", "Audio files"]:
        handle_media_files(media_type, model, temperature, top_p, max_tokens)

if __name__ == '__main__':
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        st.error("Google API key not found. Please set the 'GOOGLE_API_KEY' environment variable.")
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        main()
