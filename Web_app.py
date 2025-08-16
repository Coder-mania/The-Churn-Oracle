import os
import warnings
import logging
import streamlit as st
import mimetypes
import base64
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Disable warnings and lower transformers logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="Ask Chatbot", layout="wide")

@st.cache_resource
def get_vectorstore():
    pdf_name = "FIle_upload"
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore

def chatbot_interface():
    st.title("🗣️ Ask Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all previous messages in chat
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Type your question here...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        system_prompt = ChatPromptTemplate.from_template(
            "You are very smart at everything, you always give the best, "
            "the most accurate and most precise answers. Answer the following Question: {user_prompt}. "
            "Start the answer directly. No small talk please."
        )

        model = "llama-3.1-8b-instant"
        os.environ[
            "GROQ_API_KEY"
        ] = "API_KEY"

        groq_chat = ChatGroq(
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            model_name=model,
        )

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load document")
                return

            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
            )

            result = chain({"query": prompt})
            response = result["result"]
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {str(e)}")


def get_base64_image(image_path):
    # Guess the MIME type (e.g., 'image/png')
    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None:
        st.error("Could not determine the image MIME type.")
        return ""

    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:{mime_type};base64,{encoded}"

def home_page():
    local_image_path = "img2.png"  # Change to your image file (.png/.jpg/.gif)

    if not os.path.exists(local_image_path):
        st.error(f"Image not found: {local_image_path}")
        return

    background_base64 = get_base64_image(local_image_path)

    # Inject global background CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{background_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .content-box {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
            max-width: 600px;
            margin: auto;
            font-family: 'Times New Roman', Times, serif;
            text-align: justify;
        }}
        div.stButton {{
            display: flex;
            justify-content: center;
            margin-top: -15px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Text content inside a semi-transparent box
    st.markdown(
        """
        <div class="content-box">
            <h1 style="color: #4A90E2; text-align: center;">Welcome<br>to<br>The Churn Oracle</h1>
            <p style='font-size:20px;'>About: <br> AI-Enhanced Telecom Customer Retention with Predictive Modeling and Document Intelligence. </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Start Chatting"):
        st.session_state.chat_started = True



# Initialize chat_started flag in session state
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

if st.session_state.chat_started:
    chatbot_interface()
else:
    home_page()

