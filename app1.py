import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configure Google Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split text into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Create and save vector embeddings"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save without the parameter
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """Set up the conversational chain with Gemini"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


def process_user_input(user_question):
    """Process user question and generate response"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Load with allow_dangerous_deserialization
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        st.write("Reply: ", response["output_text"])
    except FileNotFoundError:
        st.error("Please process some PDF documents first before asking questions.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def main():
    # Page configuration
    st.set_page_config(
        page_title="Chat PDF",
        page_icon="📚",
        layout="wide"
    )

    st.header("Chat with PDF using Gemini 💭")

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("📄 Document Upload")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on Process",
            accept_multiple_files=True,
            type=['pdf']
        )

        if st.button("Process PDFs", type="primary"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return

            with st.spinner("Processing Documents..."):
                try:
                    # Process PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Documents processed successfully!")
                    st.session_state.docs_processed = True
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    st.session_state.docs_processed = False

    # Main chat interface
    user_question = st.text_input(
        "Ask a question about your documents:",
        placeholder="What would you like to know about the uploaded PDFs?"
    )

    if user_question:
        if not 'docs_processed' in st.session_state or not st.session_state.docs_processed:
            st.warning("Please upload and process PDF documents before asking questions.")
            return
        process_user_input(user_question)


if __name__ == "__main__":
    main()