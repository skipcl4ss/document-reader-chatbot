import os
import streamlit as st
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# Read the API key from a file
with open("C:\\Programming\\doc reader chatbot\\groq.txt", "r") as apiKeyfile:
    apiKey = apiKeyfile.read().strip()
# Set the API key as an environment variable
os.environ["GROQ_API_KEY"] = apiKey

def main():
    st.title("Welcome To Document Assistant")
    st.subheader("How can I help you?")

    option = st.selectbox(
        "What is the extension of the file in question?",
        ("pdf", "md", "txt"),
    )
    file = st.file_uploader("Upload your file", type = option)

    if file is not None:
        text = ""
        match option:
            case "pdf":
                with st.spinner("Processing file..."):
                    pdf = PdfReader(file)
                    for page in pdf.pages:
                        text += page.extract_text()

            case "txt" | "md":
                text = file.getvalue().decode("utf-8")

        query = st.text_input("Ask the AI a question about your file", placeholder = "What is the premise of the story?")
        confirm = st.button("Confirm")

        if confirm:
            cancel = st.button('Cancel')
            if cancel:
                st.stop()
            if query and not cancel:
                # db = processText(text)
                with st.spinner("Understanding the file content, this may take a few minutes..."):
                    db = processText(text)
                    results = db.similarity_search_with_score(query)
                    response = analyzeResults(query, results)
                st.header("AI response")
                st.markdown(response.content)

def processText(text):
    # Split the text into chunks using Langchain's RecursiveCharacterTextSplitter
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024,
        chunk_overlap = 50,
    )
    chunks = textSplitter.split_text(text)

    # Convert the chunks of text into embeddings to form a database
    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_texts(chunks, embeddings)

    return db

# TODO: Understand this function fully
def analyzeResults(query, results):
    prompt_template = """
    Answer the query based only on the given context:
    {context}
    ---
    Answer the query based on the above context: {query}
    """

    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt = prompt_template.format(context = context, query = query)

    llm = ChatGroq()
    response = llm.invoke(prompt)

    return response

main()
