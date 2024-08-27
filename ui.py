import streamlit as st
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from pprint import pprint
from langchain_groq import ChatGroq
import os

# Read the API key from a file
with open("C:\Programming\doc reader chatbot\groq.txt", "r") as file:
    apiKey = file.read().strip()
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
        pdf = PdfReader(file)
        st.header("debug")
        st.text("debug")
        st.markdown("debug")
        st.write("debug")
        # st.markdown(pdf)
        # st.markdown(pdf.pages)
        # st.markdown(len(pdf.pages))
        # st.markdown(pdf.metadata)
        # st.markdown(type(pdf))
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

        db = processText(text)

        # query = st.text_input("Ask the AI a question about your file", "What is the premise of the story?")
        query = st.text_input("Ask the AI a question about your file", "What is the premise of the story?")
        confirm = st.button("Confirm")
        if confirm:
            cancel = st.button('Cancel')
            if cancel:
                st.stop()

            if query and not cancel:
                results = db.similarity_search_with_score(query)
                # print(results)
                # print()
                pprint(results)

                response = analyzeResults(query, results)
                st.markdown(response)
                st.write(response)


    # st.header("AI response")

    # response = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum ut imperdiet est. Curabitur sollicitudin, mi porttitor auctor pharetra, tortor ipsum viverra dolor, eu dictum odio magna eu felis. Donec faucibus imperdiet tortor in tristique. Curabitur auctor sapien vel tempor tristique. Suspendisse convallis est ut hendrerit volutpat. Mauris eget elit vitae mauris finibus finibus. Phasellus sed dolor eget lorem tincidunt ullamcorper vitae id lacus. Nunc ullamcorper lorem ac tortor auctor dictum. Morbi sit amet eros aliquam, finibus quam et, hendrerit nisl. Donec erat augue, venenatis sit amet vulputate at, tincidunt sed sem. Pellentesque imperdiet nisi pharetra erat molestie, quis dapibus velit pharetra. Curabitur sollicitudin nunc sed cursus consectetur. Nulla ultricies molestie nisi et posuere.\n\nSed eleifend nisl at eleifend accumsan. Mauris purus odio, dictum sit amet arcu quis, imperdiet ultrices nisl. Etiam egestas vel sapien id porta. Praesent at accumsan justo, sit amet sollicitudin ipsum. Nullam scelerisque dignissim nisl id scelerisque. Mauris posuere est sapien, vitae imperdiet orci tincidunt vitae. Nullam in est et metus imperdiet ultrices. Nunc augue nibh, porttitor elementum libero vel, rutrum hendrerit ex. Suspendisse elementum condimentum turpis, vitae imperdiet neque molestie eu. Sed sed sapien efficitur, mattis nunc eu, porta diam. Nullam non fermentum enim. Nam et velit augue. Curabitur egestas mi vitae porta vestibulum. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Vivamus dignissim sem quis augue pellentesque, in tincidunt magna porttitor."

    # st.markdown(f"{response}")

def processText(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    textSplitter = CharacterTextSplitter(
        # separator="\n",
        chunk_size = 1000,
        chunk_overlap = 50,
        # length_function=len
    )
    chunks = textSplitter.split_text(text)
    print(f"Split {len(text)} documents into {len(chunks)} chunks.")

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
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
    # llm = ChatGroq(model="llama3-8b-8192")
    # llm = ChatGroq(model = "llama-3.1-70b-versatile")
    response = llm.invoke(prompt)

    return response

# TODO: Understand this function fully
def calculateChunkIDs(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    lastPageID = None
    currentChunkIndex = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        currentPageID = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if currentPageID == lastPageID:
            currentChunkIndex += 1
        else:
            currentChunkIndex = 0

        # Calculate the chunk ID.
        chunkID = f"{currentPageID}:{current_chunk_index}"
        lastPageID = currentPageID

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunkID

    return chunks

main()