import os
import logging

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


# ---------------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------------
logging.basicConfig(
    filename="app.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s — [%(levelname)s] — %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console)


# ---------------------------------------------------------
# BUILD VECTOR DATABASE
# ---------------------------------------------------------
def build_vector_store():
    """Create Chroma vector store from speech.txt on first run."""
    logging.info("Starting first-time vector store creation.")

    try:
        logging.info("Loading speech.txt...")
        loader = TextLoader("speech.txt")
        documents = loader.load()
    except Exception as e:
        logging.error(f"Failed to load speech.txt — {e}")
        raise e

    logging.info("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    logging.info(f"Total chunks created: {len(chunks)}")

    logging.info("Creating embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    logging.info("Storing embeddings in ChromaDB...")
    try:
        vectorstore = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory="db"
        )
        vectorstore.persist()
        logging.info("Vector store successfully created & persisted.")
    except Exception as e:
        logging.error(f"Error creating vector store — {e}")
        raise e

    return vectorstore


# ---------------------------------------------------------
# LOAD EXISTING VECTOR DATABASE
# ---------------------------------------------------------
def load_vector_store():
    """Load an existing Chroma vector store."""
    logging.info("Loading existing Chroma vector database.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        vectorstore = Chroma(
            persist_directory="db",
            embedding_function=embeddings
        )
        logging.info("Vector store loaded successfully.")
        return vectorstore

    except Exception as e:
        logging.error(f"Error loading vector store — {e}")
        raise e


# ---------------------------------------------------------
# SELECT DATABASE (LOAD OR BUILD)
# ---------------------------------------------------------
def get_vector_store():
    """Check if db exists, otherwise create it."""
    if os.path.exists("db"):
        logging.info("VectorDB folder found — loading existing DB.")
        return load_vector_store()
    else:
        logging.info("VectorDB not found — creating a new one.")
        return build_vector_store()


# ---------------------------------------------------------
# MAIN PROGRAM LOOP
# ---------------------------------------------------------
def main():
    logging.info("AmbedkarGPT initialized.")
    print("\nAmbedkarGPT — Command-Line Q&A System")
    print("Ask questions based on the 'Annihilation of Caste' speech.\n")

    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever()

    # Initialize LLM (Ollama Mistral)
    logging.info("Loading LLM model: mistral (via Ollama)")
    llm = Ollama(model="mistral")

    # Build RetrievalQA chain
    logging.info("Setting up RetrievalQA chain.")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # CLI loop
    while True:
        query = input("Your question (or type 'exit'): ")

        if query.lower() == "exit":
            logging.info("User requested exit. Shutting down.")
            print("Exiting... Goodbye.")
            break

        logging.info(f"User query: {query}")

        try:
            answer = qa.run(query)
            logging.info("Answer generated successfully.")
        except Exception as e:
            logging.error(f"Error generating answer — {e}")
            answer = "Sorry, something went wrong while generating the answer."

        print("\nAnswer:")
        print(answer)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
