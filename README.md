# AmbedkarGPT - Command-Line Q&A System

A RAG (Retrieval-Augmented Generation) based question-answering system built with LangChain that answers questions based on Dr. B.R. Ambedkar's "Annihilation of Caste" speech excerpt.

## 📋 Overview

This project demonstrates a functional RAG pipeline that:
- Loads text from a local file (`speech.txt`)
- Splits text into semantic chunks
- Creates embeddings using HuggingFace's sentence transformers
- Stores embeddings in a local ChromaDB vector database
- Retrieves relevant context based on user queries
- Generates answers using Ollama's Mistral 7B model

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **Framework**: LangChain
- **Vector Database**: ChromaDB (local, persistent)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Ollama with Mistral 7B (local, no API keys required)

## 📁 Project Structure

```
AmbedkarGPT-Intern-Task/
├── main.py              # Main application code
├── speech.txt           # Source text (Ambedkar's speech excerpt)
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── app.log             # Application logs (generated at runtime)
└── db/                 # ChromaDB vector store (created on first run)
```

## 🚀 Setup Instructions

### Prerequisites

1. **Python 3.8 or higher**
   ```bash
   python --version
   ```

2. **Ollama with Mistral 7B**
   
   Install Ollama:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```
   
   Pull the Mistral model:
   ```bash
   ollama pull mistral
   ```
   
   Verify installation:
   ```bash
   ollama list
   ```

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AmbedkarGPT-Intern-Task.git
   cd AmbedkarGPT-Intern-Task
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify speech.txt exists**
   
   Ensure `speech.txt` is present in the project root with the provided text content.

## 💻 Usage

### Running the Application

Simply run:
```bash
python main.py
```

### First Run

On the first run, the application will:
1. Load and process `speech.txt`
2. Create text chunks
3. Generate embeddings
4. Build and persist the ChromaDB vector database in the `db/` folder

This process takes a few minutes depending on your system.

### Subsequent Runs

The application will detect the existing `db/` folder and load the vector store instantly.

### Example Interaction

```
AmbedkarGPT — Command-Line Q&A System
Ask questions based on the 'Annihilation of Caste' speech.

Your question (or type 'exit'): What is the real remedy according to the text?

Answer:
According to the text, the real remedy is to destroy the belief in the sanctity 
of the shastras. The problem of caste cannot be solved through social reform alone; 
it requires overthrowing the authority of the shastras.

--------------------------------------------------

Your question (or type 'exit'): exit
Exiting... Goodbye.
```

## 🔍 How It Works

### RAG Pipeline Flow

1. **Document Loading**: `TextLoader` reads `speech.txt`
2. **Text Splitting**: `RecursiveCharacterTextSplitter` creates 300-character chunks with 50-character overlap
3. **Embedding Creation**: HuggingFace's MiniLM model converts chunks to vector embeddings
4. **Vector Storage**: ChromaDB stores embeddings locally for fast retrieval
5. **Query Processing**: User questions are embedded and matched against stored vectors
6. **Context Retrieval**: Most relevant chunks are retrieved
7. **Answer Generation**: Ollama's Mistral 7B generates contextual answers

### Key Features

- **Persistent Storage**: Vector database is created once and reused
- **Logging**: All operations logged to `app.log` and console
- **Error Handling**: Graceful error management with informative messages
- **Interactive CLI**: Simple command-line interface for continuous querying

## 📝 Dependencies

See `requirements.txt` for the complete list. Key packages include:

- `langchain` - RAG orchestration framework
- `langchain-community` - Community integrations
- `chromadb` - Vector database
- `sentence-transformers` - Embedding models
- `ollama` - Local LLM integration

## 🐛 Troubleshooting

### Common Issues

**"Ollama model not found"**
- Ensure Ollama is installed and running
- Pull the Mistral model: `ollama pull mistral`

**"speech.txt not found"**
- Verify the file exists in the project root directory
- Check file permissions

**"Module not found" errors**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

**Slow first-time setup**
- Embedding generation and vector store creation take time
- Subsequent runs will be much faster

## 📊 Logs

The application maintains detailed logs in `app.log` including:
- Vector store creation/loading status
- Query processing
- Error messages
- Performance metrics

View logs:
```bash
cat app.log
```

## 🔄 Resetting the Database

To rebuild the vector store from scratch:
```bash
rm -rf db/
python main.py
```
## 📚 Author

Name: Anand Vishwakarma
Email: anandvishwakarma21j@gmail.com

## 📚 Assignment Context

This project was created as part of the AI Intern hiring assignment for Kalpit Pvt Ltd, UK. It demonstrates:
- Understanding of RAG architecture
- LangChain framework proficiency
- Vector database implementation
- Local LLM integration
- Clean, documented code practices

## 📄 License

This project is created for educational and evaluation purposes as part of an internship assignment.

---
