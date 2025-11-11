# RAG System Group Project

A hands-on group learning project for building a Retrieval-Augmented Generation (RAG) system using Python, Docker, and open-source AI models.

## Overview

This group project teaches students how to build a question-answering AI system that can:
- Read and process documents
- Find relevant information using semantic search
- Generate accurate answers using a local LLM

Students will work in teams to practice fundamental Python skills while working with cutting-edge AI technology.

## What You'll Learn

### Python Fundamentals
- File I/O and text processing
- String manipulation and chunking
- Functions and data structures
- Loops and conditionals
- Working with external libraries

### AI/ML Concepts
- Embeddings and vector search
- Retrieval-Augmented Generation (RAG)
- Language Model integration
- System evaluation and metrics

## Project Structure

```
IS640_rag_project/
├── README.md                      # This file
├── STUDENT_PROJECT_GUIDE.md       # Detailed assignment instructions
├── docker_starter.md              # Docker setup guide
├── dockerfile                     # Docker image definition for Ollama
├── COMMANDS.txt                   # Docker commands reference
├── test_setup.ipynb              # Environment verification notebook
├── rag_env.yml                    # Conda environment file
├── .gitignore                     # Git ignore file
├── docs/                          # Sample documents
│   ├── text/                     # Text files for testing
│   │   ├── Attendance_Rules.txt
│   │   ├── Cheating_Rules.txt
│   │   └── School_Parking_Rules.txt
│   └── pdf/                      # (Optional) PDF documents
└── chroma_db/                     # Vector database storage (created on first run)
```

## Quick Start

### Prerequisites

- Python 3.12 or higher
- Anaconda or Miniconda
- Docker Desktop
- VS Code (recommended)
- 8GB RAM minimum (16GB recommended)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd IS640_rag_project
```

### Step 2: Set Up Docker

Follow the instructions in [`docker_starter.md`](docker_starter.md) to:
1. Install Docker Desktop
2. Create a Docker account
3. Set up the Docker extension for VS Code

### Step 3: Create Conda Environment

```bash
# Create environment from file
conda env create -f rag_env.yml

# Activate environment
conda activate rag3_313
```

### Step 4: Build and Start the Ollama LLM

Build the Docker image using the included `dockerfile`:

```bash
# Build the Docker image (this will take 10-15 minutes)
docker build -f dockerfile -t ollama-mistral-img .

# Start the container (CPU only)
docker run -d --rm --name ollama-mistral-offline -p 127.0.0.1:11434:11434 ollama-mistral-img

# Or with GPU support (if available)
docker run -d --rm --gpus all --name ollama-mistral-offline -p 127.0.0.1:11434:11434 ollama-mistral-img
```

For more Docker commands, see [`COMMANDS.txt`](COMMANDS.txt).

### Step 5: Verify Your Setup

Open and run `test_setup.ipynb` to verify everything is working correctly. This notebook will test:
- Python dependencies
- Ollama LLM connection
- Embedding model
- ChromaDB vector database

### Step 6: Start the Assignment

Read [`STUDENT_PROJECT_GUIDE.md`](STUDENT_PROJECT_GUIDE.md) for complete project instructions.

## Requirements

### Python Packages

All required packages are included in `rag_env.yml`:
- `sentence-transformers` - Text embeddings
- `chromadb` - Vector database
- `requests` - HTTP client for LLM
- `numpy`, `pandas` - Data processing

### Hardware

- **Minimum:** 8GB RAM, 4-core CPU
- **Recommended:** 16GB RAM, 8-core CPU, GPU (optional)
- **Storage:** 5GB free space

## Docker Setup

The project includes a `dockerfile` that builds an Ollama container with the Mistral 7B model. This provides a local LLM with:
- No API keys required
- Complete privacy (runs locally)
- Offline operation (once built)
- Free usage

The dockerfile uses a multi-stage build to:
1. Download and install Ollama
2. Pull the Mistral 7B model
3. Create a minimal runtime image

See [`docker_starter.md`](docker_starter.md) for detailed Docker Desktop setup and [`COMMANDS.txt`](COMMANDS.txt) for build/run commands.

## Group Project Assignment

This is a **group project** where teams will complete 7 TODO tasks:

1. **Document Loading** - Load text files from a folder
2. **Text Chunking** - Split documents into smaller pieces
3. **Process Documents** - Apply chunking to all documents
4. **RAG Query Function** - Build the main Q&A pipeline
5. **Test Dataset** - Create evaluation questions
6. **Evaluation Metrics** - Measure system performance
7. **Run Evaluation** - Test and analyze results

**Group Size:** 2-4 students per team

See [`STUDENT_PROJECT_GUIDE.md`](STUDENT_PROJECT_GUIDE.md) for complete details, group responsibilities, grading rubric, and submission requirements.

## Sample Documents

The `docs/text/` folder contains sample school policy documents for testing. Teams should:
1. Test with the provided documents first
2. Create their own document collection (5-10 files)
3. Choose topics the team understands well

## Troubleshooting

### Environment Issues

**Problem:** Conda environment won't create
```bash
# Try creating manually
conda create -n rag3_313 python=3.13
conda activate rag3_313
pip install sentence-transformers chromadb requests
```

### Docker Issues

**Problem:** Docker container won't start
- Verify Docker Desktop is running
- Check WSL 2 is installed (Windows)
- Enable virtualization in BIOS

**Problem:** Can't connect to Ollama
```bash
# Check if container is running
docker ps

# Test connection
curl http://127.0.0.1:11434/api/tags
```

### Notebook Issues

**Problem:** Kernel crashes when loading embedding model
- Your system may need more RAM
- Try a smaller model
- Close other applications

## Support

- **Assignment Questions:** See [`STUDENT_PROJECT_GUIDE.md`](STUDENT_PROJECT_GUIDE.md)
- **Docker Setup:** See [`docker_starter.md`](docker_starter.md)
- **Docker Commands:** See [`COMMANDS.txt`](COMMANDS.txt)
- **Technical Issues:** Check the troubleshooting section above

## Resources

### Learning Materials
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Docker Documentation](https://docs.docker.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

### Understanding RAG
- [What is RAG?](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

## License

This project is designed for educational purposes.

## Acknowledgments

Built using:
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [Mistral AI](https://mistral.ai/) - Language model

---

**Ready to build your first RAG system? Start with [`STUDENT_PROJECT_GUIDE.md`](STUDENT_PROJECT_GUIDE.md)!**
