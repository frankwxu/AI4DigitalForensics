# RAG-based Cyber Forensics Investigation Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Author

**Mohit Ajaykumar Dhabuwala**

- M.S. in Cyber Forensics and Counterterrorism
- Specialization: Digital Forensics & Incident Response (DFIR)
- Proficient in:
  - Memory, Windows, mobile, and network forensics
  - Forensic tools: Magnet AXIOM, EnCase, Volatility, Wireshark
  - Programming languages: Python, Bash, PowerShell for forensic data parsing and automation

## What is RAG?

Retrieval-Augmented Generation (RAG) enhances language model responses by combining information retrieval with text generation. It retrieves relevant information from a knowledge base and uses a language model to generate accurate, factual, and contextually appropriate answers. This enables language models to handle complex queries and access domain-specific knowledge effectively.

This project implements a RAG system to assist in cyber forensics investigations, leveraging LangChain, Hugging Face models, and FAISS for efficient retrieval and question answering over a provided knowledge base. The system processes a text-based scenario, divides it into manageable chunks, generates embeddings, stores them in a vector store, and employs a language model to answer user queries based on the retrieved information.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j1mgpfdpJp0s1eQn7JzJr3lhhB8tjXLn?usp=sharing)

## Video Demonstrations

For a visual demonstration of how this RAG system works, please refer to the following videos:

- **RAG Fundamentals:** [https://youtu.be/T-D1OfcDW1M](https://youtu.be/T-D1OfcDW1M) AND [https://youtu.be/W-ulb-DMtsM](https://youtu.be/W-ulb-DMtsM)
- **RAG Implementation:** [https://youtu.be/shiSITpK0ps](https://youtu.be/shiSITpK0ps)

## Technical Description

The system operates through these steps:

1.  **Environment Setup:** Installation of necessary Python libraries, including `langchain`, `langchain-huggingface`, `faiss-cpu`, and `huggingface_hub`.
2.  **Scenario Definition:** The cyberpunk case study is defined as a string (`document_text`).
3.  **Text Splitting:** The scenario text is divided into chunks using `RecursiveCharacterTextSplitter`, controlled by parameters like `chunk_size`, `chunk_overlap`, and `separators`.
4.  **Embeddings:** Text embeddings are generated using the Hugging Face Inference API.
5.  **Vector Store:** FAISS is used to store and retrieve text embeddings based on similarity.
6.  **Retrieval QA Chain:** LangChain's `RetrievalQA` chain combines the vector store with a language model. It retrieves relevant text chunks based on the user's query and generates an answer.
7.  **Language Model:** The Hugging Face Inference API with a specified model (`mistralai/Mistral-7B-Instruct-v0.1`) is used for response generation.
8.  **Query Processing:** The system receives user queries, retrieves relevant information from the vector store, and generates answers using the language model.

This setup enables the RAG system to answer questions related to the cyber forensics scenario.

## Dependencies

To run this project, ensure you have the following:

- **Python:** 3.7+
- **pip:** Python package installer
- **Hugging Face Account & Access Token:** Required for Hugging Face models and the Inference API.
- **Google Colab:** To execute the notebook.

**Disk space:** Google Colab's virtual environment manages disk space for dependencies.

## How This Project Works

The project uses:

- Hugging Face Models for embedding and text generation.
- LangChain for language model applications.
- FAISS for efficient similarity search.
- Google Colab for running Python code.

The system workflow is:

1.  **Load and split:** Load and divide the cyber forensics document into chunks.
2.  **Embed:** Transform each chunk into a vector representation.
3.  **Store:** Store embeddings in a FAISS index.
4.  **Query:** Transform user's question into an embedding and search the FAISS index.
5.  **Answer:** Generate an answer using a language model based on retrieved information.

## Code Overview

The code comprises:

- Document loading and processing using `RecursiveCharacterTextSplitter`.
- Embedding generation using the Hugging Face Inference API.
- FAISS vectorstore creation.
- `RetrievalQA` chain setup for question answering.
- A simple chat interface for user interaction.

## Why This Approach Is Beneficial

RAG offers these advantages:

- **Contextualized responses:** Answers are based on the provided cyber forensics document.
- **Interactive interface:** User-friendly chat interaction.
- **Efficiency:** FAISS enables fast retrieval.
- **Cloud-based execution:** Google Colab provides a convenient environment.
- **Hugging Face Integration:** Simplifies embedding and text generation.

## System Workflow Diagram

_(Flowchart image included here)_

![Flowchart](image_f6fb04.png-8c6bf71b-bc0b-4179-93a2-cc646df542c9)

## Setup and Usage

1.  **Create a Hugging Face Account** (if needed): Go to [https://huggingface.co/](https://huggingface.co/) and sign up.
2.  **Generate a Hugging Face Access Token:**
    - Log in to your Hugging Face account.
    - Go to your profile settings.
    - Find the "Access Tokens" section.
    - Create a new token.
    - Copy the generated token.
3.  **Open a Google Colab Notebook:**
4.  **Install Python dependencies:** Execute these commands in a Colab cell:

    ```bash
    !pip install transformers langchain langchain_community faiss-cpu huggingface_hub pypdf pymupdf -U langchain langchain-huggingface
    !pip install --upgrade langchain
    ```

5.  **Provide Hugging Face API Token:** Add a code cell to set the `HUGGINGFACEHUB_API_TOKEN` environment variable with your token:

    ```python
    import os
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'hf_your_token'  # Replace 'hf_your_token' with your actual token
    ```

6.  **Provide your knowledge base:** Add a cell to define `document_text` (the scenario).
7.  **Run the code:** Execute the cells to interact with the RAG system.

## Features

- **Google Colab Integration:** Streamlined setup and execution in a cloud-based setting.
- **Hugging Face Integration:** Leverages pre-trained models for embedding and text generation.
- **FAISS Vectorstore:** Enables efficient and rapid similarity search.
- **Text Chunking:** Divides documents into manageable chunks for processing.
- **Chat Interface:** Offers a simple text-based interface for user interaction.

## Contributing

Contributions are welcome! Submit issues or pull requests to improve the project.

## License

This project is released under the MIT License.
