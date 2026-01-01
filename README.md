 Vector Lab: Semantic Search and Embeddings

This project is a Python-based utility for exploring text embeddings and vector databases. It allows users to convert text into high-dimensional vectors, store them in a remote database, and perform semantic comparisons between different strings.

 Overview

The system uses OpenAI to generate 1,536-dimensional embeddings and Pinecone for remote vector storage. Unlike traditional keyword search, this tool identifies closeness based on the semantic meaning of the text. It includes a manual calculation module to verify the mathematical similarity between vectors.

 Features

- Text-to-Vector conversion using OpenAI API.
- Remote storage and retrieval using Pinecone Vector Database.
- Interactive mode for adding custom text or using pre-defined examples.
- Mathematical verification of similarity using Cosine Similarity calculations.

 Requirements

- Python 3.10 or higher
- OpenAI API Key
- Pinecone API Key

 Setup

1. Install the required libraries:
   pip install -r requirements.txt

2. Create a .env file in the root directory and add your credentials:
   OPENAI_API_KEY=your_openai_key
   PINECONE_API_KEY=your_pinecone_key

3. Run the application:
   python main.py

 Project Structure

- main.py: The main application script containing logic for embeddings and database interaction.
- .env: File for storing private API keys (not included in the repository).
- requirements.txt: List of Python dependencies.
- .gitignore: Configuration to prevent private files from being uploaded to Git.

 Usage

When the script is executed, you will be prompted to choose between using existing example text or entering your own. After text is stored in the database, you can enter a search query. The system will return the most relevant entry from the database along with a similarity score indicating how closely the meanings align.