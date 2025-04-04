StudyBridge
An intelligent chatbot system designed to assist international students with university-specific information using Retrieval Augmented Generation (RAG).

Features
Automated website scraping for university content
PDF document processing for academic regulations and handbooks
ChromaDB-based vector storage for efficient information retrieval
Conversational AI interface for student queries
Standard I/O support for flexible integration
System Requirements
Python 3.10+
NVIDIA GPU
50GB+ storage space
Internet connection for initial LLM download
Project Structure
studybridgechatbot/
├── resources/
│   ├── pdf/              # University PDFs (handbooks, regulations)
│   └── csv/              # Scraped website data
├── scripts/
│   ├── webscraper.py     # Website content extraction
│   └── study_bridge_chat.py  # Chat interface
├── notebooks/
│   └── DataPreparation.ipynb # Setup and data processing
└── requirements.txt
Authentication
Get your Hugging Face API token:

Create an account at Hugging Face
Generate an access token from your account settings
The token needs read access to download the required models
Set up your token:
Option 1: Set environment variable (recommended)

# Linux/MacOS
export HUGGINGFACE_TOKEN="your-token-here"

# Windows (Command Prompt)
set HUGGINGFACE_TOKEN=your-token-here

# Windows (PowerShell)
$env:HUGGINGFACE_TOKEN="your-token-here"
The scripts will automatically retrieve the token using:

import os
token = os.environ.get("HUGGINGFACE_TOKEN")
Option 2: Add directly in scripts (not recommended for production)

You can also add your token directly in DataPreparation.ipynb and study_bridge_chat.py

Quick Start
Clone the repository:

git clone https://github.com/rutujad9/studybridgechatbot.git
cd studybridge
Prepare your resources:

Place university PDFs in resources/pdf/
Configure and run the webscraper:
# set `base_url`
jupyter notebook notebooks/webscraper.py
Run the data preparation notebook:

jupyter notebook notebooks/DataPreparation.ipynb
This will:

Install required dependencies
Initialize ChromaDB
Set up the LLM
Start the chat interface:

python scripts/study_bridge_chat.py
Customization
Adding New Documents
Add PDFs to resources/pdf/
Update website content in resources/csv/
Re-run the data preparation notebook
Integration Options
The chat interface uses standard I/O, allowing for integration with:

Web applications
Command-line interfaces
Messaging platforms
Custom GUI applications
Security Considerations
Ensure sensitive information is not included in scraped content
Review PDFs for confidential data before processing
Configure appropriate access controls for the deployed system
