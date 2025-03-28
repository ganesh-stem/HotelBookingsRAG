# Hotel Booking Analytics System

## Overview

This system provides a comprehensive solution for hotel booking data management and analysis. It combines a Retrieval-Augmented Generation (RAG) system with a database backend to store, analyze, and provide insights from hotel booking data. The application features a RESTful API for data interaction, natural language querying capabilities, and tools for performance evaluation.

## Features

- Database integration for persistent storage of hotel booking data
- Natural language question answering about booking data
- Advanced analytics with both natural language and structured queries
- CRUD operations for booking management (add, update, delete)
- Batch import functionality for efficient data loading
- Semantic search using FAISS vector database
- Visualization of analytics data
- Multi-threaded design for concurrent access
- Comprehensive API health monitoring and performance evaluation

## System Components

1. **Main API Server** (`main.py`): Flask-based REST API that handles all requests and coordinates between components
2. **RAG System** (`hotel_booking_rag.py`): Thread-safe wrapper for the database with semantic search capabilities
3. **Database Handler** (`model_db.py`): Core database functionality with LLM and FAISS integration
4. **Data Update Tool** (`data_update.py`): Command-line utility for data manipulation and testing
5. **Performance Evaluator** (`performance_evaluation.py`): Tool for measuring and reporting system performance

## Setup and Installation

### Prerequisites

- Python 3.11 or higher
- Provided in the **requirements.txt** file in **RAG_Code** Folder

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Model Configuration
EMBEDDING_MODEL="all-MiniLM-L6-v2"
# Change the model according to your system
# Some Examples:
# mistralai/Mistral-7B-Instruct-v0.2
# meta-llama/Llama-2-7b-chat-hf
LLM_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# HuggingFace Configuration
HF_TOKEN="<hf_token>"
# Set to "false" to skip HuggingFace login if needed
HF_LOGIN_ENABLED="true"

# Database Configuration
DB_PATH = "hotel_bookings.db"
# Set to "true" to remove the database file on restart
CLEAN_DB_ON_RESTART="false"

# Dataset Configuration
DATASET_PATH="data/hotel_bookings.csv"
# Use "full" to process the entire dataset or specify a number for sampling
SAMPLE_SIZE="2500"

# API Configuration
API_PORT=8008
API_HOST="localhost"
API_URL=http://${API_HOST}:${API_PORT}

# Performance and Memory
DEVICE="cuda"
USE_4BIT_QUANTIZATION=true
```

### Installation Steps

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install required packages: `pip install -r requirements.txt`
5. 5. Download the dataset from this link: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
6. Ensure you have a hotel bookings CSV file available at the path specified in your `.env` file

## Running the System

### Start the API Server

```bash
python main.py
```

This will initialize the database, load the dataset, and start the API server.

## API Endpoints

### Health Check
- `GET /health`: Check system status and component health
- curl http://localhost:8008/health
  
### Question Answering
- `POST /ask`: Answer natural language questions about hotel bookings
  - Request body: `{"question": "What's the cancellation rate for resort hotels?"}`
  - curl -X POST http://localhost:8008/ask -H "Content-Type: application/json" -d "{\"question\": \"Which country has the highest number of bookings?\"}"
### Analytics
- `POST /analytics`: Get analytics based on natural language or structured queries
  - Natural language query: `{"query": "Show me revenue by month for 2017"}`
  - Structured query: `{"filters": {"hotel": "Resort Hotel"}, "metrics": ["revenue", "cancellation_rate"]}`
  - curl -X POST http://localhost:8008/analytics -H "Content-Type: application/json" -d "{\"query\": \"Show me the cancellation rate for resort hotels\"}"
    
### Booking Management
- `POST /bookings`: Add a new booking
  - Request body: `{"booking": {...booking data...}}`
- `PUT /bookings/{booking_id}`: Update a booking
  - Request body: `{"booking": {...updated fields...}}`
- `DELETE /bookings/{booking_id}`: Delete a booking
- `POST /bookings/batch`: Batch import bookings
  - Request body: `{"bookings": [{...booking1...}, {...booking2...}]}`

### System Operations
- `GET /metrics`: Get API usage metrics and query history

### Running Performance Evaluation

The `performance_evaluation.py` script allows you to test and benchmark the system:

```bash
# Run a full evaluation
python performance_evaluation.py --url http://localhost:8008

# Test only questions endpoint
python performance_evaluation.py --mode questions

# Test only analytics endpoint
python performance_evaluation.py --mode analytics

# Test only booking operations
python performance_evaluation.py --mode bookings
```
### Using the Data Update Tool

The `data_update.py` script provides a command-line interface for interacting with the API:
```bash
# Run demo sequence
python data_update.py --action demo

# Add a single booking
python data_update.py --action add

# Update a booking
python data_update.py --action update --id [booking_id]

# Delete a booking
python data_update.py --action delete --id [booking_id]

# Batch import bookings
python data_update.py --action batch --count [number_of_bookings]

# Ask a question
python data_update.py --action ask --question "What's the cancellation rate for resort hotels?"

# Run an analytics query
python data_update.py --action analytics --query "Show me revenue comparison between resort and city hotels"

# Force refresh the system data
python data_update.py --action refresh
```

## System Architecture

The system follows a layered architecture:

1. **API Layer** (`main.py`): Handles HTTP requests and responses
2. **Service Layer** (`hotel_booking_rag.py`): Provides thread-safe access to the database
3. **Data Layer** (`model_db.py`): Manages database operations and AI model integration
4. **Utility Layer** (`data_update.py`, `performance_evaluation.py`): Provides tools for testing and evaluation

The system uses several AI components:
- **Embedding Model**: Converts booking data and queries into vector representations
- **FAISS Index**: Enables semantic search across the dataset
- **LLM**: Generates natural language responses to questions

## Performance Considerations

- The system can be deployed with different sized models based on available resources
- 4-bit quantization is used by default to reduce memory requirements
- Thread-safe design ensures consistent access from multiple clients

## Troubleshooting

### Common Issues

1. **API not responding**: Check if the server is running and the port is correct
2. **Memory errors**: Try reducing the model size or enabling quantization
3. **Slow response times**: Adjust the `SAMPLE_SIZE` to use fewer records or upgrade hardware
4. **Database errors**: Ensure the database path is correct and the directory is writable
4. **Database Values**: Delete the database and re-run the program if working with new dataset.
