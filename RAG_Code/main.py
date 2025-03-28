import os
import logging
from flask import Flask, request, jsonify
import time
import json
import threading
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from hotel_booking_rag import HotelBookingFaissRAG
import atexit
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH", "")

API_PORT = os.getenv("API_PORT", "8008")
API_HOST = os.getenv("API_HOST", "127.0.0.1")

DB_PATH = os.getenv("DB_PATH", "hotel_bookings.db")
CLEAN_DB_ON_RESTART = os.getenv("CLEAN_DB_ON_RESTART", "false").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
rag_system = None
query_history = []

# Custom JSON encoder to handle NumPy types
def convert_numpy_types(obj):
    """Convert NumPy types to standard Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj

def initialize_rag(path) -> bool:
    """Initialize the RAG system with the hotel booking database."""
    global rag_system
    try:
        # Initialize RAG system with database integration
        logger.info(f"Initializing RAG system with database integration (thread {threading.get_ident()})")
        
        # Use default paths for database and initial CSV
        rag_system = HotelBookingFaissRAG(csv_path = path)
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        return False

def handle_errors(func):
    """Decorator to handle errors in API endpoints."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return jsonify({
                "status": "error",
                "message": str(e),
                "timestamp": time.time()
            }), 500
    wrapper.__name__ = func.__name__
    return wrapper

@app.route('/health', methods=['GET'])
@handle_errors
def health_check():
    """Check the health of the API and its dependencies."""
    if rag_system is None:
        return jsonify({
            "status": "error",
            "message": "RAG system not initialized",
            "timestamp": time.time()
        }), 500
    
    # Check if the dataframe is loaded
    if not hasattr(rag_system, 'df') or rag_system.df is None:
        return jsonify({
            "status": "error",
            "message": "Dataset not loaded",
            "timestamp": time.time()
        }), 500
    
    # Get database stats
    db_stats = rag_system.get_db_stats()
    
    # Check if the embeddings are initialized
    if not hasattr(rag_system, 'index') or rag_system.index is None:
        return jsonify({
            "status": "warning",
            "message": "FAISS index not initialized, falling back to traditional filtering",
            "timestamp": time.time()
        }), 200
    
    # Check if the LLM is initialized
    llm_status = "ok" if (hasattr(rag_system, 'pipe') and rag_system.pipe is not None) else "warning"
    llm_message = "LLM initialized" if llm_status == "ok" else "LLM not initialized, using simplified responses"
    
    # All systems operational
    return jsonify({
        "status": "ok",
        "message": "All systems operational",
        "components": {
            "database": "ok",
            "embedding_model": "ok" if hasattr(rag_system, 'embedding_model') else "error",
            "faiss_index": "ok" if hasattr(rag_system, 'index') else "error",
            "llm": llm_status
        },
        "dataset_size": len(rag_system.df),
        "database_stats": db_stats,
        "llm_status": llm_message,
        "thread_id": threading.get_ident(),
        "timestamp": time.time()
    }), 200

@app.route('/analytics', methods=['POST'])
@handle_errors
def get_analytics():
    """
    Process analytics requests with custom filters and metrics.
    """
    if rag_system is None:
        return jsonify({
            "status": "error", 
            "message": "RAG system not initialized"
        }), 500
    
    # Get request data
    data = request.get_json()
    if not data:
        return jsonify({
            "status": "error",
            "message": "No data provided"
        }), 400
    
    start_time = time.time()
    
    # Ensure we have the latest data
    rag_system.refresh_data()
    
    # Process either by natural language query or by specific filters/metrics
    if "query" in data:
        # Natural language query processing
        query = data["query"]
        
        # Track query history
        query_history.append({
            "timestamp": time.time(),
            "query": query,
            "type": "analytics",
            "thread_id": threading.get_ident()
        })
        
        # Get semantically similar records
        if hasattr(rag_system, 'faiss_available') and rag_system.faiss_available:
            semantic_indices = rag_system.faiss_semantic_search(query)
            logger.info(f"Found {len(semantic_indices)} semantically relevant records for query: {query}")
        else:
            semantic_indices = None
            logger.warning("FAISS not available, using traditional filtering")
        
        # Filter relevant data based on query and semantic search
        filtered_data = rag_system.filter_data_for_query(query, semantic_indices)
        
        # Analyze the filtered data
        analysis_results = rag_system.analyze_data(filtered_data, query, use_whole_dataset=True)
        
    else:
        # Structured analytics processing with filters and metrics
        filters = data.get("filters", {})
        metrics = data.get("metrics", [])
        
        # Track structured query history
        query_history.append({
            "timestamp": time.time(),
            "filters": filters,
            "metrics": metrics,
            "type": "structured_analytics",
            "thread_id": threading.get_ident()
        })
        
        # Apply filters to create filtered dataset
        filtered_data = rag_system.df.copy()
        
        for key, value in filters.items():
            if key in filtered_data.columns:
                if isinstance(value, dict):
                    # Handle comparison operators
                    if "gt" in value:
                        filtered_data = filtered_data[filtered_data[key] > value["gt"]]
                    elif "lt" in value:
                        filtered_data = filtered_data[filtered_data[key] < value["lt"]]
                    elif "eq" in value:
                        filtered_data = filtered_data[filtered_data[key] == value["eq"]]
                else:
                    # Direct equality comparison
                    filtered_data = filtered_data[filtered_data[key] == value]
        
        # Mock query for analysis based on metrics
        mock_query = " ".join(metrics)
        analysis_results = rag_system.analyze_data(filtered_data, mock_query, use_whole_dataset=False)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Convert NumPy types to standard Python types for JSON serialization
    serializable_results = convert_numpy_types(analysis_results)
    
    return jsonify({
        "status": "success",
        "processing_time_seconds": processing_time,
        "results": serializable_results,
        "record_count": int(len(filtered_data)),
        "thread_id": threading.get_ident(),
        "timestamp": time.time()
    }), 200

@app.route('/ask', methods=['POST'])
@handle_errors
def ask_question():
    """
    Answer questions about hotel bookings using the RAG system.
    """
    if rag_system is None:
        return jsonify({
            "status": "error", 
            "message": "RAG system not initialized"
        }), 500
    
    # Get request data
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({
            "status": "error",
            "message": "No question provided"
        }), 400
    
    question = data["question"]
    
    # Track query history
    query_history.append({
        "timestamp": time.time(),
        "question": question,
        "type": "question",
        "thread_id": threading.get_ident()
    })
    
    start_time = time.time()
    
    # Generate answer using the RAG system
    answer = rag_system.generate_answer(question)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return jsonify({
        "status": "success",
        "answer": answer,
        "processing_time_seconds": processing_time,
        "thread_id": threading.get_ident(),
        "timestamp": time.time()
    }), 200

@app.route('/bookings', methods=['POST'])
@handle_errors
def add_booking():
    """
    Add a new booking to the database.
    """
    if rag_system is None:
        return jsonify({
            "status": "error", 
            "message": "RAG system not initialized"
        }), 500
    
    # Get request data
    data = request.get_json()
    if not data or "booking" not in data:
        return jsonify({
            "status": "error",
            "message": "No booking data provided"
        }), 400
    
    booking_data = data["booking"]
    
    # Add booking to database
    booking_id = rag_system.add_booking(booking_data)
    
    if booking_id == -1:
        return jsonify({
            "status": "error",
            "message": "Failed to add booking",
            "timestamp": time.time()
        }), 500
    
    return jsonify({
        "status": "success",
        "message": "Booking added successfully",
        "booking_id": booking_id,
        "thread_id": threading.get_ident(),
        "timestamp": time.time()
    }), 201

@app.route('/bookings/<int:booking_id>', methods=['PUT'])
@handle_errors
def update_booking(booking_id):
    """
    Update an existing booking.
    """
    if rag_system is None:
        return jsonify({
            "status": "error", 
            "message": "RAG system not initialized"
        }), 500
    
    # Get request data
    data = request.get_json()
    if not data or "booking" not in data:
        return jsonify({
            "status": "error",
            "message": "No booking data provided"
        }), 400
    
    booking_data = data["booking"]
    
    # Update booking
    success = rag_system.update_booking(int(booking_id), booking_data)
    
    if not success:
        return jsonify({
            "status": "error",
            "message": f"Failed to update booking {booking_id}",
            "timestamp": time.time()
        }), 404
    
    return jsonify({
        "status": "success",
        "message": f"Booking {booking_id} updated successfully",
        "thread_id": threading.get_ident(),
        "timestamp": time.time()
    }), 200

@app.route('/bookings/<int:booking_id>', methods=['DELETE'])
@handle_errors
def delete_booking(booking_id):
    """Delete a booking from the database."""
    if rag_system is None:
        return jsonify({
            "status": "error", 
            "message": "RAG system not initialized"
        }), 500
    
    # Delete booking
    success = rag_system.delete_booking(int(booking_id))
    
    if not success:
        return jsonify({
            "status": "error",
            "message": f"Failed to delete booking {booking_id}",
            "timestamp": time.time()
        }), 404
    
    return jsonify({
        "status": "success",
        "message": f"Booking {booking_id} deleted successfully",
        "thread_id": threading.get_ident(),
        "timestamp": time.time()
    }), 200

@app.route('/bookings/batch', methods=['POST'])
@handle_errors
def batch_import():
    """
    Import a batch of bookings.
    """
    if rag_system is None:
        return jsonify({
            "status": "error", 
            "message": "RAG system not initialized"
        }), 500
    
    # Get request data
    data = request.get_json()
    if not data or "bookings" not in data or not isinstance(data["bookings"], list):
        return jsonify({
            "status": "error",
            "message": "No bookings array provided"
        }), 400
    
    bookings = data["bookings"]
    
    # Import bookings
    success_count = rag_system.batch_import_bookings(bookings)
    
    return jsonify({
        "status": "success",
        "message": f"Imported {success_count} of {len(bookings)} bookings successfully",
        "success_count": success_count,
        "total_count": len(bookings),
        "thread_id": threading.get_ident(),
        "timestamp": time.time()
    }), 201

@app.route('/metrics', methods=['GET'])
@handle_errors
def get_metrics():
    """
    Get API usage metrics and query history.
    """
    if rag_system is None:
        return jsonify({
            "status": "error", 
            "message": "RAG system not initialized"
        }), 500
    
    # Get database stats
    db_stats = rag_system.get_db_stats()
    
    # Get update log
    update_log = rag_system.get_update_log(20)  # Get most recent 20 updates
    
    if not query_history:
        return jsonify({
            "status": "success",
            "message": "No queries recorded yet",
            "metrics": {
                "total_queries": 0,
                "analytics_queries": 0,
                "questions": 0
            },
            "database": {
                "stats": db_stats,
                "recent_updates": update_log
            },
            "thread_id": threading.get_ident(),
            "timestamp": time.time()
        }), 200
    
    # Calculate metrics
    total_queries = len(query_history)
    analytics_queries = sum(1 for q in query_history if q.get("type") in ["analytics", "structured_analytics"])
    questions = sum(1 for q in query_history if q.get("type") == "question")
    
    # Get top questions (most recent 10)
    recent_questions = [q.get("question") for q in query_history 
                        if q.get("type") == "question" and "question" in q][-10:]
    
    # Calculate average response times
    avg_time = 0
    if "processing_time_seconds" in query_history[0]:
        avg_time = sum(q.get("processing_time_seconds", 0) for q in query_history) / total_queries
    
    return jsonify({
        "status": "success",
        "metrics": {
            "total_queries": total_queries,
            "analytics_queries": analytics_queries,
            "questions": questions,
            "avg_response_time": avg_time
        },
        "recent_questions": recent_questions,
        "query_history": query_history[-20:],  # Return most recent 20 queries
        "database": {
            "stats": db_stats,
            "recent_updates": update_log
        },
        "thread_id": threading.get_ident(),
        "timestamp": time.time()
    }), 200

@app.route('/refresh', methods=['POST'])
@handle_errors
def force_refresh():
    """Force refresh of the RAG system data from the database."""
    if rag_system is None:
        return jsonify({
            "status": "error", 
            "message": "RAG system not initialized"
        }), 500
    
    # Force refresh
    rag_system.refresh_data(force=True)
    
    return jsonify({
        "status": "success",
        "message": "RAG system data refreshed from database",
        "record_count": len(rag_system.df),
        "thread_id": threading.get_ident(),
        "timestamp": time.time()
    }), 200

# Add a cleanup function to close the database connection when the application exits
def cleanup():
    if rag_system is not None:
        rag_system.close()
        logger.info("Closed database connection on application exit")

atexit.register(cleanup)

if __name__ == "__main__":
    # If CLEAN_DB_ON_RESTART is true, remove the database file
    if CLEAN_DB_ON_RESTART and os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            logger.info(f"Database file {DB_PATH} removed as requested in configuration")
        except Exception as e:
            logger.error(f"Failed to remove database file {DB_PATH}: {e}")

    # Initialize the RAG system
    if initialize_rag(DATASET_PATH):
        logger.info("RAG system initialized successfully")
    else:
        logger.error("Failed to initialize RAG system")
    
    # Run the Flask app
    port = int(os.environ.get("PORT", API_PORT))
    app.run(host=API_HOST, port=port, debug=False, threaded=True)