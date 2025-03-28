import logging
import time
import threading
from typing import List, Dict, Any, Optional
from model_db import HotelBookingDatabase
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = os.getenv("DB_PATH", "hotel_bookings.db")

# Mutex for database operations
db_lock = threading.Lock()

class HotelBookingFaissRAG:
    """
    Thread-safe wrapper for the HotelBookingDatabase class.
    Provides additional functionality for refreshing data and managing concurrent access.
    """
    
    def __init__(self, csv_path, db_path: str = DB_PATH):
        """Initialize the system with hotel booking data and database."""
        # Initialize database connection
        self.db = HotelBookingDatabase(db_path, csv_path)
        
        # Initialize last update timestamp
        self.last_update = time.time()
        
        # Store local reference to df and other properties from db
        self.df = self.db.df if hasattr(self.db, 'df') else None
        self.embedding_model = getattr(self.db, 'embedding_model', None)
        self.index = getattr(self.db, 'index', None)
        self.faiss_available = getattr(self.db, 'faiss_available', False)
        self.llm_available = getattr(self.db, 'llm_available', False)
        self.pipe = getattr(self.db, 'pipe', None)
        
        logger.info(f"Initialized RAG wrapper with {len(self.df) if self.df is not None else 0} records")
    
    def refresh_data(self, force: bool = False) -> bool:
        """
        Refresh data from the database if needed.
        Will update if force=True or if data is older than 5 minutes.
        
        Returns:
            True if data was refreshed, False otherwise
        """
        current_time = time.time()
        time_diff = current_time - self.last_update
        
        # Update if forced or if more than 5 minutes have passed
        if force or time_diff > 300:  # 300 seconds = 5 minutes
            logger.info(f"Refreshing data from database (thread {threading.get_ident()})")
            
            # Use thread lock to ensure safe database access
            with db_lock:
                # Get fresh data
                self.df = self.db.get_all_bookings()
                
                # Update references to the database properties
                self.embedding_model = getattr(self.db, 'embedding_model', None)
                self.index = getattr(self.db, 'index', None)
                self.faiss_available = getattr(self.db, 'faiss_available', False)
                self.llm_available = getattr(self.db, 'llm_available', False)
                self.pipe = getattr(self.db, 'pipe', None)
                
                self.last_update = current_time
            
            logger.info(f"Data refreshed with {len(self.df)} records")
            return True
        
        return False
    
    def generate_answer(self, query: str) -> str:
        """
        Generate an answer to a natural language query about hotel bookings.
        Delegates to the database's generate_answer method.
        
        Args:
            query: Natural language query
            
        Returns:
            Answer string
        """
        # Refresh data if needed
        self.refresh_data()
        
        # Use thread lock to ensure safe database access for complex operations
        with db_lock:
            return self.db.generate_answer(query)
    
    def filter_data_for_query(self, query: str, semantic_indices=None):
        """
        Filter the dataset based on query keywords and semantic search.
        Delegates to the database's filter_data_for_query method.
        
        Args:
            query: Natural language query
            semantic_indices: Optional pre-filtered semantic indices
            
        Returns:
            Filtered DataFrame
        """
        # Make sure we have the latest data
        self.refresh_data()
        
        # Use the database method directly
        return self.db.filter_data_for_query(query, semantic_indices)
    
    def analyze_data(self, filtered_df, query: str, use_whole_dataset: bool = True):
        """
        Analyze the filtered dataset based on the query.
        Delegates to the database's analyze_data method.
        
        Args:
            filtered_df: The filtered DataFrame to analyze
            query: The user query
            use_whole_dataset: Whether to use the whole dataset for analysis
            
        Returns:
            Dictionary with analysis results
        """
        return self.db.analyze_data(filtered_df, query, use_whole_dataset)
    
    def faiss_semantic_search(self, query: str, top_k: int = 500):
        """
        Perform semantic search using FAISS index.
        Delegates to the database's faiss_semantic_search method.
        
        Args:
            query: Natural language query
            top_k: Maximum number of results to return
            
        Returns:
            Array of indices
        """
        # Use thread lock to ensure safe access to FAISS index
        with db_lock:
            return self.db.faiss_semantic_search(query, top_k)
    
    def add_booking(self, booking_data: Dict[str, Any]) -> int:
        """
        Add a new booking to the database.
        
        Args:
            booking_data: Dictionary containing booking information
            
        Returns:
            Inserted record ID or -1 if error
        """
        # Use thread lock to ensure safe database writes
        with db_lock:
            booking_id = self.db.add_booking(booking_data)
            if booking_id != -1:
                # Force refresh local data
                self.refresh_data(force=True)
        
        return booking_id
    
    def update_booking(self, booking_id: int, booking_data: Dict[str, Any]) -> bool:
        """
        Update an existing booking.
        
        Args:
            booking_id: ID of the booking to update
            booking_data: Dictionary containing updated booking information
            
        Returns:
            True if successful, False otherwise
        """
        # Use thread lock to ensure safe database writes
        with db_lock:
            success = self.db.update_booking(booking_id, booking_data)
            if success:
                # Force refresh local data
                self.refresh_data(force=True)
        
        return success
    
    def delete_booking(self, booking_id: int) -> bool:
        """
        Delete a booking from the database.
        
        Args:
            booking_id: ID of the booking to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Use thread lock to ensure safe database writes
        with db_lock:
            success = self.db.delete_booking(booking_id)
            if success:
                # Force refresh local data
                self.refresh_data(force=True)
        
        return success
    
    def batch_import_bookings(self, bookings: List[Dict[str, Any]]) -> int:
        """
        Import a batch of bookings.
        
        Args:
            bookings: List of booking dictionaries
            
        Returns:
            Number of bookings successfully imported
        """
        success_count = 0
        
        # Use thread lock to ensure safe database writes
        with db_lock:
            for booking in bookings:
                if self.db.add_booking(booking) != -1:
                    success_count += 1
            
            if success_count > 0:
                # Force refresh local data
                self.refresh_data(force=True)
        
        return success_count
    
    def get_db_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        return self.db.get_db_stats()
    
    def get_update_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent database update log entries.
        
        Args:
            limit: Maximum number of log entries to return
            
        Returns:
            List of log entry dictionaries
        """
        return self.db.get_update_log(limit)
    
    def close(self):
        """Close database connection and release resources."""
        if hasattr(self, 'db'):
            self.db.close()
            logger.info(f"Database connection closed for thread {threading.get_ident()}")