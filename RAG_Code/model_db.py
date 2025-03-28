import os
import logging
import pandas as pd
import numpy as np
import sqlite3
import threading
import re
import torch
import faiss
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Handle sample size - can be "full" or a number
SAMPLE_SIZE_ENV = os.getenv("SAMPLE_SIZE", "2500")
USE_FULL_DATASET = SAMPLE_SIZE_ENV.lower() == "full"
SAMPLE_SIZE = None if USE_FULL_DATASET else int(SAMPLE_SIZE_ENV)

USE_4BIT_QUANTIZATION = os.getenv("USE_4BIT_QUANTIZATION", "true").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-local storage for database connections
thread_local = threading.local()

# Authenticate with Hugging Face if enabled
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_LOGIN_ENABLED = os.getenv("HF_LOGIN_ENABLED", "true").lower() == "true"

if HF_LOGIN_ENABLED and HF_TOKEN:
    try:
        from huggingface_hub import login
        login(HF_TOKEN)
        logger.info("Authenticated with Hugging Face Hub")
    except Exception as e:
        logger.warning(f"Failed to authenticate with Hugging Face Hub: {str(e)}")
else:
    logger.info("Hugging Face authentication skipped based on configuration")

class HotelBookingDatabase:
    """
    Advanced database handler for hotel booking data with real-time update capabilities,
    integrated with FAISS vector search and LLM question answering.
    """
    
    def __init__(self, db_path: str = "hotel_bookings.db", csv_path: Optional[str] = None):
        """
        Initialize the database connection and RAG components.
        
        Args:
            db_path: Path to the SQLite database file
            csv_path: Optional path to CSV file to initialize the database
        """
        self.db_path = db_path
        
        # Set up database
        self.setup_database()
        
        # Initialize RAG components
        self.embedding_model = None
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.index = None
        self.text_descriptions = None
        self.original_indices = None
        self.faiss_available = False
        self.llm_available = False
        
        # If CSV path provided, initialize database from CSV
        if csv_path and os.path.exists(csv_path):
            print(f'Using CSV File: {csv_path}')
            print('=================================================')
            logger.info(f'Initializing database from CSV: {csv_path}')
            self.initialize_from_csv(csv_path)
            
            # Initialize RAG components
            df = self.get_all_bookings()  # Get data from database
            if not df.empty:
                # Clean data for RAG
                print(f'Clean Data for RAG')
                self.df = self.clean_data_for_rag(df)
                
                # Initialize LLM for answering
                self.init_llm()
                
                # Initialize FAISS index for semantic search
                self.init_faiss()
    
    def get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(thread_local, "conn"):
            logger.info(f"Creating new database connection for thread {threading.get_ident()}")
            thread_local.conn = sqlite3.connect(self.db_path)
        return thread_local.conn
    
    def setup_database(self) -> bool:
        """Set up the database schema."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create bookings table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hotel TEXT,
                is_canceled INTEGER,
                lead_time INTEGER,
                arrival_date_year INTEGER,
                arrival_date_month TEXT,
                arrival_date_week_number INTEGER,
                arrival_date_day_of_month INTEGER,
                stays_in_weekend_nights INTEGER,
                stays_in_week_nights INTEGER,
                adults INTEGER,
                children REAL,
                babies INTEGER,
                meal TEXT,
                country TEXT,
                market_segment TEXT,
                distribution_channel TEXT,
                is_repeated_guest INTEGER,
                previous_cancellations INTEGER,
                previous_bookings_not_canceled INTEGER,
                reserved_room_type TEXT,
                assigned_room_type TEXT,
                booking_changes INTEGER,
                deposit_type TEXT,
                agent REAL,
                company REAL,
                days_in_waiting_list INTEGER,
                customer_type TEXT,
                adr REAL,
                required_car_parking_spaces INTEGER,
                total_of_special_requests INTEGER,
                reservation_status TEXT,
                reservation_status_date TEXT,
                total_nights INTEGER,
                total_guests INTEGER,
                revenue REAL,
                has_previous_stay INTEGER,
                has_previous_cancellations INTEGER,
                has_booking_changes INTEGER,
                requires_parking INTEGER,
                has_special_requests INTEGER,
                waitlist_category TEXT,
                last_updated TEXT
            )
            ''')
            
            # Create an index for common query fields
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_hotel ON bookings (hotel)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON bookings (arrival_date_year, arrival_date_month)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_country ON bookings (country)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_canceled ON bookings (is_canceled)')
            
            # Create update log table to track changes
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS update_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                operation TEXT,
                record_count INTEGER,
                details TEXT
            )
            ''')
            
            # Create embeddings table to store vector embeddings
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                booking_id INTEGER,
                embedding_data BLOB,
                last_updated TEXT,
                FOREIGN KEY (booking_id) REFERENCES bookings (id) ON DELETE CASCADE
            )
            ''')
            
            # Create query history table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                query TEXT,
                query_type TEXT,
                response TEXT,
                processing_time REAL
            )
            ''')
            
            conn.commit()
            logger.info("Database setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up database: {str(e)}")
            return False
    
    def initialize_from_csv(self, csv_path: str) -> bool:
        """Initialize the database from a CSV file."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check if the database is already populated
            cursor.execute("SELECT COUNT(*) FROM bookings")
            if cursor.fetchone()[0] > 0:
                logger.info("Database already contains data, skipping initialization")
                return True
            
            # Read CSV file
            if USE_FULL_DATASET:
                logger.info("Using full dataset as requested in configuration")
            else:
                logger.info(f"Sampling {SAMPLE_SIZE} records from dataset")
                df = pd.read_csv(csv_path).sample(n=SAMPLE_SIZE, random_state=42)

            logger.info(f"Loaded CSV with {len(df)} records")
            
            # Convert dates
            df['reservation_status_date'] = df['reservation_status_date'].astype('string')

            # Clean data
            df['children'] = df['children'].fillna(0)
            df['agent'] = df['agent'].fillna(-1)
            df['company'] = df['company'].fillna(-1)
            
            # Ensure boolean columns are properly handled
            bool_cols = ['is_canceled', 'is_repeated_guest']
            for col in bool_cols:
                if col in df.columns:
                    df[col] = df[col].astype(int)

            # Ensure string columns are properly handled
            string_cols = ['hotel', 'arrival_date_month', 'meal', 'country', 
                          'market_segment', 'distribution_channel', 'reserved_room_type', 
                          'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status']
            
            for col in string_cols:
                if col in df.columns:
                    # Convert to string and replace NaN/None with 'Unknown'
                    df[col] = df[col].astype(str)
                    df[col] = df[col].replace(['nan', 'None'], 'Unknown')

            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:  # Numerical columns
                    if df[col].nunique() > 30:
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
                elif df[col].dtype in ['object', 'category', 'string']:  # Categorical columns
                    df[col] = df[col].fillna(df[col].mode()[0])
            
            # Calculate additional fields
            df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
            df['total_guests'] = df['adults'] + df['children'] + df['babies']
            df['revenue'] = df['adr'] * df['total_nights']
            # Flag repeated guests
            df['has_previous_stay'] = (df['previous_bookings_not_canceled'] > 0).astype(int)
            
            # Flag previous cancellations
            df['has_previous_cancellations'] = (df['previous_cancellations'] > 0).astype(int)
            
            # Flag for bookings with changes
            df['has_booking_changes'] = (df['booking_changes'] > 0).astype(int)
            
            # Flag for bookings with parking requirements
            df['requires_parking'] = (df['required_car_parking_spaces'] > 0).astype(int)
            
            # Flag for bookings with special requests
            df['has_special_requests'] = (df['total_of_special_requests'] > 0).astype(int)
            
            # Waitlist category
            df['waitlist_category'] = pd.cut(
                df['days_in_waiting_list'],
                bins=[-1, 0, 7, 30, float('inf')],
                labels=['No wait', '1-7 days', '8-30 days', 'Over 30 days']
            )
            # Add last_updated timestamp
            df['last_updated'] = datetime.now().isoformat()
            
            # Insert data into database
            # Convert DataFrame to a list of tuples for batch insertion
            records = df.to_dict('records')
            for record in records:
                placeholders = ', '.join(['?'] * len(record))
                columns = ', '.join(record.keys())
                query = f"INSERT INTO bookings ({columns}) VALUES ({placeholders})"
                cursor.execute(query, list(record.values()))
            
            # Log the update
            cursor.execute(
                "INSERT INTO update_log (timestamp, operation, record_count, details) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), "initial_import", len(df), f"Imported from {csv_path}")
            )
            
            conn.commit()
            logger.info(f"Initialized database with {len(df)} records from CSV")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing database from CSV: {str(e)}")
            if conn:
                conn.rollback()
            return False
    
    def clean_data_for_rag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data for RAG."""
        # Create a copy to avoid modifying the original dataframe
        df_clean = df.copy()
        
        # Convert dates
        if 'reservation_status_date' in df_clean.columns:
            df_clean['reservation_status_date'] = pd.to_datetime(df_clean['reservation_status_date'], format="%Y-%m-%d", errors='coerce')
        
        # Handle sampling based on configuration
        if USE_FULL_DATASET:
            logger.info("Using full dataset as requested in configuration")
        else:
            logger.info(f"Sampling {SAMPLE_SIZE} records from dataset")
            df_clean = df_clean.sample(n=SAMPLE_SIZE, random_state=42)
        
            # Clean data
            df_clean['children'] = df_clean['children'].fillna(0)
            df_clean['agent'] = df_clean['agent'].fillna(-1)
            df_clean['company'] = df_clean['company'].fillna(-1)
            
            # Ensure boolean columns are properly handled
            bool_cols = ['is_canceled', 'is_repeated_guest']
            for col in bool_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype(int)

            # Ensure string columns are properly handled
            string_cols = ['hotel', 'arrival_date_month', 'meal', 'country', 
                            'market_segment', 'distribution_channel', 'reserved_room_type', 
                            'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status']
            
            for col in string_cols:
                if col in df_clean.columns:
                    # Convert to string and replace NaN/None with 'Unknown'
                    df_clean[col] = df_clean[col].astype(str)
                    df_clean[col] = df_clean[col].replace(['nan', 'None'], 'Unknown')

            for col in df_clean.columns:
                if df_clean[col].dtype in ['int64', 'float64']:  # Numerical columns
                    if df_clean[col].nunique() > 30:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                    else:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                elif df_clean[col].dtype in ['object', 'category', 'string']:  # Categorical columns
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
            # Calculate additional fields
            df_clean['total_nights'] = df_clean['stays_in_weekend_nights'] + df_clean['stays_in_week_nights']
            df_clean['total_guests'] = df_clean['adults'] + df['children'] + df_clean['babies']
            df_clean['revenue'] = df_clean['adr'] * df_clean['total_nights']
            
            # Flag repeated guests
            df_clean['has_previous_stay'] = (df_clean['previous_bookings_not_canceled'] > 0).astype(int)
            
            # Flag previous cancellations
            df_clean['has_previous_cancellations'] = (df_clean['previous_cancellations'] > 0).astype(int)
            
            # Flag for bookings with changes
            df_clean['has_booking_changes'] = (df_clean['booking_changes'] > 0).astype(int)
            
            # Flag for bookings with parking requirements
            df_clean['requires_parking'] = (df_clean['required_car_parking_spaces'] > 0).astype(int)
            
            # Flag for bookings with special requests
            df_clean['has_special_requests'] = (df_clean['total_of_special_requests'] > 0).astype(int)
            
            # Waitlist category
            df_clean['waitlist_category'] = pd.cut(
                df_clean['days_in_waiting_list'],
                bins=[-1, 0, 7, 30, float('inf')],
                labels=['No wait', '1-7 days', '8-30 days', 'Over 30 days']
            )
            # Add last_updated timestamp
            df_clean['last_updated'] = datetime.now().isoformat()
        
        # Calculate derived fields
        if 'total_nights' not in df_clean.columns:
            df_clean['total_nights'] = df_clean['stays_in_weekend_nights'] + df_clean['stays_in_week_nights']
        
        if 'total_guests' not in df_clean.columns:
            df_clean['total_guests'] = df_clean['adults'] + df_clean['children'] + df_clean['babies']
        
        if 'revenue' not in df_clean.columns:
            df_clean['revenue'] = df_clean['adr'] * df_clean['total_nights']
        
        logger.info("Data cleaning for RAG complete")
        return df_clean
    
    def init_llm(self):
        """Initialize the LLM for answering questions."""
        logger.info(f"Initializing LLM on {DEVICE}...")
        
        try:
            # Initialize tokenizer with proper error handling
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            
            logger.info("Loading model...")
            # For memory constraints, add low_cpu_mem_usage and load in 8bit if needed
            load_options = {
                "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }

            if USE_4BIT_QUANTIZATION:
                load_options["bnb_4bit_compute_dtype"] =  torch.float16
                load_options["load_in_4bit"] = True
            else:
                load_options["load_in_8bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, **load_options)
            
            logger.info("Creating pipeline...")
            # Create pipeline with proper error handling
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
            )
            
            logger.info("LLM initialized successfully")
            self.llm_available = True
        
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            logger.warning("Using simplified mode without LLM...")
            # Set a flag to indicate we're running without LLM
            self.llm_available = False
            # Define a simple answer function as fallback
            self.pipe = None
    
    def init_faiss(self):
        """Initialize FAISS index and embeddings for the dataset."""
        logger.info("Initializing FAISS index...")
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
            # Create descriptive texts for each record to embed
            # We'll combine important fields into text descriptions for better semantic matching
            self.text_descriptions = self.generate_text_descriptions()
            
            # Generate embeddings
            logger.info("Generating embeddings for dataset...")
            embeddings = self.embedding_model.encode(self.text_descriptions, show_progress_bar=True)
            
            # Ensure embeddings are normalized for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity with normalized vectors
            self.index.add(embeddings)
            
            logger.info(f"FAISS index created with {self.index.ntotal} vectors of dimension {embedding_dim}")
            
            # Store original indices for retrieval
            self.original_indices = np.arange(len(self.df))
            self.faiss_available = True
            
        except Exception as e:
            logger.error(f"Error initializing FAISS: {str(e)}")
            logger.warning("Using traditional filtering fallback...")
            self.faiss_available = False
    
    def generate_text_descriptions(self) -> List[str]:
        """Generate text descriptions for each record for semantic search."""
        descriptions = []
        
        for _, row in self.df.iterrows():
            # Create a descriptive text that captures key information about the booking
            desc = (
                f"A {row['hotel']} booking in {row['arrival_date_month']} {row['arrival_date_year']} "
                f"for {row['total_nights']} nights with {row['total_guests']} guests. "
                f"Room type {row['reserved_room_type']} (assigned type {row['assigned_room_type']}) "
                f"at rate ${row['adr']:.2f} per night. "
                f"Customer from {row['country']} booked through {row['market_segment']} channel "
                f"with {row['lead_time']} days lead time. "
                f"Meal package: {row['meal']}. "
                f"Booking has {row['booking_changes']} changes and was on waitlist for {row['days_in_waiting_list']} days. "
                f"Customer has {row['previous_bookings_not_canceled']} previous stays and {row['previous_cancellations']} previous cancellations. "
                f"Requested {row['required_car_parking_spaces']} parking spaces and made {row['total_of_special_requests']} special requests. "
                f"Cancellation status: {'Cancelled' if row['is_canceled'] == 1 else 'Not cancelled'}. "
                f"Reservation status: {row['reservation_status']} as of {row['reservation_status_date']}."
            )
            descriptions.append(desc)
        
        return descriptions
    
    def faiss_semantic_search(self, query: str, top_k: int = 500) -> np.ndarray:
        """Perform semantic search using FAISS index."""
        if not self.faiss_available:
            # Return all indices if FAISS is not available
            return self.original_indices
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar records
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.df)))
        
        # Map to original dataframe indices
        result_indices = self.original_indices[indices[0]]
        
        return result_indices
    
    def filter_data_for_query(self, query: str, semantic_indices: np.ndarray = None) -> pd.DataFrame:
        """Filter the dataset based on query keywords and semantic search."""
        # Start with semantically retrieved records if available
        if semantic_indices is not None and len(semantic_indices) > 0:
            filtered_df = self.df.iloc[semantic_indices].copy()
        else:
            filtered_df = self.df.copy()
        
        # Extract year, month if present in the query
        year_match = re.search(r'(\d{4})', query)
        year = int(year_match.group(1)) if year_match else None
        
        month_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
        month_match = re.search(month_pattern, query, re.IGNORECASE)
        month = month_match.group(1) if month_match else None
        
        # Match lead time patterns
        lead_time_patterns = {
            r'(last minute|last-minute)': (0, 7),
            r'(short notice|short-notice)': (0, 30),
            r'(advance|early)': (90, 1000)
        }
        
        for pattern, (min_days, max_days) in lead_time_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                filtered_df = filtered_df[(filtered_df['lead_time'] >= min_days) & 
                                          (filtered_df['lead_time'] <= max_days)]
        
        # Filter by reservation status
        if 'canceled' in query.lower() or 'cancelled' in query.lower():
            filtered_df = filtered_df[filtered_df['is_canceled'] == 1]
        elif 'not canceled' in query.lower() or 'not cancelled' in query.lower():
            filtered_df = filtered_df[filtered_df['is_canceled'] == 0]
        
        # Filter by check-out status
        if 'checked out' in query.lower() or 'check-out' in query.lower():
            filtered_df = filtered_df[filtered_df['reservation_status'] == 'Check-Out']
        
        # Filter by no-show status
        if 'no-show' in query.lower() or 'no show' in query.lower():
            filtered_df = filtered_df[filtered_df['reservation_status'] == 'No-Show']
        
        # Filter by repeat guests
        if 'repeat' in query.lower() or 'returning' in query.lower():
            filtered_df = filtered_df[filtered_df['is_repeated_guest'] == 1]
        elif 'first time' in query.lower() or 'new customer' in query.lower():
            filtered_df = filtered_df[filtered_df['is_repeated_guest'] == 0]
        
        # Filter by special requests
        if 'special request' in query.lower():
            filtered_df = filtered_df[filtered_df['total_of_special_requests'] > 0]
        
        # Filter by parking requirements
        if 'parking' in query.lower():
            filtered_df = filtered_df[filtered_df['required_car_parking_spaces'] > 0]

        # Apply filters based on extracted information
        if year:
            filtered_df = filtered_df[filtered_df['arrival_date_year'] == year]
        
        if month:
            filtered_df = filtered_df[filtered_df['arrival_date_month'].str.lower() == month.lower()]
            
        # Hotel type filter
        if 'resort' in query.lower():
            filtered_df = filtered_df[filtered_df['hotel'] == 'Resort Hotel']
        elif 'city' in query.lower():
            filtered_df = filtered_df[filtered_df['hotel'] == 'City Hotel']
            
        return filtered_df
    
    def analyze_data(self, filtered_df: pd.DataFrame, query: str, use_whole_dataset: bool = True) -> Dict[str, Any]:
        """
        Analyze the filtered dataset based on the query.
        
        Args:
            filtered_df: The filtered DataFrame to analyze
            query: The user query
            use_whole_dataset: Whether to use the whole dataset for analysis after identifying key filters
        """
        # Identify the filters we want to apply to the whole dataset
        filters = {}
        
        # Extract key filters from the filtered dataframe's conditions
        if len(filtered_df) > 0:
            # Extract year filter if present and consistent
            if len(filtered_df['arrival_date_year'].unique()) == 1:
                filters['year'] = filtered_df['arrival_date_year'].iloc[0]
            
            # Extract month filter if present and consistent
            if len(filtered_df['arrival_date_month'].unique()) == 1:
                filters['month'] = filtered_df['arrival_date_month'].iloc[0]
            
            # Extract country filter if present and consistent
            if len(filtered_df['country'].unique()) == 1:
                filters['country'] = filtered_df['country'].iloc[0]
            
            # Extract hotel type filter if present and consistent
            if len(filtered_df['hotel'].unique()) == 1:
                filters['hotel'] = filtered_df['hotel'].iloc[0]
        
        # Apply the identified filters to the whole dataset if requested
        if use_whole_dataset:
            analysis_df = self.df.copy()
            
            # Apply each identified filter to the whole dataset
            if 'year' in filters:
                analysis_df = analysis_df[analysis_df['arrival_date_year'] == filters['year']]
            
            if 'month' in filters:
                analysis_df = analysis_df[analysis_df['arrival_date_month'] == filters['month']]
            
            if 'country' in filters:
                analysis_df = analysis_df[analysis_df['country'] == filters['country']]
            
            if 'hotel' in filters:
                analysis_df = analysis_df[analysis_df['hotel'] == filters['hotel']]
        else:
            # Use the filtered dataset directly
            analysis_df = filtered_df
        
        # Perform analysis on the dataset
        results = {}
        
        # Add dataset stats
        results['total_records'] = len(analysis_df)
        results['filtered_records'] = len(filtered_df)
        
        # Basic stats on analyzed data
        if len(analysis_df) > 0:
            results['hotel_distribution'] = analysis_df['hotel'].value_counts().to_dict()
            
            # Check for revenue calculation
            if 'revenue' in query.lower() or 'income' in query.lower():
                total_revenue = analysis_df['revenue'].sum()
                avg_revenue = analysis_df['revenue'].mean()
                results['total_revenue'] = total_revenue
                results['avg_revenue'] = avg_revenue
                
                # Revenue by hotel type
                results['revenue_by_hotel'] = analysis_df.groupby('hotel')['revenue'].sum().to_dict()
                
            # Check for cancellation analysis
            if 'cancel' in query.lower():
                total_bookings = len(analysis_df)
                canceled = analysis_df['is_canceled'].sum()
                cancellation_rate = canceled / total_bookings if total_bookings > 0 else 0
                results['total_bookings'] = total_bookings
                results['canceled_bookings'] = canceled
                results['cancellation_rate'] = cancellation_rate
                
                # Get cancellation by location
                if 'country' in query.lower() or 'location' in query.lower():
                    cancel_by_country = analysis_df.groupby('country')['is_canceled'].agg(['sum', 'count'])
                    cancel_by_country['rate'] = cancel_by_country['sum'] / cancel_by_country['count']
                    results['cancellation_by_country'] = cancel_by_country.sort_values('sum', ascending=False).head(10).to_dict()
            
            # Check for price/ADR analysis
            if 'price' in query.lower() or 'adr' in query.lower() or 'rate' in query.lower():
                avg_price = analysis_df['adr'].mean()
                median_price = analysis_df['adr'].median()
                min_price = analysis_df['adr'].min()
                max_price = analysis_df['adr'].max()
                results['avg_price'] = avg_price
                results['median_price'] = median_price
                results['min_price'] = min_price
                results['max_price'] = max_price
                
                # ADR by hotel type
                results['adr_by_hotel'] = analysis_df.groupby('hotel')['adr'].mean().to_dict()
                
            # Check for seasonality analysis
            if 'season' in query.lower() or 'month' in query.lower():
                bookings_by_month = analysis_df.groupby('arrival_date_month')['is_canceled'].count()
                # Order months chronologically
                month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                              'July', 'August', 'September', 'October', 'November', 'December']
                bookings_by_month = bookings_by_month.reindex(month_order)
                results['bookings_by_month'] = bookings_by_month.to_dict()
                
                # Revenue by month
                revenue_by_month = analysis_df.groupby('arrival_date_month')['revenue'].sum()
                revenue_by_month = revenue_by_month.reindex(month_order)
                results['revenue_by_month'] = revenue_by_month.to_dict()
                
            # Check for guest analysis
            if 'guest' in query.lower() or 'customer' in query.lower():
                avg_guests = analysis_df['total_guests'].mean()
                results['avg_guests_per_booking'] = avg_guests
                
                # Guests by market segment
                results['guests_by_segment'] = analysis_df.groupby('market_segment')['total_guests'].sum().sort_values(ascending=False).to_dict()
            
            # Analysis for repeated guests
            if 'repeat' in query.lower() or 'return' in query.lower() or 'loyal' in query.lower():
                repeat_guests = analysis_df['is_repeated_guest'].sum()
                repeat_rate = repeat_guests / len(analysis_df) if len(analysis_df) > 0 else 0
                results['repeat_guests'] = repeat_guests
                results['repeat_rate'] = repeat_rate
                
                # Previous stays and cancellations
                results['avg_previous_stays'] = analysis_df['previous_bookings_not_canceled'].mean()
                results['avg_previous_cancellations'] = analysis_df['previous_cancellations'].mean()
            
            # Analysis for room types
            if 'room' in query.lower() or 'type' in query.lower():
                room_type_distribution = analysis_df['reserved_room_type'].value_counts().to_dict()
                room_change_rate = (analysis_df['reserved_room_type'] != analysis_df['assigned_room_type']).mean()
                results['room_type_distribution'] = room_type_distribution
                results['room_change_rate'] = room_change_rate
            
            # Analysis for booking changes
            if 'change' in query.lower() or 'modify' in query.lower():
                bookings_with_changes = (analysis_df['booking_changes'] > 0).sum()
                change_rate = bookings_with_changes / len(analysis_df) if len(analysis_df) > 0 else 0
                avg_changes = analysis_df['booking_changes'].mean()
                results['bookings_with_changes'] = bookings_with_changes
                results['change_rate'] = change_rate
                results['avg_changes'] = avg_changes
            
            # Analysis for special requests
            if 'special' in query.lower() or 'request' in query.lower():
                bookings_with_requests = (analysis_df['total_of_special_requests'] > 0).sum()
                request_rate = bookings_with_requests / len(analysis_df) if len(analysis_df) > 0 else 0
                avg_requests = analysis_df['total_of_special_requests'].mean()
                results['bookings_with_requests'] = bookings_with_requests
                results['request_rate'] = request_rate
                results['avg_requests'] = avg_requests
            
            # Analysis for waitlist
            if 'wait' in query.lower() or 'waitlist' in query.lower():
                bookings_waitlisted = (analysis_df['days_in_waiting_list'] > 0).sum()
                waitlist_rate = bookings_waitlisted / len(analysis_df) if len(analysis_df) > 0 else 0
                avg_waitlist_days = analysis_df['days_in_waiting_list'].mean()
                results['bookings_waitlisted'] = bookings_waitlisted
                results['waitlist_rate'] = waitlist_rate
                results['avg_waitlist_days'] = avg_waitlist_days
                results['waitlist_distribution'] = analysis_df['waitlist_category'].value_counts().to_dict()
            
            # Analysis for meal preferences
            if 'meal' in query.lower() or 'food' in query.lower() or 'board' in query.lower():
                meal_distribution = analysis_df['meal'].value_counts().to_dict()
                results['meal_distribution'] = meal_distribution
                
                # Meal preferences by country (top 5)
                meal_by_country = analysis_df.groupby(['country', 'meal']).size().unstack(fill_value=0)
                top_countries = meal_by_country.sum(axis=1).sort_values(ascending=False).head(5).index
                results['meal_preferences_by_country'] = meal_by_country.loc[top_countries].to_dict()
            
             # Analysis for distribution channels
            if 'channel' in query.lower() or 'book' in query.lower() or 'segment' in query.lower():
               channel_distribution = analysis_df['distribution_channel'].value_counts().to_dict()
               segment_distribution = analysis_df['market_segment'].value_counts().to_dict()
               results['channel_distribution'] = channel_distribution
               results['segment_distribution'] = segment_distribution
               
            # Revenue by channel
            if 'revenue' in analysis_df.columns:
                results['revenue_by_channel'] = analysis_df.groupby('distribution_channel')['revenue'].sum().to_dict()
                results['revenue_by_segment'] = analysis_df.groupby('market_segment')['revenue'].sum().to_dict()
           
            # Analysis for deposit types
            if 'deposit' in query.lower() or 'payment' in query.lower():
               deposit_distribution = analysis_df['deposit_type'].value_counts().to_dict()
               results['deposit_distribution'] = deposit_distribution
               
               # Cancellation by deposit type
               cancel_by_deposit = analysis_df.groupby('deposit_type')['is_canceled'].agg(['sum', 'count'])
               cancel_by_deposit['rate'] = cancel_by_deposit['sum'] / cancel_by_deposit['count']
               results['cancellation_by_deposit'] = cancel_by_deposit.to_dict()
           
            # Analysis for weekend vs weekday stays
            if 'weekend' in query.lower() or 'weekday' in query.lower() or 'week' in query.lower():
               avg_weekend_nights = analysis_df['stays_in_weekend_nights'].mean()
               avg_week_nights = analysis_df['stays_in_week_nights'].mean()
               weekend_ratio = avg_weekend_nights / (avg_weekend_nights + avg_week_nights) if (avg_weekend_nights + avg_week_nights) > 0 else 0
               results['avg_weekend_nights'] = avg_weekend_nights
               results['avg_week_nights'] = avg_week_nights
               results['weekend_ratio'] = weekend_ratio
               
               # Stays with only weekend nights
               weekend_only = (analysis_df['stays_in_weekend_nights'] > 0) & (analysis_df['stays_in_week_nights'] == 0)
               results['weekend_only_stays'] = weekend_only.sum()
               results['weekend_only_rate'] = weekend_only.mean()
           
            # Analysis for agents and companies
            if 'agent' in query.lower() or 'company' in query.lower():
               # Top agents by number of bookings
               top_agents = analysis_df['agent'].value_counts().head(10).to_dict()
               results['top_agents'] = top_agents
               
            # Top companies by number of bookings
            if 'company' in analysis_df.columns:
                   top_companies = analysis_df['company'].value_counts().head(10).to_dict()
                   results['top_companies'] = top_companies
           
            # Analysis for stay duration
            if 'stay' in query.lower() or 'duration' in query.lower() or 'length' in query.lower():
               avg_stay = analysis_df['total_nights'].mean()
               median_stay = analysis_df['total_nights'].median()
               max_stay = analysis_df['total_nights'].max()
               results['avg_stay'] = avg_stay
               results['median_stay'] = median_stay
               results['max_stay'] = max_stay
               
               # Stay duration by hotel type
               results['stay_by_hotel'] = analysis_df.groupby('hotel')['total_nights'].mean().to_dict()
        return results
    
    def generate_answer(self, query: str) -> str:
        """Generate an answer using semantic search and whole dataset analysis."""
        logger.info(f"Processing query: {query}")
        start_time = datetime.now()
        
        # Refresh the dataframe from the database
        self.df = self.get_all_bookings()
        
        # Record the query in history
        query_type = "question"
        
        # First, get semantically similar records using FAISS
        if self.faiss_available:
            semantic_indices = self.faiss_semantic_search(query, top_k=min(500, len(self.df)))
            logger.info(f"Found {len(semantic_indices)} semantically relevant records")
        else:
            semantic_indices = None
            logger.warning("FAISS not available, using traditional filtering")
        
        # Filter relevant data based on query and semantic search
        filtered_data = self.filter_data_for_query(query, semantic_indices)
        
        if len(filtered_data) == 0:
            response = "I couldn't find any data matching your query criteria."
            self.record_query(query, query_type, response, (datetime.now() - start_time).total_seconds())
            return response
        
        # Analyze the filtered data, using the whole dataset with the same filters
        analysis_results = self.analyze_data(filtered_data, query, use_whole_dataset=True)
        
        # Prepare context with dataset summary
        context = f"Analysis based on {analysis_results.get('total_records', 0)} hotel booking records"
        if analysis_results.get('filtered_records', 0) != analysis_results.get('total_records', 0):
            context += f" (initial filter matched {analysis_results.get('filtered_records', 0)} records)"
        context += ":\n\n"
        
        # Add hotel distribution
        if 'hotel_distribution' in analysis_results:
            context += "Hotel types in data:\n"
            for hotel, count in analysis_results['hotel_distribution'].items():
                context += f"- {hotel}: {count} bookings\n"
            context += "\n"

        # Add room type context if relevant
        if 'room' in query.lower() or 'type' in query.lower():
            if 'room_type_distribution' in analysis_results:
                context += "Room type distribution:\n"
                for room_type, count in analysis_results['room_type_distribution'].items():
                    context += f"- Type {room_type}: {count} bookings\n"
                
                if 'room_change_rate' in analysis_results:
                    context += f"- Room change rate: {analysis_results['room_change_rate']*100:.1f}%\n"
                context += "\n"
        
        # Add special requests context if relevant
        if 'special' in query.lower() or 'request' in query.lower():
            if 'bookings_with_requests' in analysis_results:
                context += "Special requests analysis:\n"
                context += f"- Bookings with special requests: {analysis_results['bookings_with_requests']}\n"
                context += f"- Special request rate: {analysis_results['request_rate']*100:.1f}%\n"
                context += f"- Average requests per booking: {analysis_results['avg_requests']:.2f}\n"
                context += "\n"

        # Add all analysis results
        context += "Analysis results:\n"
        for key, value in analysis_results.items():
            if key in ['total_records', 'filtered_records', 'hotel_distribution']:
                continue  # Already included above
                
            if isinstance(value, dict):
                context += f"{key}:\n"
                for k, v in value.items():
                    if isinstance(v, float):
                        context += f"  {k}: {v:.2f}\n"
                    else:
                        context += f"  {k}: {v}\n"
            elif isinstance(value, float):
                context += f"{key}: {value:.2f}\n"
            else:
                context += f"{key}: {value}\n"
        
        # Generate answer with LLM (if available)
        if self.llm_available:
            # Create prompt for LLM
            prompt = f"""
            <s>[INST] You are a hotel data analyst assistant. Use only the information provided in the context to answer the question. Be specific, concise, and provide numerical results when available.

            Context:
            {context}
            
            Question: {query}
            
            Answer: [/INST]
            """
            
            answer = self.pipe(prompt)[0]['generated_text']
            response = answer.split("[/INST]")[-1].strip()
        else:
            # Simple response based on analysis if LLM not available
            response = f"Based on the analysis of {analysis_results.get('total_records', 0)} hotel bookings:\n"
            if 'cancel' in query.lower() and 'canceled_bookings' in analysis_results:
                response += f"- Total bookings: {analysis_results.get('total_bookings')}\n"
                response += f"- Cancelled bookings: {analysis_results.get('canceled_bookings')}\n"
                response += f"- Cancellation rate: {analysis_results.get('cancellation_rate', 0)*100:.2f}%\n"
            
            if 'revenue' in query.lower() and 'total_revenue' in analysis_results:
                response += f"- Total revenue: ${analysis_results.get('total_revenue'):.2f}\n"
                response += f"- Average revenue per booking: ${analysis_results.get('avg_revenue'):.2f}\n"
        
        # Record the query and response
        processing_time = (datetime.now() - start_time).total_seconds()
        self.record_query(query, query_type, response, processing_time)
        
        return response
    
    def record_query(self, query: str, query_type: str, response: str, processing_time: float):
        """Record a query in the query history."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO query_history (timestamp, query, query_type, response, processing_time) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), query, query_type, response, processing_time)
            )
            
            conn.commit()
            logger.info(f"Recorded {query_type} query in history: '{query[:50]}...'")
            
        except Exception as e:
            logger.error(f"Error recording query: {str(e)}")
    
    def get_query_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query history."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM query_history ORDER BY timestamp DESC LIMIT ?", 
                (limit,)
            )
            
            columns = [desc[0] for desc in cursor.description]
            query_history = []
            
            for row in cursor.fetchall():
                query_entry = dict(zip(columns, row))
                query_history.append(query_entry)
                
            return query_history
            
        except Exception as e:
            logger.error(f"Error retrieving query history: {str(e)}")
            return []
    
    def add_booking(self, booking_data: Dict[str, Any]) -> int:
        """
        Add a new booking to the database.
        
        Args:
            booking_data: Dictionary containing booking information
            
        Returns:
            Inserted record ID or -1 if error
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Add calculated fields if not present
            if 'total_nights' not in booking_data:
                weekend_nights = booking_data.get('stays_in_weekend_nights', 0)
                week_nights = booking_data.get('stays_in_week_nights', 0)
                booking_data['total_nights'] = weekend_nights + week_nights
            
            if 'total_guests' not in booking_data:
                adults = booking_data.get('adults', 0)
                children = booking_data.get('children', 0)
                babies = booking_data.get('babies', 0)
                booking_data['total_guests'] = adults + children + babies
            
            if 'revenue' not in booking_data:
                adr = booking_data.get('adr', 0)
                booking_data['revenue'] = adr * booking_data['total_nights']
            
            # Add timestamp
            booking_data['last_updated'] = datetime.now().isoformat()
            
            # Insert into database
            columns = ', '.join(booking_data.keys())
            placeholders = ', '.join(['?'] * len(booking_data))
            query = f"INSERT INTO bookings ({columns}) VALUES ({placeholders})"
            cursor.execute(query, list(booking_data.values()))
            
            # Log the update
            cursor.execute(
                "INSERT INTO update_log (timestamp, operation, record_count, details) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), "add_booking", 1, f"Added booking for {booking_data.get('country', 'Unknown')}")
            )
            
            conn.commit()
            
            # Get the ID of the inserted record
            booking_id = cursor.lastrowid
            logger.info(f"Added new booking with ID {booking_id}")
            
            # Update FAISS index if available
            if hasattr(self, 'embedding_model') and self.embedding_model is not None:
                self.update_vector_index()
                
            return booking_id
            
        except Exception as e:
            logger.error(f"Error adding booking: {str(e)}")
            conn = self.get_connection()
            if conn:
                conn.rollback()
            return -1
    
    def update_booking(self, booking_id: int, booking_data: Dict[str, Any]) -> bool:
        """
        Update an existing booking.
        
        Args:
            booking_id: ID of the booking to update
            booking_data: Dictionary containing updated booking information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Add timestamp
            booking_data['last_updated'] = datetime.now().isoformat()
            
            # Prepare update statement
            set_clause = ', '.join([f"{k} = ?" for k in booking_data.keys()])
            values = list(booking_data.values())
            values.append(booking_id)
            
            # Update database
            query = f"UPDATE bookings SET {set_clause} WHERE id = ?"
            cursor.execute(query, values)
            
            # Check if the update affected any rows
            if cursor.rowcount == 0:
                logger.warning(f"No booking found with ID {booking_id}")
                conn.rollback()
                return False
            
            # Log the update
            cursor.execute(
                "INSERT INTO update_log (timestamp, operation, record_count, details) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), "update_booking", 1, f"Updated booking ID {booking_id}")
            )
            
            conn.commit()
            logger.info(f"Updated booking with ID {booking_id}")
            
            # Update FAISS index if available
            if hasattr(self, 'embedding_model') and self.embedding_model is not None:
                self.update_vector_index()
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating booking: {str(e)}")
            conn = self.get_connection()
            if conn:
                conn.rollback()
            return False
    
    def delete_booking(self, booking_id: int) -> bool:
        """
        Delete a booking from the database.
        
        Args:
            booking_id: ID of the booking to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM bookings WHERE id = ?", (booking_id,))
            
            # Check if the delete affected any rows
            if cursor.rowcount == 0:
                logger.warning(f"No booking found with ID {booking_id}")
                conn.rollback()
                return False
            
            # Also delete from embeddings table if it exists
            cursor.execute("DELETE FROM embeddings WHERE booking_id = ?", (booking_id,))
            
            # Log the update
            cursor.execute(
                "INSERT INTO update_log (timestamp, operation, record_count, details) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), "delete_booking", 1, f"Deleted booking ID {booking_id}")
            )
            
            conn.commit()
            logger.info(f"Deleted booking with ID {booking_id}")
            
            # Update FAISS index if available
            if hasattr(self, 'embedding_model') and self.embedding_model is not None:
                self.update_vector_index()
                
            return True
            
        except Exception as e:
            logger.error(f"Error deleting booking: {str(e)}")
            conn = self.get_connection()
            if conn:
                conn.rollback()
            return False
    
    def get_all_bookings(self) -> pd.DataFrame:
        """
        Retrieve all bookings from the database.
        
        Returns:
            DataFrame containing all bookings
        """
        try:
            conn = self.get_connection()
            
            query = "SELECT * FROM bookings"
            df = pd.read_sql_query(query, conn)
            logger.info(f"Retrieved {len(df)} bookings from database")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving bookings: {str(e)}")
            return pd.DataFrame()
    
    def query_bookings(self, conditions: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Query bookings with specific conditions.
        
        Args:
            conditions: Dictionary mapping column names to values
            
        Returns:
            DataFrame containing matching bookings
        """
        try:
            conn = self.get_connection()
            
            query = "SELECT * FROM bookings"
            params = []
            
            if conditions and len(conditions) > 0:
                where_clauses = []
                for key, value in conditions.items():
                    if isinstance(value, dict):
                        # Handle comparison operators (gt, lt, eq)
                        for op, val in value.items():
                            if op == "gt":
                                where_clauses.append(f"{key} > ?")
                                params.append(val)
                            elif op == "lt":
                                where_clauses.append(f"{key} < ?")
                                params.append(val)
                            elif op == "eq":
                                where_clauses.append(f"{key} = ?")
                                params.append(val)
                    else:
                        where_clauses.append(f"{key} = ?")
                        params.append(value)
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
            
            df = pd.read_sql_query(query, conn, params=params)
            logger.info(f"Query returned {len(df)} bookings")
            return df
            
        except Exception as e:
            logger.error(f"Error querying bookings: {str(e)}")
            return pd.DataFrame()
    
    def get_update_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent database update log entries.
        
        Args:
            limit: Maximum number of log entries to return
            
        Returns:
            List of log entry dictionaries
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM update_log ORDER BY timestamp DESC LIMIT ?", 
                (limit,)
            )
            columns = [desc[0] for desc in cursor.description]
            log_entries = []
            
            for row in cursor.fetchall():
                log_entry = dict(zip(columns, row))
                log_entries.append(log_entry)
                
            return log_entries
            
        except Exception as e:
            logger.error(f"Error retrieving update log: {str(e)}")
            return []
    
    def get_db_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM bookings")
            total_bookings = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM bookings WHERE is_canceled = 1")
            canceled_bookings = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT country) FROM bookings")
            distinct_countries = cursor.fetchone()[0]
            
            cursor.execute("SELECT hotel, COUNT(*) FROM bookings GROUP BY hotel")
            hotel_distribution = {hotel: count for hotel, count in cursor.fetchall()}
            
            cursor.execute("SELECT arrival_date_year, COUNT(*) FROM bookings GROUP BY arrival_date_year")
            year_distribution = {year: count for year, count in cursor.fetchall()}
            
            cursor.execute("SELECT MAX(last_updated) FROM bookings")
            last_update = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM update_log")
            total_updates = cursor.fetchone()[0]
            
            return {
                "total_bookings": total_bookings,
                "canceled_bookings": canceled_bookings,
                "cancellation_rate": canceled_bookings / total_bookings if total_bookings > 0 else 0,
                "distinct_countries": distinct_countries,
                "hotel_distribution": hotel_distribution,
                "year_distribution": year_distribution,
                "last_update": last_update,
                "total_updates": total_updates
            }
            
        except Exception as e:
            logger.error(f"Error retrieving database statistics: {str(e)}")
            return {}
    
    def update_vector_index(self):
        """
        Update the FAISS vector index after database changes.
        This regenerates the index from the current database state.
        """
        try:
            # Refresh dataframe from database
            self.df = self.get_all_bookings()
            
            # Clean data
            self.df = self.clean_data_for_rag(self.df)
            
            # Generate new text descriptions
            self.text_descriptions = self.generate_text_descriptions()
            
            # Generate new embeddings
            logger.info("Generating updated embeddings for dataset...")
            embeddings = self.embedding_model.encode(self.text_descriptions, show_progress_bar=True)
            
            # Ensure embeddings are normalized for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create new FAISS index
            embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity with normalized vectors
            self.index.add(embeddings)
            
            # Update original indices
            self.original_indices = np.arange(len(self.df))
            
            logger.info(f"FAISS index updated with {self.index.ntotal} vectors")
            self.faiss_available = True
            
        except Exception as e:
            logger.error(f"Error updating vector index: {str(e)}")
            logger.warning("FAISS index update failed, using traditional filtering")
            self.faiss_available = False
    
    def semantic_search(self, query: str, top_k: int = 20) -> pd.DataFrame:
        """
        Perform a semantic search and return matching bookings.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            DataFrame containing semantically matching bookings
        """
        try:
            if not self.faiss_available:
                logger.warning("FAISS not available, performing regular query instead")
                return self.filter_data_for_query(query)
            
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search for similar records
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.df)))
            
            # Map to original dataframe indices
            result_indices = self.original_indices[indices[0]]
            
            # Return matching records
            result_df = self.df.iloc[result_indices].copy()
            
            # Add semantic score
            result_df['semantic_score'] = scores[0]
            
            logger.info(f"Semantic search returned {len(result_df)} results")
            return result_df
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return pd.DataFrame()
    
    def analyze_query(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze booking data based on a natural language query or filters.
        
        Args:
            query: Natural language query
            filters: Optional dictionary of specific filters to apply
            
        Returns:
            Dictionary with analysis results
        """
        start_time = datetime.now()
        query_type = "analytics"
        
        try:
            # Refresh the dataframe from the database
            self.df = self.get_all_bookings()
            
            # Apply filters if provided
            if filters and len(filters) > 0:
                filtered_df = self.df.copy()
                for key, value in filters.items():
                    if key in filtered_df.columns:
                        if isinstance(value, dict):
                            # Handle comparison operators
                            for op, val in value.items():
                                if op == "gt":
                                    filtered_df = filtered_df[filtered_df[key] > val]
                                elif op == "lt":
                                    filtered_df = filtered_df[filtered_df[key] < val]
                                elif op == "eq":
                                    filtered_df = filtered_df[filtered_df[key] == val]
                        else:
                            # Direct equality comparison
                            filtered_df = filtered_df[filtered_df[key] == value]
                
                # Analyze with the provided filters
                analysis_results = self.analyze_data(filtered_df, query, use_whole_dataset=False)
            else:
                # Use semantic search and natural language filtering
                if self.faiss_available:
                    semantic_indices = self.faiss_semantic_search(query, top_k=min(500, len(self.df)))
                    logger.info(f"Found {len(semantic_indices)} semantically relevant records")
                else:
                    semantic_indices = None
                    logger.warning("FAISS not available, using traditional filtering")
                
                # Filter relevant data based on query and semantic search
                filtered_data = self.filter_data_for_query(query, semantic_indices)
                
                if len(filtered_data) == 0:
                    return {"error": "No data found matching the query criteria."}
                
                # Analyze the filtered data
                analysis_results = self.analyze_data(filtered_data, query, use_whole_dataset=True)
            
            # Record the query
            processing_time = (datetime.now() - start_time).total_seconds()
            self.record_query(query, query_type, str(analysis_results), processing_time)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return {"error": str(e)}
    
    def close(self):
        """Close the database connection for the current thread."""
        if hasattr(thread_local, "conn"):
            thread_local.conn.close()
            delattr(thread_local, "conn")
            logger.info(f"Database connection closed for thread {threading.get_ident()}")
            
        # Free up model resources
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("Released model resources")