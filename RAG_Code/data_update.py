import requests
import json
import random
import time
import pandas as pd
from datetime import datetime, timedelta
import argparse
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API endpoint
API_BASE_URL = "http://localhost:8008"

def check_api_health():
    """Check if the API is up and running."""
    try:
        response = requests.get(f'{API_BASE_URL}/health')
        return response.status_code == 200
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return False

def generate_random_booking():
    """Generate a random booking entry for testing."""
    # Hotel type
    hotel_type = random.choice(['Resort Hotel', 'City Hotel'])
    
    # Dates
    today = datetime.now()
    arrival_year = today.year
    arrival_month = random.choice([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    arrival_day = random.randint(1, 28)
    arrival_week = (arrival_day // 7) + 1
    
    # Nights
    weekend_nights = random.randint(0, 3)
    week_nights = random.randint(1, 7)
    total_nights = weekend_nights + week_nights
    
    # Guests
    adults = random.randint(1, 4)
    children = random.randint(0, 2)
    babies = random.randint(0, 1)
    total_guests = adults + children + babies
    
    # Country (random from top countries)
    countries = ['PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'USA', 'BRA', 'NLD', 'CHE', 'BEL', 'AUT', 'CAN', 'AUS']
    country = random.choice(countries)
    
    # Market segment and distribution channel
    market_segment = random.choice(['Direct', 'Online TA', 'Offline TA/TO', 'Corporate', 'Groups'])
    distribution_channel = random.choice(['Direct', 'TA/TO', 'Corporate'])
    
    # Room type
    room_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    reserved_room_type = random.choice(room_types)
    assigned_room_type = reserved_room_type if random.random() < 0.8 else random.choice(room_types)
    
    # Meal
    meal = random.choice(['BB', 'HB', 'FB', 'SC'])
    
    # Cancellation
    is_canceled = 1 if random.random() < 0.3 else 0
    
    # ADR (Average Daily Rate)
    adr_base = 80 if hotel_type == 'City Hotel' else 100
    adr_variation = random.uniform(0.5, 2.0)
    adr = adr_base * adr_variation
    
    # Generate booking
    booking = {
        "hotel": hotel_type,
        "is_canceled": is_canceled,
        "lead_time": random.randint(1, 365),
        "arrival_date_year": arrival_year,
        "arrival_date_month": arrival_month,
        "arrival_date_week_number": arrival_week,
        "arrival_date_day_of_month": arrival_day,
        "stays_in_weekend_nights": weekend_nights,
        "stays_in_week_nights": week_nights,
        "adults": adults,
        "children": float(children),
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": 1 if random.random() < 0.1 else 0,
        "previous_cancellations": random.randint(0, 2),
        "previous_bookings_not_canceled": random.randint(0, 3),
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "booking_changes": random.randint(0, 3),
        "deposit_type": random.choice(['No Deposit', 'Non Refund', 'Refundable']),
        "agent": float(random.randint(1, 300)),
        "company": float(random.randint(1, 200)) if random.random() < 0.2 else float(-1),
        "days_in_waiting_list": random.randint(0, 50),
        "customer_type": random.choice(['Transient', 'Transient-Party', 'Contract', 'Group']),
        "adr": adr,
        "required_car_parking_spaces": random.randint(0, 2),
        "total_of_special_requests": random.randint(0, 5),
        "reservation_status": "Check-Out" if not is_canceled else "Canceled",
        "reservation_status_date": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
        "has_previous_stay": 1 if booking["previous_bookings_not_canceled"] > 0 else 0,
        "has_previous_cancellations": 1 if booking["previous_cancellations"] > 0 else 0,
        "has_booking_changes": 0, 
        "requires_parking": 1 if booking["required_car_parking_spaces"] > 0 else 0,
        "has_special_requests": 1 if booking["total_of_special_requests"] > 0 else 0,
        "waitlist_category": get_waitlist_category(booking["days_in_waiting_list"])
    }
    
    # Calculate derived fields
    booking["total_nights"] = weekend_nights + week_nights
    booking["total_guests"] = adults + children + babies
    booking["revenue"] = adr * booking["total_nights"]
    
    return booking

def add_booking():
    """Add a new booking to the system."""
    booking = generate_random_booking()
    
    try:
        response = requests.post(
            f'{API_BASE_URL}/bookings',
            json={"booking": booking}
        )
        
        if response.status_code == 201:
            result = response.json()
            booking_id = result.get('booking_id')
            print(f"Successfully added booking with ID {booking_id}")
            return booking_id
        else:
            print(f"Failed to add booking: {response.text}")
            return None
    except Exception as e:
        print(f"Error adding booking: {e}")
        return None

def update_booking(booking_id):
    """Update an existing booking."""
    # Get a new random booking data
    updated_booking = generate_random_booking()
    
    # We're just updating a subset of fields for demonstration
    update_data = {
        "is_canceled": updated_booking["is_canceled"],
        "stays_in_weekend_nights": updated_booking["stays_in_weekend_nights"],
        "stays_in_week_nights": updated_booking["stays_in_week_nights"],
        "adr": updated_booking["adr"],
        "required_car_parking_spaces": updated_booking["required_car_parking_spaces"],
        "total_of_special_requests": updated_booking["total_of_special_requests"],
        "reservation_status": updated_booking["reservation_status"]
    }
    
    try:
        response = requests.put(
            f'{API_BASE_URL}/bookings/{booking_id}',
            json={"booking": update_data}
        )
        
        if response.status_code == 200:
            print(f"Successfully updated booking {booking_id}")
            return True
        else:
            print(f"Failed to update booking: {response.text}")
            return False
    except Exception as e:
        print(f"Error updating booking: {e}")
        return False

def delete_booking(booking_id):
    """Delete a booking."""
    try:
        response = requests.delete(f'{API_BASE_URL}/bookings/{booking_id}')
        
        if response.status_code == 200:
            print(f"Successfully deleted booking {booking_id}")
            return True
        else:
            print(f"Failed to delete booking: {response.text}")
            return False
    except Exception as e:
        print(f"Error deleting booking: {e}")
        return False

def batch_import(count=10):
    """Import a batch of bookings."""
    bookings = [generate_random_booking() for _ in range(count)]
    
    try:
        response = requests.post(
            f'{API_BASE_URL}/bookings/batch',
            json={"bookings": bookings}
        )
        
        if response.status_code == 201:
            result = response.json()
            success_count = result.get('success_count', 0)
            print(f"Successfully imported {success_count} of {count} bookings")
            return success_count
        else:
            print(f"Failed to import bookings: {response.text}")
            return 0
    except Exception as e:
        print(f"Error importing bookings: {e}")
        return 0

def ask_question(question):
    """Ask a question to the RAG system."""
    try:
        response = requests.post(
            f'{API_BASE_URL}/ask',
            json={"question": question}
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', '')
            processing_time = result.get('processing_time_seconds', 0)
            print(f"\nQuestion: {question}")
            print(f"Answer: {answer}")
            print(f"Processing time: {processing_time:.2f} seconds")
            return answer
        else:
            print(f"Failed to ask question: {response.text}")
            return None
    except Exception as e:
        print(f"Error asking question: {e}")
        return None

def force_refresh():
    """Force refresh of the system data."""
    try:
        response = requests.post(f'{API_BASE_URL}/refresh')
        
        if response.status_code == 200:
            result = response.json()
            record_count = result.get('record_count', 0)
            print(f"Successfully refreshed system data with {record_count} records")
            return True
        else:
            print(f"Failed to refresh system: {response.text}")
            return False
    except Exception as e:
        print(f"Error refreshing system: {e}")
        return False

def get_analytics(query=None, filters=None, metrics=None):
    """Get analytics from the system."""
    data = {}
    
    if query:
        data["query"] = query
    elif filters or metrics:
        if filters:
            data["filters"] = filters
        if metrics:
            data["metrics"] = metrics
    else:
        print("Either query or filters/metrics must be provided")
        return None
    
    try:
        response = requests.post(
            f'{API_BASE_URL}/analytics',
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            processing_time = result.get('processing_time_seconds', 0)
            print(f"\nAnalytics query: {query if query else 'Structured query'}")
            print(f"Results: {json.dumps(result.get('results', {}), indent=2)}")
            print(f"Processing time: {processing_time:.2f} seconds")
            return result.get('results', {})
        else:
            print(f"Failed to get analytics: {response.text}")
            return None
    except Exception as e:
        print(f"Error getting analytics: {e}")
        return None

def get_waitlist_category(days_in_waiting_list):
    if days_in_waiting_list == 0:
        return "No wait"
    elif 1 <= days_in_waiting_list <= 7:
        return "1-7 days"
    elif 8 <= days_in_waiting_list <= 30:
        return "8-30 days"
    else:
        return "Over 30 days"
    
def main():
    parser = argparse.ArgumentParser(description="Hotel Booking Data Update Demo")
    parser.add_argument('--action', choices=['add', 'update', 'delete', 'batch', 'ask', 'analytics', 'refresh', 'demo'], default='demo',
                      help='Action to perform (default: demo)')
    parser.add_argument('--count', type=int, default=10, help='Number of bookings for batch import (default: 10)')
    parser.add_argument('--id', type=int, help='Booking ID for update/delete operations')
    parser.add_argument('--question', type=str, help='Question to ask the system')
    parser.add_argument('--query', type=str, help='Analytics query')
    
    args = parser.parse_args()
    
    # Check if API is running
    if not check_api_health():
        print("API is not running or not responding. Please start the API server first.")
        return
    
    print("API is up and running!")
    
    if args.action == 'add':
        add_booking()
    elif args.action == 'update':
        if not args.id:
            print("Booking ID is required for update operation")
            return
        update_booking(args.id)
    elif args.action == 'delete':
        if not args.id:
            print("Booking ID is required for delete operation")
            return
        delete_booking(args.id)
    elif args.action == 'batch':
        batch_import(args.count)
    elif args.action == 'ask':
        if not args.question:
            print("Question is required for ask operation")
            return
        ask_question(args.question)
    elif args.action == 'analytics':
        if not args.query:
            print("Query is required for analytics operation")
            return
        get_analytics(query=args.query)
    elif args.action == 'refresh':
        force_refresh()
    elif args.action == 'demo':
        print("Running demo sequence...")
        
        # 1. Add a few bookings
        print("\n1. Adding 3 individual bookings...")
        booking_ids = []
        for _ in range(3):
            booking_id = add_booking()
            if booking_id:
                booking_ids.append(booking_id)
            time.sleep(1)
        
        # 2. Ask a question about the data
        print("\n2. Asking question about the data...")
        ask_question("How many hotel bookings do we have so far?")
        time.sleep(2)
        
        # 3. Batch import
        print("\n3. Batch importing 5 bookings...")
        batch_import(5)
        time.sleep(2)
        
        # 4. Ask another question
        print("\n4. Asking a new question after batch import...")
        ask_question("What's the current cancellation rate?")
        time.sleep(2)
        
        # 5. Run analytics
        print("\n5. Running analytics query...")
        get_analytics(query="Show me revenue comparison between resort and city hotels")
        time.sleep(2)
        
        # 6. Update a booking if we have IDs
        if booking_ids:
            print(f"\n6. Updating booking {booking_ids[0]}...")
            update_booking(booking_ids[0])
            time.sleep(2)
            
            # 7. Delete a booking
            print(f"\n7. Deleting booking {booking_ids[-1]}...")
            delete_booking(booking_ids[-1])
            time.sleep(2)
        
        # 8. Force refresh
        print("\n8. Forcing system data refresh...")
        force_refresh()
        time.sleep(2)
        
        # 9. Final question
        print("\n9. Final question after all updates...")
        ask_question("What's the revenue comparison between city and resort hotels?")
        
        print("\nDemo completed!")

if __name__ == "__main__":
    main()