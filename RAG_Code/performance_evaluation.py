import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from typing import List, Dict, Any
import statistics
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8008")

class APIPerformanceEvaluator:
    def __init__(self, api_url: str = "http://localhost:8008"):
        """Initialize the API performance evaluator."""
        self.api_url = api_url
        self.test_questions = [
            "What was the total revenue for July 2017?",
            "What's the cancellation rate for resort hotels?",
            "Which country has the highest number of bookings?",
            "How many bookings include special requests?",
            "Which countries had the highest booking cancellations?",
            "What percentage of bookings are from repeat guests?",
            "What is the average price of hotel bookings?"
        ]
        
        # Pre-defined expected answers (simplified for demonstration)
        self.expected_answers = {
            "What was the total revenue for July 2017?": ["revenue", "july", "2017", "$"],
            "What's the cancellation rate for resort hotels?": ["cancellation", "rate", "resort", "percent", "%"],
            "Which country has the highest number of bookings?": ["country", "bookings", "highest"],
            "How many bookings include special requests?": ["special requests", "bookings"],
            "Which locations had the highest booking cancellations?": ["location", "cancellations", "most", "highest"],
            "What percentage of bookings are from repeat guests?": ["repeat guests", "percentage", "bookings"],
            "What is the average price of hotel bookings?": ["average", "price", "bookings"],
        }
        
        self.results = {
            "response_times": [],
            "accuracy_scores": [],
            "errors": [],
            "detailed_results": []
        }
    
    def check_health(self) -> Dict[str, Any]:
        """Check if the API is healthy and ready for testing."""
        try:
            response = requests.get(f"{self.api_url}/health")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def evaluate_accuracy(self, question: str, answer: str) -> float:
        """
        Evaluate the accuracy of the answer based on expected keywords.
        Simple implementation - counts how many expected keywords are present.
        Returns a score between 0 and 1.
        """
        if question not in self.expected_answers:
            return 0.0
        
        expected_keywords = self.expected_answers[question]
        keywords_found = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
        accuracy = keywords_found / len(expected_keywords)
        
        return accuracy
    
    def test_ask_endpoint(self, questions: List[str] = None) -> List[Dict[str, Any]]:
        """Test the /ask endpoint with a list of questions."""
        if questions is None:
            questions = self.test_questions
        
        results = []
        
        for question in questions:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_url}/ask",
                    json={"question": question}
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    accuracy = self.evaluate_accuracy(question, answer)
                    
                    result = {
                        "question": question,
                        "response_time": response_time,
                        "status_code": response.status_code,
                        "answer": answer,
                        "accuracy": accuracy
                    }
                    
                    self.results["response_times"].append(response_time)
                    self.results["accuracy_scores"].append(accuracy)
                else:
                    result = {
                        "question": question,
                        "response_time": response_time,
                        "status_code": response.status_code,
                        "error": response.text
                    }
                    self.results["errors"].append(result)
                
                results.append(result)
                self.results["detailed_results"].append(result)
                
                # Wait a bit to avoid overwhelming the API
                time.sleep(0.5)
                
            except Exception as e:
                error_result = {
                    "question": question,
                    "error": str(e),
                    "response_time": time.time() - start_time
                }
                results.append(error_result)
                self.results["errors"].append(error_result)
                self.results["detailed_results"].append(error_result)
                
        return results
    
    def test_analytics_endpoint(self, queries: List[str] = None) -> List[Dict[str, Any]]:
        """Test the /analytics endpoint with sample queries."""
        if queries is None:
            queries = [
                "Show me the revenue trends for 2017",
                "What's the monthly cancellation rate for resort hotels?",
                "Compare the average daily rate between city and resort hotels",
                "Which countries contribute the most to our revenue?",
                "What's the typical stay duration for guests with children?"
            ]
        
        results = []
        
        for query in queries:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_url}/analytics",
                    json={"query": query}
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    result = {
                        "query": query,
                        "response_time": response_time,
                        "status_code": response.status_code,
                        "processing_time": data.get("processing_time_seconds", 0),
                        "results_size": len(str(data.get("results", {})))
                    }
                    
                    self.results["response_times"].append(response_time)
                else:
                    result = {
                        "query": query,
                        "response_time": response_time,
                        "status_code": response.status_code,
                        "error": response.text
                    }
                    self.results["errors"].append(result)
                
                results.append(result)
                self.results["detailed_results"].append(result)
                
                # Wait a bit to avoid overwhelming the API
                time.sleep(1)
                
            except Exception as e:
                error_result = {
                    "query": query,
                    "error": str(e),
                    "response_time": time.time() - start_time
                }
                results.append(error_result)
                self.results["errors"].append(error_result)
                self.results["detailed_results"].append(error_result)
                
        return results
    
    def test_specific_analytics(self) -> List[Dict[str, Any]]:
        """Test specific analytics with filters and metrics aligned with test questions."""
        test_cases = [
            {
                # For "What was the total revenue for July 2017?"
                "filters": {"arrival_date_month": "July", "arrival_date_year": 2017},
                "metrics": ["revenue", "total_bookings"]
            },
            {
                # For "What's the cancellation rate for resort hotels?"
                "filters": {"hotel": "Resort Hotel"},
                "metrics": ["cancellation_rate", "total_bookings", "canceled_bookings"]
            },
            {
                # For "Which country has the highest number of bookings?"
                "filters": {},
                "metrics": ["bookings_by_country", "top_countries"]
            },
            {
                # For "How many bookings include special requests?"
                "filters": {"total_of_special_requests": {"gt": 0}},
                "metrics": ["total_bookings", "special_requests_count"]
            },
            {
                # For "Which locations had the highest booking cancellations?"
                "filters": {"is_canceled": 1},
                "metrics": ["cancellations_by_country", "top_cancellation_locations"]
            },
            {
                # For "What percentage of bookings are from repeat guests?"
                "filters": {"is_repeated_guest": 1},
                "metrics": ["total_bookings", "repeat_guest_percentage"]
            }
        ]
        
        results = []
        
        for case in test_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_url}/analytics",
                    json=case
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    result = {
                        "case": case,
                        "response_time": response_time,
                        "status_code": response.status_code,
                        "processing_time": data.get("processing_time_seconds", 0),
                        "results_summary": {
                            "size": len(str(data.get("results", {}))),
                            "keys": list(data.get("results", {}).keys())
                        }
                    }
                    
                    self.results["response_times"].append(response_time)
                else:
                    result = {
                        "case": case,
                        "response_time": response_time,
                        "status_code": response.status_code,
                        "error": response.text
                    }
                    self.results["errors"].append(result)
                
                results.append(result)
                self.results["detailed_results"].append(result)
                
                # Wait a bit to avoid overwhelming the API
                time.sleep(1)
                
            except Exception as e:
                error_result = {
                    "case": case,
                    "error": str(e),
                    "response_time": time.time() - start_time
                }
                results.append(error_result)
                self.results["errors"].append(error_result)
                self.results["detailed_results"].append(error_result)
                
        return results
    
    def test_booking_operations(self) -> Dict[str, Any]:
        """Test basic booking operations (add, update, delete)."""
        from data_update import generate_random_booking
        
        results = {
            "add": None,
            "update": None,
            "delete": None,
            "response_times": {
                "add": 0,
                "update": 0,
                "delete": 0
            }
        }
        
        # Test adding a booking
        try:
            booking = generate_random_booking()
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/bookings",
                json={"booking": booking}
            )
            add_time = time.time() - start_time
            
            if response.status_code == 201:
                booking_id = response.json().get("booking_id")
                results["add"] = {
                    "success": True,
                    "booking_id": booking_id,
                    "status_code": response.status_code
                }
                results["response_times"]["add"] = add_time
                self.results["response_times"].append(add_time)
                
                # Test updating the booking
                if booking_id:
                    update_data = {
                        "is_canceled": 1 if booking["is_canceled"] == 0 else 0,
                        "adr": booking["adr"] * 1.1
                    }
                    
                    start_time = time.time()
                    response = requests.put(
                        f"{self.api_url}/bookings/{booking_id}",
                        json={"booking": update_data}
                    )
                    update_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        results["update"] = {
                            "success": True,
                            "booking_id": booking_id,
                            "status_code": response.status_code
                        }
                        results["response_times"]["update"] = update_time
                        self.results["response_times"].append(update_time)
                    else:
                        results["update"] = {
                            "success": False,
                            "status_code": response.status_code,
                            "error": response.text
                        }
                        self.results["errors"].append({
                            "operation": "update_booking",
                            "error": response.text
                        })
                    
                    # Test deleting the booking
                    start_time = time.time()
                    response = requests.delete(f"{self.api_url}/bookings/{booking_id}")
                    delete_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        results["delete"] = {
                            "success": True,
                            "booking_id": booking_id,
                            "status_code": response.status_code
                        }
                        results["response_times"]["delete"] = delete_time
                        self.results["response_times"].append(delete_time)
                    else:
                        results["delete"] = {
                            "success": False,
                            "status_code": response.status_code,
                            "error": response.text
                        }
                        self.results["errors"].append({
                            "operation": "delete_booking",
                            "error": response.text
                        })
            else:
                results["add"] = {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text
                }
                self.results["errors"].append({
                    "operation": "add_booking",
                    "error": response.text
                })
        except Exception as e:
            results["error"] = str(e)
            self.results["errors"].append({
                "operation": "booking_operations",
                "error": str(e)
            })
        
        return results
    
    def test_batch_import(self, count: int = 5) -> Dict[str, Any]:
        """Test batch import of bookings."""
        from data_update import generate_random_booking
        
        try:
            # Generate random bookings
            bookings = [generate_random_booking() for _ in range(count)]
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/bookings/batch",
                json={"bookings": bookings}
            )
            response_time = time.time() - start_time
            
            if response.status_code == 201:
                result = response.json()
                success_count = result.get("success_count", 0)
                
                batch_result = {
                    "success": True,
                    "requested_count": count,
                    "success_count": success_count,
                    "response_time": response_time,
                    "status_code": response.status_code
                }
                
                self.results["response_times"].append(response_time)
                return batch_result
            else:
                batch_result = {
                    "success": False,
                    "requested_count": count,
                    "status_code": response.status_code,
                    "error": response.text,
                    "response_time": response_time
                }
                
                self.results["errors"].append({
                    "operation": "batch_import",
                    "error": response.text
                })
                return batch_result
                
        except Exception as e:
            error_result = {
                "success": False,
                "requested_count": count,
                "error": str(e),
                "response_time": 0
            }
            
            self.results["errors"].append({
                "operation": "batch_import",
                "error": str(e)
            })
            return error_result
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        if not self.results["response_times"]:
            return {"status": "error", "message": "No test results available"}
        
        avg_response_time = sum(self.results["response_times"]) / len(self.results["response_times"])
        median_response_time = statistics.median(self.results["response_times"])
        max_response_time = max(self.results["response_times"])
        min_response_time = min(self.results["response_times"])
        
        avg_accuracy = sum(self.results["accuracy_scores"]) / len(self.results["accuracy_scores"]) if self.results["accuracy_scores"] else 0
        
        # Sort response times to calculate percentiles
        sorted_times = sorted(self.results["response_times"])
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "api_url": self.api_url,
            "total_tests": len(self.results["response_times"]) + len(self.results["errors"]),
            "successful_tests": len(self.results["response_times"]),
            "failed_tests": len(self.results["errors"]),
            "performance": {
                "avg_response_time": avg_response_time,
                "median_response_time": median_response_time,
                "max_response_time": max_response_time,
                "min_response_time": min_response_time,
                "response_time_p90": sorted_times[int(0.9 * len(sorted_times))] if len(sorted_times) > 1 else sorted_times[0],
                "response_time_p95": sorted_times[int(0.95 * len(sorted_times))] if len(sorted_times) > 1 else sorted_times[0],
            },
            "accuracy": {
                "avg_accuracy": avg_accuracy,
                "min_accuracy": min(self.results["accuracy_scores"]) if self.results["accuracy_scores"] else 0,
                "max_accuracy": max(self.results["accuracy_scores"]) if self.results["accuracy_scores"] else 0,
            },
            "errors": self.results["errors"]
        }
        
        return report
    
    def visualize_results(self, save_path: str = None):
        """Create visualizations of the performance metrics."""
        if not self.results["response_times"]:
            print("No data to visualize")
            return
            
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Response time histogram
        axs[0, 0].hist(self.results["response_times"], bins=10, color='skyblue', edgecolor='black')
        axs[0, 0].set_title('Response Time Distribution')
        axs[0, 0].set_xlabel('Response Time (s)')
        axs[0, 0].set_ylabel('Frequency')
        
        # Accuracy histogram
        if self.results["accuracy_scores"]:
            axs[0, 1].hist(self.results["accuracy_scores"], bins=10, color='lightgreen', edgecolor='black')
            axs[0, 1].set_title('Accuracy Score Distribution')
            axs[0, 1].set_xlabel('Accuracy Score')
            axs[0, 1].set_ylabel('Frequency')
        else:
            axs[0, 1].text(0.5, 0.5, 'No accuracy data available', 
                        horizontalalignment='center', verticalalignment='center')
        
        # Question response time comparison
        question_times = []
        question_labels = []
        
        for i, question in enumerate(self.test_questions):
            for result in self.results.get("detailed_results", []):
                if result.get("question") == question and "response_time" in result:
                    question_times.append(result["response_time"])
                    question_labels.append(f"Q{i+1}")
                    break
        
        if question_times:
            axs[1, 0].bar(question_labels, question_times, color='salmon')
            axs[1, 0].set_title('Response Time by Question')
            axs[1, 0].set_xlabel('Question')
            axs[1, 0].set_ylabel('Response Time (s)')
            plt.setp(axs[1, 0].xaxis.get_majorticklabels(), rotation=45)
        else:
            axs[1, 0].text(0.5, 0.5, 'No question-specific data available', 
                        horizontalalignment='center', verticalalignment='center')
        
        # Error rate pie chart
        total = len(self.results["response_times"]) + len(self.results["errors"])
        if total > 0:
            success = len(self.results["response_times"])
            failed = len(self.results["errors"])
            axs[1, 1].pie([success, failed], labels=['Success', 'Failed'], 
                        autopct='%1.1f%%', colors=['lightgreen', 'salmon'])
            axs[1, 1].set_title('API Request Results')
        else:
            axs[1, 1].text(0.5, 0.5, 'No data available', 
                        horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def save_report(self, report, output_dir: str = "performance_reports"):
        """Save the performance report to a file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"performance_report_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {filename}")
        
        # Save visualization
        vis_filename = os.path.join(output_dir, f"performance_vis_{timestamp}.png")
        self.visualize_results(save_path=vis_filename)
        
        return filename
    
    def run_full_evaluation(self):
        """Run a full evaluation of the API performance."""
        print("Starting API performance evaluation...")
        
        # Check API health
        health = self.check_health()
        if health.get("status") != "ok":
            print(f"API health check failed: {health.get('message', 'Unknown error')}")
            return
            
        print(f"API health check passed: {health.get('message', 'API is healthy')}")
        
        # Test /ask endpoint
        print("\nTesting /ask endpoint...")
        ask_results = self.test_ask_endpoint()
        
        print(f"Completed {len(ask_results)} question tests")
        print(f"Average response time: {sum(self.results['response_times']) / len(self.results['response_times']):.2f}s")
        if self.results["accuracy_scores"]:
            print(f"Average accuracy: {sum(self.results['accuracy_scores']) / len(self.results['accuracy_scores']):.2f}")
        
        # Test /analytics endpoint
        print("\nTesting /analytics endpoint with queries...")
        analytics_results = self.test_analytics_endpoint()
        print(f"Completed {len(analytics_results)} analytics query tests")
        
        # Test specific analytics
        print("\nTesting /analytics endpoint with specific filters and metrics...")
        specific_results = self.test_specific_analytics()
        print(f"Completed {len(specific_results)} specific analytics tests")
        
        # Test booking operations
        print("\nTesting booking operations...")
        booking_results = self.test_booking_operations()
        print(f"Booking operations test: Add {'SUCCESS' if booking_results.get('add', {}).get('success') else 'FAIL'}, Update {'SUCCESS' if booking_results.get('update', {}).get('success') else 'FAIL'}, Delete {'SUCCESS' if booking_results.get('delete', {}).get('success') else 'FAIL'}")
        
        # Test batch import
        print("\nTesting batch import...")
        batch_results = self.test_batch_import(5)
        print(f"Batch import test: {'SUCCESS' if batch_results.get('success') else 'FAIL'}, Imported {batch_results.get('success_count', 0)}/{batch_results.get('requested_count', 0)} bookings")
        
        # Generate and save report
        print("\nGenerating performance report...")
        report = self.generate_performance_report()
        report_file = self.save_report(report)
        
        print("\nEvaluation complete!")
        print(f"Report saved to {report_file}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hotel Booking API performance')
    parser.add_argument('--url', default="http://localhost:8008", help='API base URL')
    parser.add_argument('--output', default='performance_reports', help='Output directory for reports')
    parser.add_argument('--mode', choices=['full', 'questions', 'analytics', 'bookings'], default='full', 
                      help='Evaluation mode (default: full)')
    
    args = parser.parse_args()
    
    evaluator = APIPerformanceEvaluator(api_url=args.url)
    
    # Check API health
    health = evaluator.check_health()
    if health.get("status") != "ok":
        print(f"API health check failed: {health.get('message', 'Unknown error')}")
        return
            
    print(f"API health check passed: {health.get('message', 'API is healthy')}")
    
    # Run evaluation based on mode
    if args.mode == 'full' or args.mode == 'questions':
        print("\nTesting /ask endpoint...")
        ask_results = evaluator.test_ask_endpoint()
    
    if args.mode == 'full' or args.mode == 'analytics':
        print("\nTesting /analytics endpoint with queries...")
        analytics_results = evaluator.test_analytics_endpoint()
        
        print("\nTesting /analytics endpoint with specific filters and metrics...")
        specific_results = evaluator.test_specific_analytics()
    
    """if args.mode == 'full' or args.mode == 'bookings':
        print("\nTesting booking operations...")
        booking_results = evaluator.test_booking_operations()
        
        print("\nTesting batch import...")
        batch_results = evaluator.test_batch_import(5)"""
    
    # Generate and save report
    print("\nGenerating performance report...")
    report = evaluator.generate_performance_report()
    report_file = evaluator.save_report(report, args.output)
    
    print("\nEvaluation complete!")
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    main()