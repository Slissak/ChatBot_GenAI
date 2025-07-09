from main import sched_agent
from datetime import datetime, timedelta

def test_sched_agent():
    print("Initializing sched_agent...")
    scheduler = sched_agent()
    
    # Test cases
    test_cases = [
        {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "time_frame": "morning",
            "description": "Today's morning slots"
        },
        {
            "date": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            "time_frame": "afternoon",
            "description": "Tomorrow's afternoon slots"
        },
        {
            "date": (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'),
            "time_frame": None,
            "description": "Day after tomorrow all slots"
        }
    ]
    
    print("\nTesting sched_agent functionality...")
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Date: {test['date']}")
        print(f"Time Frame: {test['time_frame'] if test['time_frame'] else 'All day'}")
        
        try:
            response = scheduler.schedule_interview(
                date_str=test["date"],
                time_frame=test["time_frame"]
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_sched_agent() 