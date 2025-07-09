import os
from dotenv import load_dotenv
import pyodbc
from datetime import datetime

def test_db_connection():
    # Load environment variables
    load_dotenv()
    
    # Database configuration
    db_config = {
        'driver': os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server'),
        'server': os.getenv('DB_SERVER'),
        'database': os.getenv('DB_NAME'),
        'port': os.getenv('DB_PORT'),
        'uid': os.getenv('DB_USER'),
        'pwd': os.getenv('DB_PASSWORD'),
        'TrustServerCertificate': 'yes'  # Add this for development environments
    }
    
    print("Database Configuration:")
    print(f"Driver: {db_config['driver']}")
    print(f"Server: {db_config['server']}")
    print(f"Database: {db_config['database']}")
    print(f"Port: {db_config['port']}")
    print(f"User: {db_config['uid']}")
    
    try:
        # Create connection string
        conn_str = ';'.join([f"{k}={v}" for k, v in db_config.items()])
        print("\nAttempting to connect to database...")
        
        # Test connection
        conn = pyodbc.connect(conn_str)
        print("Successfully connected to the database!")
        
        # Test query
        cursor = conn.cursor()
        test_query = """
        SELECT TOP 5 ScheduleID, [date], [time], position, available
        FROM dbo.Schedule
        WHERE position = 'Python Dev'
        ORDER BY [date], [time]
        """
        
        print("\nExecuting test query...")
        cursor.execute(test_query)
        rows = cursor.fetchall()
        
        print("\nQuery Results:")
        for row in rows:
            print(f"ScheduleID: {row.ScheduleID}")
            print(f"Date: {row.date}")
            print(f"Time: {row.time}")
            print(f"Position: {row.position}")
            print(f"Available: {row.available}")
            print("-" * 30)
        
        conn.close()
        print("\nDatabase connection closed successfully.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    test_db_connection() 