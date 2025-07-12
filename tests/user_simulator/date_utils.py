"""
Date utilities for generating realistic date and time requests in scheduling scenarios.
Supports both specific dates and relative date expressions.
"""

from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple

class DateGenerator:
    """Generates realistic date and time requests for interview scheduling"""
    
    def __init__(self):
        """Initialize with current date as reference"""
        self.today = datetime.now().date()
        
        # Define relative date expressions
        self.relative_dates = [
            "tomorrow",
            "the day after tomorrow",
            "next Monday",
            "next Tuesday", 
            "next Wednesday",
            "next Thursday",
            "next Friday",
            "this Friday",
            "next week",
            "in 3 days",
            "in a week",
            "in two weeks"
        ]
        
        # Define time preferences
        self.time_preferences = [
            "morning",
            "afternoon", 
            "evening",
            None  # No specific time preference
        ]
        
        # Define specific time suggestions
        self.specific_times = [
            "9:00 AM",
            "10:00 AM", 
            "11:00 AM",
            "2:00 PM",
            "3:00 PM",
            "4:00 PM"
        ]
        
        # Weekdays for business scheduling
        self.weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    def get_random_date_request(self) -> str:
        """Generate a random date request (specific or relative)"""
        use_relative = random.choice([True, False])
        
        if use_relative:
            return self._generate_relative_date_request()
        else:
            return self._generate_specific_date_request()
    
    def _generate_relative_date_request(self) -> str:
        """Generate a relative date request like 'tomorrow morning' or 'next Friday'"""
        relative_date = random.choice(self.relative_dates)
        time_pref = random.choice(self.time_preferences)
        
        if time_pref:
            return f"{relative_date} {time_pref}"
        else:
            return relative_date
    
    def _generate_specific_date_request(self) -> str:
        """Generate a specific date request like '2025-07-15 in the morning'"""
        # Generate date 1-14 days in the future
        days_ahead = random.randint(1, 14)
        target_date = self.today + timedelta(days=days_ahead)
        
        # Skip weekends for business scheduling
        while target_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            days_ahead += 1
            target_date = self.today + timedelta(days=days_ahead)
        
        date_str = target_date.strftime("%Y-%m-%d")
        time_pref = random.choice(self.time_preferences)
        
        if time_pref:
            return f"{date_str} in the {time_pref}"
        else:
            return date_str
    
    def get_scheduling_response_with_date(self, base_response: str) -> str:
        """Add a date suggestion to a scheduling response"""
        date_request = self.get_random_date_request()
        
        scheduling_phrases = [
            f"How about {date_request}?",
            f"I'm available {date_request}.",
            f"Could we schedule for {date_request}?",
            f"Would {date_request} work?",
            f"I'm free {date_request}.",
            f"Let's meet {date_request}."
        ]
        
        phrase = random.choice(scheduling_phrases)
        
        # Combine base response with date suggestion
        return f"{base_response} {phrase}"
    
    def generate_multiple_preferences(self, count: int = 2) -> List[str]:
        """Generate multiple date/time preferences for detailed personas"""
        preferences = []
        
        for _ in range(count):
            pref = self.get_random_date_request()
            if pref not in preferences:
                preferences.append(pref)
        
        return preferences
    
    def get_flexible_response(self) -> str:
        """Generate a flexible scheduling response"""
        flexible_phrases = [
            "I'm flexible with timing",
            "Any time next week works for me",
            "I can adjust my schedule",
            "Pretty flexible on dates",
            "Whatever works best for you"
        ]
        
        return random.choice(flexible_phrases)
    
    def get_specific_time_request(self) -> str:
        """Generate a request with specific time"""
        specific_time = random.choice(self.specific_times)
        date_part = self.get_random_date_request()
        
        return f"{date_part} at {specific_time}"
    
    def convert_relative_to_specific(self, relative_date: str) -> str:
        """Convert relative date to specific date (for testing purposes)"""
        relative_lower = relative_date.lower()
        
        if "tomorrow" in relative_lower:
            target_date = self.today + timedelta(days=1)
        elif "day after tomorrow" in relative_lower:
            target_date = self.today + timedelta(days=2)
        elif "next week" in relative_lower:
            target_date = self.today + timedelta(days=7)
        elif "in 3 days" in relative_lower:
            target_date = self.today + timedelta(days=3)
        elif "in a week" in relative_lower:
            target_date = self.today + timedelta(days=7)
        elif "in two weeks" in relative_lower:
            target_date = self.today + timedelta(days=14)
        elif "monday" in relative_lower:
            days_ahead = (0 - self.today.weekday()) % 7
            if days_ahead == 0:  # If today is Monday, get next Monday
                days_ahead = 7
            target_date = self.today + timedelta(days=days_ahead)
        elif "tuesday" in relative_lower:
            days_ahead = (1 - self.today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = self.today + timedelta(days=days_ahead)
        elif "wednesday" in relative_lower:
            days_ahead = (2 - self.today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = self.today + timedelta(days=days_ahead)
        elif "thursday" in relative_lower:
            days_ahead = (3 - self.today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = self.today + timedelta(days=days_ahead)
        elif "friday" in relative_lower:
            days_ahead = (4 - self.today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = self.today + timedelta(days=days_ahead)
        else:
            # Default to tomorrow
            target_date = self.today + timedelta(days=1)
        
        return target_date.strftime("%Y-%m-%d")

class SchedulingBehaviorGenerator:
    """Generates persona-specific scheduling behaviors"""
    
    def __init__(self, date_generator: DateGenerator):
        self.date_gen = date_generator
    
    def get_eager_scheduling_response(self) -> str:
        """Generate enthusiastic scheduling response"""
        responses = [
            "Yes! I'd love to schedule an interview.",
            "That would be fantastic!",
            "I'm very excited to meet with you.",
            "Absolutely! When would work best?"
        ]
        
        base = random.choice(responses)
        return self.date_gen.get_scheduling_response_with_date(base)
    
    def get_skeptical_scheduling_response(self) -> str:
        """Generate cautious scheduling response"""
        responses = [
            "I suppose we could set up a meeting.",
            "I'm willing to discuss this further.",
            "We could schedule something to talk more."
        ]
        
        base = random.choice(responses)
        return self.date_gen.get_scheduling_response_with_date(base)
    
    def get_direct_scheduling_response(self) -> str:
        """Generate brief, direct scheduling response"""
        responses = [
            "Let's schedule it.",
            "Fine, when?",
            "Sure.",
            "What times are available?"
        ]
        
        base = random.choice(responses)
        if random.choice([True, False]):
            return self.date_gen.get_scheduling_response_with_date(base)
        else:
            return base
    
    def get_detail_oriented_scheduling_response(self) -> str:
        """Generate thorough scheduling response"""
        responses = [
            "I'd like to schedule a comprehensive discussion.",
            "Let's set up a detailed interview.",
            "I'm interested in a thorough conversation about the role."
        ]
        
        base = random.choice(responses)
        preferences = self.date_gen.generate_multiple_preferences(2)
        pref_text = " or ".join(preferences)
        
        return f"{base} I'm available {pref_text}."
    
    def get_indecisive_scheduling_response(self) -> str:
        """Generate hesitant scheduling response"""
        responses = [
            "Maybe we could schedule something...",
            "I'm not sure about timing, but we could try to meet.",
            "I suppose we could set something up, though I'm uncertain about my schedule."
        ]
        
        return random.choice(responses)
    
    def get_disinterested_scheduling_response(self) -> str:
        """Generate polite decline"""
        responses = [
            "Thank you, but I don't think I need to schedule anything.",
            "I appreciate the offer, but I'm not interested in proceeding.",
            "Thanks, but this doesn't seem like the right fit."
        ]
        
        return random.choice(responses) 