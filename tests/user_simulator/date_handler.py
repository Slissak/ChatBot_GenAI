"""
Date Handling System for Conversation Scenarios - Provides comprehensive date parsing 
and generation for scheduling scenarios, supporting both specific dates and relative 
date references.
"""

import re
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import calendar

class DateType(Enum):
    """Types of date references"""
    SPECIFIC = "specific"  # "2025-01-15", "January 15th"
    RELATIVE = "relative"  # "tomorrow", "next week"
    RANGE = "range"       # "next week", "this month"
    TIME_SPECIFIC = "time_specific"  # "tomorrow at 2pm"

class TimeFrame(Enum):
    """Time frames for scheduling"""
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    ANYTIME = "anytime"

@dataclass
class ParsedDate:
    """Represents a parsed date with context"""
    original_text: str
    parsed_date: date
    date_type: DateType
    time_frame: Optional[TimeFrame] = None
    specific_time: Optional[str] = None
    confidence: float = 1.0
    is_business_day: bool = True
    alternative_dates: List[date] = None
    
    def __post_init__(self):
        if self.alternative_dates is None:
            self.alternative_dates = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "parsed_date": self.parsed_date.isoformat(),
            "date_type": self.date_type.value,
            "time_frame": self.time_frame.value if self.time_frame else None,
            "specific_time": self.specific_time,
            "confidence": self.confidence,
            "is_business_day": self.is_business_day,
            "alternative_dates": [d.isoformat() for d in self.alternative_dates]
        }

class DateParser:
    """Advanced date parser for natural language date references"""
    
    def __init__(self, base_date: Optional[date] = None):
        self.logger = logging.getLogger('date_parser')
        self.base_date = base_date or date.today()
        
        # Relative date patterns
        self.relative_patterns = {
            # Days
            r'\btomorrow\b': lambda: self.base_date + timedelta(days=1),
            r'\btoday\b': lambda: self.base_date,
            r'\byesterday\b': lambda: self.base_date - timedelta(days=1),
            r'\bin (\d+) days?\b': lambda m: self.base_date + timedelta(days=int(m.group(1))),
            r'\b(\d+) days? from now\b': lambda m: self.base_date + timedelta(days=int(m.group(1))),
            
            # Weeks
            r'\bnext week\b': lambda: self.base_date + timedelta(weeks=1),
            r'\bthis week\b': lambda: self.base_date,
            r'\blast week\b': lambda: self.base_date - timedelta(weeks=1),
            r'\bin (\d+) weeks?\b': lambda m: self.base_date + timedelta(weeks=int(m.group(1))),
            
            # Specific days of week
            r'\bnext (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b': self._next_weekday,
            r'\bthis (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b': self._this_weekday,
            r'\blast (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b': self._last_weekday,
        }
        
        # Specific date patterns
        self.specific_patterns = [
            # ISO format: 2025-01-15
            r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',
            # US format: 01/15/2025, 1/15/25
            r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b',
            # European format: 15/01/2025, 15.01.2025
            r'\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b',
            # Month day year: January 15, 2025; Jan 15 2025
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2}),?\s*(\d{2,4})?\b',
            # Day month year: 15 January 2025, 15th Jan
            r'\b(\d{1,2})(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*(\d{2,4})?\b'
        ]
        
        # Time frame patterns
        self.time_patterns = {
            r'\bin the morning\b': TimeFrame.MORNING,
            r'\bmorning\b': TimeFrame.MORNING,
            r'\bam\b': TimeFrame.MORNING,
            r'\bin the afternoon\b': TimeFrame.AFTERNOON,
            r'\bafternoon\b': TimeFrame.AFTERNOON,
            r'\bpm\b': TimeFrame.AFTERNOON,
            r'\bin the evening\b': TimeFrame.EVENING,
            r'\bevening\b': TimeFrame.EVENING,
            r'\banytime\b': TimeFrame.ANYTIME,
            r'\bany time\b': TimeFrame.ANYTIME,
        }
        
        # Month name mappings
        self.month_names = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        
        # Weekday mappings
        self.weekday_names = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
    
    def parse_date(self, text: str) -> Optional[ParsedDate]:
        """Parse a date from natural language text"""
        text_lower = text.lower().strip()
        self.logger.debug(f"Parsing date from: '{text}'")
        
        # Try relative dates first
        relative_result = self._parse_relative_date(text_lower)
        if relative_result:
            return relative_result
        
        # Try specific dates
        specific_result = self._parse_specific_date(text_lower)
        if specific_result:
            return specific_result
        
        self.logger.warning(f"Could not parse date from: '{text}'")
        return None
    
    def _parse_relative_date(self, text: str) -> Optional[ParsedDate]:
        """Parse relative date expressions"""
        for pattern, func in self.relative_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if callable(func):
                        if pattern in [r'\bnext (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                                     r'\bthis (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                                     r'\blast (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b']:
                            parsed_date = func(match)
                        else:
                            parsed_date = func() if not match.groups() else func(match)
                    else:
                        parsed_date = func
                    
                    time_frame = self._extract_time_frame(text)
                    
                    return ParsedDate(
                        original_text=text,
                        parsed_date=parsed_date,
                        date_type=DateType.RELATIVE,
                        time_frame=time_frame,
                        confidence=0.9,
                        is_business_day=self._is_business_day(parsed_date)
                    )
                except Exception as e:
                    self.logger.error(f"Error parsing relative date '{text}': {e}")
                    continue
        
        return None
    
    def _parse_specific_date(self, text: str) -> Optional[ParsedDate]:
        """Parse specific date formats"""
        for pattern in self.specific_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    parsed_date = self._extract_date_from_match(match, pattern)
                    if parsed_date:
                        time_frame = self._extract_time_frame(text)
                        
                        return ParsedDate(
                            original_text=text,
                            parsed_date=parsed_date,
                            date_type=DateType.SPECIFIC,
                            time_frame=time_frame,
                            confidence=0.95,
                            is_business_day=self._is_business_day(parsed_date)
                        )
                except Exception as e:
                    self.logger.error(f"Error parsing specific date '{text}': {e}")
                    continue
        
        return None
    
    def _extract_date_from_match(self, match: re.Match, pattern: str) -> Optional[date]:
        """Extract date object from regex match"""
        groups = match.groups()
        
        if pattern == r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b':
            # ISO format: YYYY-MM-DD
            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
            return date(year, month, day)
        
        elif pattern == r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b':
            # US format: MM/DD/YYYY
            month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
            if year < 100:
                year += 2000  # Assume 21st century for 2-digit years
            return date(year, month, day)
        
        elif pattern == r'\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b':
            # European format: DD/MM/YYYY or DD.MM.YYYY
            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
            if year < 100:
                year += 2000
            return date(year, month, day)
        
        elif 'january|february' in pattern and len(groups) >= 3:
            # Month day year format
            month_name = groups[0].lower()
            day = int(groups[1])
            year = int(groups[2]) if groups[2] else self.base_date.year
            
            month = self.month_names.get(month_name)
            if month:
                return date(year, month, day)
        
        elif len(groups) >= 3 and groups[2]:
            # Day month year format
            day = int(groups[0])
            month_name = groups[2].lower()
            year = int(groups[3]) if len(groups) > 3 and groups[3] else self.base_date.year
            
            month = self.month_names.get(month_name)
            if month:
                return date(year, month, day)
        
        return None
    
    def _extract_time_frame(self, text: str) -> Optional[TimeFrame]:
        """Extract time frame from text"""
        for pattern, time_frame in self.time_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return time_frame
        return None
    
    def _next_weekday(self, match: re.Match) -> date:
        """Get next occurrence of specified weekday"""
        weekday_name = match.group(1).lower()
        target_weekday = self.weekday_names[weekday_name]
        
        days_ahead = target_weekday - self.base_date.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        return self.base_date + timedelta(days=days_ahead)
    
    def _this_weekday(self, match: re.Match) -> date:
        """Get occurrence of specified weekday this week"""
        weekday_name = match.group(1).lower()
        target_weekday = self.weekday_names[weekday_name]
        
        days_diff = target_weekday - self.base_date.weekday()
        return self.base_date + timedelta(days=days_diff)
    
    def _last_weekday(self, match: re.Match) -> date:
        """Get last occurrence of specified weekday"""
        weekday_name = match.group(1).lower()
        target_weekday = self.weekday_names[weekday_name]
        
        days_back = self.base_date.weekday() - target_weekday
        if days_back <= 0:
            days_back += 7
        
        return self.base_date - timedelta(days=days_back)
    
    def _is_business_day(self, target_date: date) -> bool:
        """Check if date is a business day (Monday-Friday)"""
        return target_date.weekday() < 5

class DateGenerator:
    """Generates dates for scenario testing"""
    
    def __init__(self, base_date: Optional[date] = None):
        self.base_date = base_date or date.today()
        self.parser = DateParser(base_date)
    
    def generate_future_business_dates(self, count: int = 5) -> List[date]:
        """Generate future business days"""
        dates = []
        current_date = self.base_date + timedelta(days=1)
        
        while len(dates) < count:
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                dates.append(current_date)
            current_date += timedelta(days=1)
        
        return dates
    
    def generate_relative_date_expressions(self) -> List[str]:
        """Generate common relative date expressions for testing"""
        return [
            "tomorrow",
            "next week",
            "next Monday",
            "next Friday",
            "in 3 days",
            "in 2 weeks",
            "this Friday",
            "next Tuesday morning",
            "tomorrow afternoon",
            "next week anytime"
        ]
    
    def generate_specific_date_expressions(self, count: int = 5) -> List[str]:
        """Generate specific date expressions for testing"""
        future_dates = self.generate_future_business_dates(count)
        expressions = []
        
        for target_date in future_dates:
            # Multiple formats for the same date
            expressions.extend([
                target_date.strftime("%Y-%m-%d"),  # ISO format
                target_date.strftime("%m/%d/%Y"),  # US format
                target_date.strftime("%B %d, %Y"), # Full month name
                target_date.strftime("%b %d %Y"),  # Abbreviated month
                f"{target_date.strftime('%B %d')} in the morning",
                f"{target_date.strftime('%m/%d')} afternoon"
            ])
        
        return expressions
    
    def generate_scenario_dates(self, scenario_type: str) -> Dict[str, Any]:
        """Generate appropriate dates for specific scenario types"""
        if scenario_type == "immediate_scheduling":
            return {
                "preferred_date": "tomorrow",
                "alternatives": ["next Monday", "next Tuesday"],
                "time_frame": "morning"
            }
        
        elif scenario_type == "flexible_scheduling":
            return {
                "preferred_date": "next week",
                "alternatives": self.generate_future_business_dates(3),
                "time_frame": "anytime"
            }
        
        elif scenario_type == "specific_date_request":
            target_date = self.base_date + timedelta(days=7)
            return {
                "preferred_date": target_date.strftime("%Y-%m-%d"),
                "alternatives": [target_date + timedelta(days=i) for i in [1, 2, 3]],
                "time_frame": "afternoon"
            }
        
        else:
            # Default scenario
            return {
                "preferred_date": "next week",
                "alternatives": self.generate_future_business_dates(3),
                "time_frame": "anytime"
            }

class ConversationDateContext:
    """Manages date context for conversation scenarios"""
    
    def __init__(self, base_date: Optional[date] = None):
        self.base_date = base_date or date.today()
        self.parser = DateParser(base_date)
        self.generator = DateGenerator(base_date)
        self.conversation_dates: Dict[str, ParsedDate] = {}
    
    def set_conversation_date(self, conversation_id: str, date_text: str) -> Optional[ParsedDate]:
        """Set and parse date for a conversation"""
        parsed = self.parser.parse_date(date_text)
        if parsed:
            self.conversation_dates[conversation_id] = parsed
        return parsed
    
    def get_conversation_date(self, conversation_id: str) -> Optional[ParsedDate]:
        """Get parsed date for a conversation"""
        return self.conversation_dates.get(conversation_id)
    
    def format_for_agent(self, parsed_date: ParsedDate) -> str:
        """Format date for agent consumption (ISO format)"""
        return parsed_date.parsed_date.isoformat()
    
    def format_for_user(self, parsed_date: ParsedDate) -> str:
        """Format date for user-friendly display"""
        target_date = parsed_date.parsed_date
        
        # Calculate days from today
        days_diff = (target_date - self.base_date).days
        
        if days_diff == 0:
            return "today"
        elif days_diff == 1:
            return "tomorrow"
        elif days_diff == -1:
            return "yesterday"
        elif 1 < days_diff <= 7:
            return f"in {days_diff} days ({target_date.strftime('%A')})"
        else:
            return target_date.strftime("%B %d, %Y")
    
    def suggest_alternatives(self, parsed_date: ParsedDate, count: int = 3) -> List[date]:
        """Suggest alternative dates near the requested date"""
        base_date = parsed_date.parsed_date
        alternatives = []
        
        # Suggest next few business days
        current = base_date + timedelta(days=1)
        while len(alternatives) < count:
            if current.weekday() < 5:  # Business day
                alternatives.append(current)
            current += timedelta(days=1)
        
        return alternatives

# Convenience functions
def parse_date_from_text(text: str, base_date: Optional[date] = None) -> Optional[ParsedDate]:
    """Quick function to parse date from text"""
    parser = DateParser(base_date)
    return parser.parse_date(text)

def generate_test_dates(scenario_type: str = "default", base_date: Optional[date] = None) -> Dict[str, Any]:
    """Quick function to generate test dates for scenarios"""
    generator = DateGenerator(base_date)
    return generator.generate_scenario_dates(scenario_type)

def format_date_for_scheduling(date_text: str, base_date: Optional[date] = None) -> Optional[str]:
    """Format natural language date for scheduling system"""
    parsed = parse_date_from_text(date_text, base_date)
    if parsed:
        return parsed.parsed_date.isoformat()
    return None 