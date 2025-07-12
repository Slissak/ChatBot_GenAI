"""
Batch Monitor - Real-time monitoring and alerting for automated test execution.
Tracks performance trends, detects anomalies, and provides actionable insights.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from pathlib import Path
import statistics
from enum import Enum

# Import our core systems
from tests.automation.test_runner import TestResult, TestStatus, TestConfiguration, AutomatedTestRunner
from tests.automation.test_reporter import TestReporter, generate_quick_summary

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    """System alert"""
    level: AlertLevel
    message: str
    timestamp: datetime
    category: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "details": self.details,
            "resolved": self.resolved
        }

@dataclass
class BatchMetrics:
    """Metrics for a batch of test executions"""
    batch_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    timeout_tests: int = 0
    average_duration: float = 0.0
    average_validation_score: float = 0.0
    throughput_tests_per_minute: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return self.successful_tests / self.total_tests if self.total_tests > 0 else 0.0
    
    @property
    def failure_rate(self) -> float:
        return self.failed_tests / self.total_tests if self.total_tests > 0 else 0.0
    
    @property
    def duration_minutes(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return 0.0

@dataclass
class MonitoringConfiguration:
    """Configuration for batch monitoring"""
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "success_rate_warning": 0.8,
        "success_rate_critical": 0.6,
        "average_duration_warning": 60.0,
        "average_duration_critical": 120.0,
        "validation_score_warning": 0.7,
        "validation_score_critical": 0.5,
        "throughput_warning": 5.0,  # tests per minute
        "throughput_critical": 2.0
    })
    
    monitoring_interval_seconds: int = 30
    retention_days: int = 30
    enable_real_time_alerts: bool = True
    enable_trend_analysis: bool = True
    alert_history_limit: int = 1000
    metrics_history_limit: int = 100

class TrendAnalyzer:
    """Analyzes trends in test execution metrics"""
    
    def __init__(self, history_limit: int = 100):
        self.history_limit = history_limit
        self.metrics_history: deque = deque(maxlen=history_limit)
    
    def add_batch_metrics(self, metrics: BatchMetrics):
        """Add batch metrics to trend analysis"""
        self.metrics_history.append(metrics)
    
    def detect_performance_trends(self) -> List[Dict[str, Any]]:
        """Detect performance trends over recent batches"""
        if len(self.metrics_history) < 5:
            return []
        
        trends = []
        recent_batches = list(self.metrics_history)[-10:]  # Last 10 batches
        older_batches = list(self.metrics_history)[-20:-10] if len(self.metrics_history) >= 20 else []
        
        if older_batches:
            # Success rate trend
            recent_success_rate = statistics.mean([b.success_rate for b in recent_batches])
            older_success_rate = statistics.mean([b.success_rate for b in older_batches])
            
            if recent_success_rate < older_success_rate - 0.1:  # 10% decrease
                trends.append({
                    "type": "declining_success_rate",
                    "severity": "warning",
                    "message": f"Success rate declining: {recent_success_rate:.1%} vs {older_success_rate:.1%}",
                    "recent_value": recent_success_rate,
                    "previous_value": older_success_rate
                })
            
            # Duration trend
            recent_duration = statistics.mean([b.average_duration for b in recent_batches])
            older_duration = statistics.mean([b.average_duration for b in older_batches])
            
            if recent_duration > older_duration * 1.2:  # 20% increase
                trends.append({
                    "type": "increasing_duration",
                    "severity": "warning", 
                    "message": f"Test duration increasing: {recent_duration:.1f}s vs {older_duration:.1f}s",
                    "recent_value": recent_duration,
                    "previous_value": older_duration
                })
            
            # Throughput trend
            recent_throughput = statistics.mean([b.throughput_tests_per_minute for b in recent_batches if b.throughput_tests_per_minute > 0])
            older_throughput = statistics.mean([b.throughput_tests_per_minute for b in older_batches if b.throughput_tests_per_minute > 0])
            
            if recent_throughput < older_throughput * 0.8:  # 20% decrease
                trends.append({
                    "type": "declining_throughput",
                    "severity": "warning",
                    "message": f"Throughput declining: {recent_throughput:.1f} vs {older_throughput:.1f} tests/min",
                    "recent_value": recent_throughput,
                    "previous_value": older_throughput
                })
        
        return trends
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        if not self.metrics_history:
            return {}
        
        recent_batches = list(self.metrics_history)[-10:]
        
        return {
            "total_batches_analyzed": len(self.metrics_history),
            "recent_batches_count": len(recent_batches),
            "average_success_rate": statistics.mean([b.success_rate for b in recent_batches]),
            "average_duration": statistics.mean([b.average_duration for b in recent_batches]),
            "average_throughput": statistics.mean([b.throughput_tests_per_minute for b in recent_batches if b.throughput_tests_per_minute > 0]),
            "success_rate_trend": "stable",  # Would calculate actual trend
            "duration_trend": "stable",
            "throughput_trend": "stable"
        }

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.logger = logging.getLogger('alert_manager')
        self.alert_history: deque = deque(maxlen=config.alert_history_limit)
        self.alert_callbacks: List[Callable[[Alert], None]] = []
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def trigger_alert(self, level: AlertLevel, message: str, category: str, details: Dict[str, Any] = None):
        """Trigger a new alert"""
        alert = Alert(
            level=level,
            message=message,
            timestamp=datetime.now(),
            category=category,
            details=details or {}
        )
        
        self.alert_history.append(alert)
        self.logger.log(
            logging.ERROR if level in [AlertLevel.ERROR, AlertLevel.CRITICAL] else logging.WARNING,
            f"ALERT [{level.value.upper()}] {category}: {message}"
        )
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def check_metrics_for_alerts(self, metrics: BatchMetrics):
        """Check batch metrics against alert thresholds"""
        thresholds = self.config.alert_thresholds
        
        # Success rate alerts
        if metrics.success_rate < thresholds["success_rate_critical"]:
            self.trigger_alert(
                AlertLevel.CRITICAL,
                f"Critical low success rate: {metrics.success_rate:.1%}",
                "performance",
                {"success_rate": metrics.success_rate, "batch_id": metrics.batch_id}
            )
        elif metrics.success_rate < thresholds["success_rate_warning"]:
            self.trigger_alert(
                AlertLevel.WARNING,
                f"Low success rate: {metrics.success_rate:.1%}",
                "performance",
                {"success_rate": metrics.success_rate, "batch_id": metrics.batch_id}
            )
        
        # Duration alerts
        if metrics.average_duration > thresholds["average_duration_critical"]:
            self.trigger_alert(
                AlertLevel.CRITICAL,
                f"Critical high test duration: {metrics.average_duration:.1f}s",
                "performance",
                {"average_duration": metrics.average_duration, "batch_id": metrics.batch_id}
            )
        elif metrics.average_duration > thresholds["average_duration_warning"]:
            self.trigger_alert(
                AlertLevel.WARNING,
                f"High test duration: {metrics.average_duration:.1f}s", 
                "performance",
                {"average_duration": metrics.average_duration, "batch_id": metrics.batch_id}
            )
        
        # Validation score alerts
        if metrics.average_validation_score < thresholds["validation_score_critical"]:
            self.trigger_alert(
                AlertLevel.CRITICAL,
                f"Critical low validation scores: {metrics.average_validation_score:.2f}",
                "quality",
                {"average_validation_score": metrics.average_validation_score, "batch_id": metrics.batch_id}
            )
        elif metrics.average_validation_score < thresholds["validation_score_warning"]:
            self.trigger_alert(
                AlertLevel.WARNING,
                f"Low validation scores: {metrics.average_validation_score:.2f}",
                "quality", 
                {"average_validation_score": metrics.average_validation_score, "batch_id": metrics.batch_id}
            )
        
        # Throughput alerts
        if metrics.throughput_tests_per_minute < thresholds["throughput_critical"]:
            self.trigger_alert(
                AlertLevel.CRITICAL,
                f"Critical low throughput: {metrics.throughput_tests_per_minute:.1f} tests/min",
                "performance",
                {"throughput": metrics.throughput_tests_per_minute, "batch_id": metrics.batch_id}
            )
        elif metrics.throughput_tests_per_minute < thresholds["throughput_warning"]:
            self.trigger_alert(
                AlertLevel.WARNING,
                f"Low throughput: {metrics.throughput_tests_per_minute:.1f} tests/min",
                "performance",
                {"throughput": metrics.throughput_tests_per_minute, "batch_id": metrics.batch_id}
            )
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        recent_alerts = self.get_recent_alerts(24)
        
        return {
            "total_alerts_24h": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.level == AlertLevel.CRITICAL]),
            "error_alerts": len([a for a in recent_alerts if a.level == AlertLevel.ERROR]),
            "warning_alerts": len([a for a in recent_alerts if a.level == AlertLevel.WARNING]),
            "categories": dict(defaultdict(int, {alert.category: len([a for a in recent_alerts if a.category == alert.category]) for alert in recent_alerts}))
        }

class BatchMonitor:
    """Main batch monitoring system"""
    
    def __init__(self, config: MonitoringConfiguration = None):
        self.config = config or MonitoringConfiguration()
        self.logger = logging.getLogger('batch_monitor')
        
        # Core components
        self.alert_manager = AlertManager(self.config)
        self.trend_analyzer = TrendAnalyzer(self.config.metrics_history_limit)
        
        # State tracking
        self.active_batches: Dict[str, BatchMetrics] = {}
        self.completed_batches: deque = deque(maxlen=self.config.metrics_history_limit)
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        # Setup directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Setup monitoring directories"""
        Path("monitoring_data").mkdir(exist_ok=True)
        Path("monitoring_reports").mkdir(exist_ok=True)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("🔍 Batch monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("⏹️ Batch monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_active_batches()
                self._analyze_trends()
                time.sleep(self.config.monitoring_interval_seconds)
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
    
    def _check_active_batches(self):
        """Check active batches for issues"""
        current_time = datetime.now()
        
        for batch_id, metrics in self.active_batches.items():
            # Check for stalled batches
            if (current_time - metrics.start_time).total_seconds() > 300:  # 5 minutes
                self.alert_manager.trigger_alert(
                    AlertLevel.WARNING,
                    f"Batch may be stalled: {batch_id}",
                    "execution",
                    {"batch_id": batch_id, "duration_minutes": (current_time - metrics.start_time).total_seconds() / 60}
                )
    
    def _analyze_trends(self):
        """Analyze performance trends"""
        if self.config.enable_trend_analysis:
            trends = self.trend_analyzer.detect_performance_trends()
            
            for trend in trends:
                level = AlertLevel.WARNING if trend["severity"] == "warning" else AlertLevel.ERROR
                self.alert_manager.trigger_alert(
                    level,
                    trend["message"],
                    "trend",
                    trend
                )
    
    def register_batch_start(self, batch_id: str) -> BatchMetrics:
        """Register the start of a new batch"""
        metrics = BatchMetrics(
            batch_id=batch_id,
            start_time=datetime.now()
        )
        
        self.active_batches[batch_id] = metrics
        self.logger.info(f"📊 Registered batch start: {batch_id}")
        
        return metrics
    
    def update_batch_progress(self, batch_id: str, test_results: List[TestResult]):
        """Update batch progress with current test results"""
        if batch_id not in self.active_batches:
            self.logger.warning(f"Unknown batch ID: {batch_id}")
            return
        
        metrics = self.active_batches[batch_id]
        
        # Update metrics
        metrics.total_tests = len(test_results)
        metrics.successful_tests = len([r for r in test_results if r.status == TestStatus.COMPLETED])
        metrics.failed_tests = len([r for r in test_results if r.status == TestStatus.FAILED])
        metrics.timeout_tests = len([r for r in test_results if r.status == TestStatus.TIMEOUT])
        
        durations = [r.duration_seconds for r in test_results if r.duration_seconds > 0]
        if durations:
            metrics.average_duration = statistics.mean(durations)
        
        validation_scores = [r.validation_score for r in test_results if r.validation_score > 0]
        if validation_scores:
            metrics.average_validation_score = statistics.mean(validation_scores)
        
        # Calculate throughput
        elapsed_minutes = (datetime.now() - metrics.start_time).total_seconds() / 60
        if elapsed_minutes > 0:
            metrics.throughput_tests_per_minute = metrics.total_tests / elapsed_minutes
        
        self.logger.debug(f"Updated batch {batch_id}: {metrics.total_tests} tests, {metrics.success_rate:.1%} success rate")
    
    def complete_batch(self, batch_id: str, test_results: List[TestResult]):
        """Mark batch as completed and perform final analysis"""
        if batch_id not in self.active_batches:
            self.logger.warning(f"Unknown batch ID: {batch_id}")
            return
        
        # Final update
        self.update_batch_progress(batch_id, test_results)
        
        metrics = self.active_batches[batch_id]
        metrics.end_time = datetime.now()
        
        # Check for alerts
        self.alert_manager.check_metrics_for_alerts(metrics)
        
        # Add to trend analysis
        self.trend_analyzer.add_batch_metrics(metrics)
        
        # Move to completed batches
        self.completed_batches.append(metrics)
        del self.active_batches[batch_id]
        
        self.logger.info(f"✅ Completed batch {batch_id}: {metrics.success_rate:.1%} success rate in {metrics.duration_minutes:.1f} minutes")
        
        # Save batch data
        self._save_batch_data(metrics, test_results)
    
    def _save_batch_data(self, metrics: BatchMetrics, test_results: List[TestResult]):
        """Save batch data for historical analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monitoring_data/batch_{metrics.batch_id}_{timestamp}.json"
        
        data = {
            "metrics": {
                "batch_id": metrics.batch_id,
                "start_time": metrics.start_time.isoformat(),
                "end_time": metrics.end_time.isoformat() if metrics.end_time else None,
                "total_tests": metrics.total_tests,
                "successful_tests": metrics.successful_tests,
                "failed_tests": metrics.failed_tests,
                "timeout_tests": metrics.timeout_tests,
                "success_rate": metrics.success_rate,
                "average_duration": metrics.average_duration,
                "average_validation_score": metrics.average_validation_score,
                "throughput_tests_per_minute": metrics.throughput_tests_per_minute
            },
            "test_results": [result.to_dict() for result in test_results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "active_batches": {
                batch_id: {
                    "batch_id": metrics.batch_id,
                    "start_time": metrics.start_time.isoformat(),
                    "duration_minutes": (datetime.now() - metrics.start_time).total_seconds() / 60,
                    "total_tests": metrics.total_tests,
                    "success_rate": metrics.success_rate,
                    "throughput": metrics.throughput_tests_per_minute
                }
                for batch_id, metrics in self.active_batches.items()
            },
            "recent_batches": [
                {
                    "batch_id": metrics.batch_id,
                    "start_time": metrics.start_time.isoformat(),
                    "duration_minutes": metrics.duration_minutes,
                    "total_tests": metrics.total_tests,
                    "success_rate": metrics.success_rate,
                    "average_validation_score": metrics.average_validation_score
                }
                for metrics in list(self.completed_batches)[-10:]
            ],
            "alert_summary": self.alert_manager.get_alert_summary(),
            "performance_summary": self.trend_analyzer.get_performance_summary(),
            "system_status": "healthy" if len(self.alert_manager.get_recent_alerts(1)) == 0 else "issues_detected"
        }
    
    def generate_monitoring_report(self, filename: Optional[str] = None) -> str:
        """Generate detailed monitoring report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monitoring_reports/monitoring_report_{timestamp}.json"
        
        dashboard_data = self.get_monitoring_dashboard()
        recent_alerts = self.alert_manager.get_recent_alerts(24)
        
        report_data = {
            **dashboard_data,
            "recent_alerts": [alert.to_dict() for alert in recent_alerts],
            "configuration": {
                "alert_thresholds": self.config.alert_thresholds,
                "monitoring_interval_seconds": self.config.monitoring_interval_seconds,
                "retention_days": self.config.retention_days
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"📋 Monitoring report saved to: {filename}")
        return filename

# Console alert callback
def console_alert_callback(alert: Alert):
    """Simple console alert callback"""
    level_emoji = {
        AlertLevel.INFO: "ℹ️",
        AlertLevel.WARNING: "⚠️",
        AlertLevel.ERROR: "❌", 
        AlertLevel.CRITICAL: "🚨"
    }
    
    print(f"{level_emoji.get(alert.level, '📢')} ALERT [{alert.level.value.upper()}] {alert.category}: {alert.message}")

# Convenience functions
def create_monitored_test_runner(config: TestConfiguration = None, monitoring_config: MonitoringConfiguration = None) -> tuple[AutomatedTestRunner, BatchMonitor]:
    """Create test runner with integrated monitoring"""
    test_config = config or TestConfiguration()
    monitor_config = monitoring_config or MonitoringConfiguration()
    
    runner = AutomatedTestRunner(test_config)
    monitor = BatchMonitor(monitor_config)
    
    # Add console alerts
    monitor.alert_manager.add_alert_callback(console_alert_callback)
    
    return runner, monitor 