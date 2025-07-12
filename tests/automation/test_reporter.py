"""
Test Reporter - Comprehensive reporting system for automated test results.
Generates detailed analysis, statistics, and insights from test execution data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import statistics
from collections import defaultdict, Counter
import html as html_escape

# Import our core systems
from tests.automation.test_runner import TestResult, TestStatus, TestConfiguration
from tests.user_simulator.scenarios import ConversationScenarios
from tests.user_simulator.personas import get_all_persona_configs

@dataclass
class ReportConfiguration:
    """Configuration for report generation"""
    include_detailed_logs: bool = True
    include_flow_analysis: bool = True
    include_statistical_analysis: bool = True
    include_recommendations: bool = True
    generate_html_report: bool = True
    generate_json_report: bool = True
    generate_csv_summary: bool = True
    output_directory: str = "test_reports"

@dataclass
class ScenarioAnalysis:
    """Analysis for a specific scenario"""
    scenario_id: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    success_rate: float
    average_duration: float
    average_validation_score: float
    common_issues: List[str]
    persona_performance: Dict[str, float]  # persona_name -> success_rate

@dataclass
class PersonaAnalysis:
    """Analysis for a specific persona"""
    persona_name: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    success_rate: float
    average_duration: float
    average_validation_score: float
    scenario_performance: Dict[str, float]  # scenario_id -> success_rate
    common_issues: List[str]

@dataclass
class TestReport:
    """Comprehensive test report"""
    report_id: str
    generation_time: datetime
    execution_summary: Dict[str, Any]
    scenario_analysis: List[ScenarioAnalysis]
    persona_analysis: List[PersonaAnalysis]
    statistical_insights: Dict[str, Any]
    recommendations: List[str]
    raw_data: List[Dict[str, Any]]

class TestReporter:
    """Main test reporting engine"""
    
    def __init__(self, config: ReportConfiguration = None):
        self.config = config or ReportConfiguration()
        self.logger = logging.getLogger('test_reporter')
        self._setup_directories()
    
    def _setup_directories(self):
        """Setup necessary directories for report generation"""
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, test_results: List[TestResult], execution_info: Dict[str, Any] = None) -> TestReport:
        """Generate comprehensive report from test results"""
        self.logger.info(f"📊 Generating comprehensive report for {len(test_results)} test results")
        
        report = TestReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generation_time=datetime.now(),
            execution_summary=self._generate_execution_summary(test_results, execution_info),
            scenario_analysis=self._analyze_scenarios(test_results),
            persona_analysis=self._analyze_personas(test_results),
            statistical_insights=self._generate_statistical_insights(test_results),
            recommendations=self._generate_recommendations(test_results),
            raw_data=[result.to_dict() for result in test_results]
        )
        
        self.logger.info(f"✅ Report generated: {report.report_id}")
        return report
    
    def _generate_execution_summary(self, test_results: List[TestResult], execution_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate execution summary statistics"""
        if not test_results:
            return {"status": "no_tests"}
        
        total_tests = len(test_results)
        successful_tests = len([r for r in test_results if r.status == TestStatus.COMPLETED])
        failed_tests = len([r for r in test_results if r.status == TestStatus.FAILED])
        timeout_tests = len([r for r in test_results if r.status == TestStatus.TIMEOUT])
        
        durations = [r.duration_seconds for r in test_results if r.duration_seconds > 0]
        validation_scores = [r.validation_score for r in test_results if r.validation_score > 0]
        
        summary = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "timeout_tests": timeout_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "failure_rate": failed_tests / total_tests if total_tests > 0 else 0,
            "timeout_rate": timeout_tests / total_tests if total_tests > 0 else 0,
        }
        
        if durations:
            summary.update({
                "average_duration_seconds": statistics.mean(durations),
                "median_duration_seconds": statistics.median(durations),
                "min_duration_seconds": min(durations),
                "max_duration_seconds": max(durations),
                "duration_std_dev": statistics.stdev(durations) if len(durations) > 1 else 0
            })
        
        if validation_scores:
            summary.update({
                "average_validation_score": statistics.mean(validation_scores),
                "median_validation_score": statistics.median(validation_scores),
                "min_validation_score": min(validation_scores),
                "max_validation_score": max(validation_scores),
                "high_scoring_tests": len([s for s in validation_scores if s >= 0.8]),
                "low_scoring_tests": len([s for s in validation_scores if s < 0.6])
            })
        
        if execution_info:
            summary.update(execution_info)
        
        return summary
    
    def _analyze_scenarios(self, test_results: List[TestResult]) -> List[ScenarioAnalysis]:
        """Analyze performance by scenario"""
        scenario_data = defaultdict(list)
        
        # Group results by scenario
        for result in test_results:
            scenario_data[result.test_case.scenario_id].append(result)
        
        scenario_analyses = []
        
        for scenario_id, results in scenario_data.items():
            total_tests = len(results)
            successful_tests = len([r for r in results if r.status == TestStatus.COMPLETED])
            failed_tests = len([r for r in results if r.status == TestStatus.FAILED])
            
            durations = [r.duration_seconds for r in results if r.duration_seconds > 0]
            validation_scores = [r.validation_score for r in results if r.validation_score > 0]
            
            # Persona performance for this scenario
            persona_performance = {}
            persona_groups = defaultdict(list)
            for result in results:
                persona_groups[result.test_case.persona_name].append(result)
            
            for persona_name, persona_results in persona_groups.items():
                persona_successful = len([r for r in persona_results if r.status == TestStatus.COMPLETED])
                persona_total = len(persona_results)
                persona_performance[persona_name] = persona_successful / persona_total if persona_total > 0 else 0
            
            # Common issues
            all_errors = []
            for result in results:
                all_errors.extend(result.errors)
            common_issues = [issue for issue, count in Counter(all_errors).most_common(3)]
            
            analysis = ScenarioAnalysis(
                scenario_id=scenario_id,
                total_tests=total_tests,
                successful_tests=successful_tests,
                failed_tests=failed_tests,
                success_rate=successful_tests / total_tests if total_tests > 0 else 0,
                average_duration=statistics.mean(durations) if durations else 0,
                average_validation_score=statistics.mean(validation_scores) if validation_scores else 0,
                common_issues=common_issues,
                persona_performance=persona_performance
            )
            
            scenario_analyses.append(analysis)
        
        # Sort by success rate (lowest first to highlight problems)
        scenario_analyses.sort(key=lambda x: x.success_rate)
        
        return scenario_analyses
    
    def _analyze_personas(self, test_results: List[TestResult]) -> List[PersonaAnalysis]:
        """Analyze performance by persona"""
        persona_data = defaultdict(list)
        
        # Group results by persona
        for result in test_results:
            persona_data[result.test_case.persona_name].append(result)
        
        persona_analyses = []
        
        for persona_name, results in persona_data.items():
            total_tests = len(results)
            successful_tests = len([r for r in results if r.status == TestStatus.COMPLETED])
            failed_tests = len([r for r in results if r.status == TestStatus.FAILED])
            
            durations = [r.duration_seconds for r in results if r.duration_seconds > 0]
            validation_scores = [r.validation_score for r in results if r.validation_score > 0]
            
            # Scenario performance for this persona
            scenario_performance = {}
            scenario_groups = defaultdict(list)
            for result in results:
                scenario_groups[result.test_case.scenario_id].append(result)
            
            for scenario_id, scenario_results in scenario_groups.items():
                scenario_successful = len([r for r in scenario_results if r.status == TestStatus.COMPLETED])
                scenario_total = len(scenario_results)
                scenario_performance[scenario_id] = scenario_successful / scenario_total if scenario_total > 0 else 0
            
            # Common issues
            all_errors = []
            for result in results:
                all_errors.extend(result.errors)
            common_issues = [issue for issue, count in Counter(all_errors).most_common(3)]
            
            analysis = PersonaAnalysis(
                persona_name=persona_name,
                total_tests=total_tests,
                successful_tests=successful_tests,
                failed_tests=failed_tests,
                success_rate=successful_tests / total_tests if total_tests > 0 else 0,
                average_duration=statistics.mean(durations) if durations else 0,
                average_validation_score=statistics.mean(validation_scores) if validation_scores else 0,
                scenario_performance=scenario_performance,
                common_issues=common_issues
            )
            
            persona_analyses.append(analysis)
        
        # Sort by success rate (lowest first to highlight problems)
        persona_analyses.sort(key=lambda x: x.success_rate)
        
        return persona_analyses
    
    def _generate_statistical_insights(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Generate statistical insights and patterns"""
        if not test_results:
            return {}
        
        insights = {}
        
        # Performance distribution
        validation_scores = [r.validation_score for r in test_results if r.validation_score > 0]
        if validation_scores:
            insights["score_distribution"] = {
                "excellent": len([s for s in validation_scores if s >= 0.9]),
                "good": len([s for s in validation_scores if 0.8 <= s < 0.9]),
                "fair": len([s for s in validation_scores if 0.6 <= s < 0.8]),
                "poor": len([s for s in validation_scores if s < 0.6])
            }
        
        # Duration patterns
        durations = [r.duration_seconds for r in test_results if r.duration_seconds > 0]
        if durations:
            insights["duration_patterns"] = {
                "very_fast": len([d for d in durations if d < 10]),
                "fast": len([d for d in durations if 10 <= d < 30]),
                "normal": len([d for d in durations if 30 <= d < 60]),
                "slow": len([d for d in durations if d >= 60])
            }
        
        # Error patterns
        all_errors = []
        for result in test_results:
            all_errors.extend(result.errors)
        
        if all_errors:
            error_counts = Counter(all_errors)
            insights["top_errors"] = [
                {"error": error, "count": count, "percentage": count / len(test_results) * 100}
                for error, count in error_counts.most_common(5)
            ]
        
        # Success patterns by combination
        combination_performance = defaultdict(list)
        for result in test_results:
            key = f"{result.test_case.scenario_id}_{result.test_case.persona_name}"
            combination_performance[key].append(result.status == TestStatus.COMPLETED)
        
        best_combinations = []
        worst_combinations = []
        
        for combo, successes in combination_performance.items():
            success_rate = sum(successes) / len(successes)
            scenario_id, persona_name = combo.split('_', 1)
            
            combo_data = {
                "scenario_id": scenario_id,
                "persona_name": persona_name,
                "success_rate": success_rate,
                "total_tests": len(successes)
            }
            
            if success_rate >= 0.9 and len(successes) >= 3:
                best_combinations.append(combo_data)
            elif success_rate <= 0.5 and len(successes) >= 3:
                worst_combinations.append(combo_data)
        
        insights["performance_combinations"] = {
            "best_combinations": sorted(best_combinations, key=lambda x: x["success_rate"], reverse=True)[:5],
            "worst_combinations": sorted(worst_combinations, key=lambda x: x["success_rate"])[:5]
        }
        
        return insights
    
    def _generate_recommendations(self, test_results: List[TestResult]) -> List[str]:
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        if not test_results:
            recommendations.append("No test results available for analysis")
            return recommendations
        
        # Overall success rate analysis
        total_tests = len(test_results)
        successful_tests = len([r for r in test_results if r.status == TestStatus.COMPLETED])
        success_rate = successful_tests / total_tests
        
        if success_rate < 0.7:
            recommendations.append(f"🚨 Low overall success rate ({success_rate:.1%}). Consider reviewing test scenarios and system implementation.")
        elif success_rate < 0.85:
            recommendations.append(f"⚠️ Moderate success rate ({success_rate:.1%}). Some improvements needed.")
        else:
            recommendations.append(f"✅ Good overall success rate ({success_rate:.1%}). System performing well.")
        
        # Scenario-specific recommendations
        scenario_data = defaultdict(list)
        for result in test_results:
            scenario_data[result.test_case.scenario_id].append(result)
        
        for scenario_id, results in scenario_data.items():
            scenario_success_rate = len([r for r in results if r.status == TestStatus.COMPLETED]) / len(results)
            
            if scenario_success_rate < 0.5:
                recommendations.append(f"🔴 Scenario '{scenario_id}' has low success rate ({scenario_success_rate:.1%}). Review scenario logic and expectations.")
        
        # Persona-specific recommendations
        persona_data = defaultdict(list)
        for result in test_results:
            persona_data[result.test_case.persona_name].append(result)
        
        for persona_name, results in persona_data.items():
            persona_success_rate = len([r for r in results if r.status == TestStatus.COMPLETED]) / len(results)
            
            if persona_success_rate < 0.5:
                recommendations.append(f"🔴 Persona '{persona_name}' has low success rate ({persona_success_rate:.1%}). Review persona behavior patterns.")
        
        # Performance recommendations
        durations = [r.duration_seconds for r in test_results if r.duration_seconds > 0]
        if durations:
            avg_duration = statistics.mean(durations)
            if avg_duration > 60:
                recommendations.append(f"⏱️ Average test duration is high ({avg_duration:.1f}s). Consider optimizing conversation flows.")
        
        # Validation score recommendations
        validation_scores = [r.validation_score for r in test_results if r.validation_score > 0]
        if validation_scores:
            avg_score = statistics.mean(validation_scores)
            if avg_score < 0.7:
                recommendations.append(f"📊 Low average validation score ({avg_score:.2f}). Review agent flow patterns and success criteria.")
        
        # Error-based recommendations
        all_errors = []
        for result in test_results:
            all_errors.extend(result.errors)
        
        if all_errors:
            error_counts = Counter(all_errors)
            most_common_error = error_counts.most_common(1)[0]
            error_rate = most_common_error[1] / total_tests
            
            if error_rate > 0.2:
                recommendations.append(f"🐛 Common error affecting {error_rate:.1%} of tests: '{most_common_error[0]}'. Prioritize fixing this issue.")
        
        return recommendations
    
    def save_json_report(self, report: TestReport, filename: Optional[str] = None) -> str:
        """Save report as JSON file"""
        if filename is None:
            filename = f"{report.report_id}.json"
        
        filepath = Path(self.config.output_directory) / filename
        
        report_data = {
            "report_id": report.report_id,
            "generation_time": report.generation_time.isoformat(),
            "execution_summary": report.execution_summary,
            "scenario_analysis": [
                {
                    "scenario_id": sa.scenario_id,
                    "total_tests": sa.total_tests,
                    "successful_tests": sa.successful_tests,
                    "failed_tests": sa.failed_tests,
                    "success_rate": sa.success_rate,
                    "average_duration": sa.average_duration,
                    "average_validation_score": sa.average_validation_score,
                    "common_issues": sa.common_issues,
                    "persona_performance": sa.persona_performance
                }
                for sa in report.scenario_analysis
            ],
            "persona_analysis": [
                {
                    "persona_name": pa.persona_name,
                    "total_tests": pa.total_tests,
                    "successful_tests": pa.successful_tests,
                    "failed_tests": pa.failed_tests,
                    "success_rate": pa.success_rate,
                    "average_duration": pa.average_duration,
                    "average_validation_score": pa.average_validation_score,
                    "scenario_performance": pa.scenario_performance,
                    "common_issues": pa.common_issues
                }
                for pa in report.persona_analysis
            ],
            "statistical_insights": report.statistical_insights,
            "recommendations": report.recommendations,
            "raw_data": report.raw_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 JSON report saved to: {filepath}")
        return str(filepath)
    
    def save_html_report(self, report: TestReport, filename: Optional[str] = None) -> str:
        """Save report as HTML file"""
        if filename is None:
            filename = f"{report.report_id}.html"
        
        filepath = Path(self.config.output_directory) / filename
        
        html_content = self._generate_html_content(report)
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"📄 HTML report saved to: {filepath}")
        return str(filepath)
    
    def _generate_html_content(self, report: TestReport) -> str:
        """Generate HTML content for the report"""
        # This is a simplified HTML template - in production, you'd want a more sophisticated template engine
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report - {report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ background-color: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
        .section {{ margin: 30px 0; }}
        .section h2 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f8f9fa; }}
        .recommendation {{ background-color: #e3f2fd; border-left: 4px solid #2196f3; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Execution Report</h1>
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Generated:</strong> {report.generation_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Execution Summary</h2>
        <div class="summary">
            <div class="metric">
                <div class="metric-value success">{report.execution_summary.get('successful_tests', 0)}</div>
                <div class="metric-label">Successful Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value danger">{report.execution_summary.get('failed_tests', 0)}</div>
                <div class="metric-label">Failed Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report.execution_summary.get('success_rate', 0):.1%}</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report.execution_summary.get('average_duration_seconds', 0):.1f}s</div>
                <div class="metric-label">Avg Duration</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Scenario Analysis</h2>
        <table>
            <tr>
                <th>Scenario ID</th>
                <th>Total Tests</th>
                <th>Success Rate</th>
                <th>Avg Duration</th>
                <th>Avg Score</th>
            </tr>
        """
        
        for scenario in report.scenario_analysis:
            success_class = "success" if scenario.success_rate >= 0.8 else "warning" if scenario.success_rate >= 0.6 else "danger"
            html += f"""
            <tr>
                <td>{html_escape.escape(scenario.scenario_id)}</td>
                <td>{scenario.total_tests}</td>
                <td class="{success_class}">{scenario.success_rate:.1%}</td>
                <td>{scenario.average_duration:.1f}s</td>
                <td>{scenario.average_validation_score:.2f}</td>
            </tr>
            """
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Persona Analysis</h2>
        <table>
            <tr>
                <th>Persona</th>
                <th>Total Tests</th>
                <th>Success Rate</th>
                <th>Avg Duration</th>
                <th>Avg Score</th>
            </tr>
        """
        
        for persona in report.persona_analysis:
            success_class = "success" if persona.success_rate >= 0.8 else "warning" if persona.success_rate >= 0.6 else "danger"
            html += f"""
            <tr>
                <td>{html_escape.escape(persona.persona_name)}</td>
                <td>{persona.total_tests}</td>
                <td class="{success_class}">{persona.success_rate:.1%}</td>
                <td>{persona.average_duration:.1f}s</td>
                <td>{persona.average_validation_score:.2f}</td>
            </tr>
            """
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        """
        
        for recommendation in report.recommendations:
            html += f'<div class="recommendation">{html_escape.escape(recommendation)}</div>'
        
        html += """
    </div>
    
</body>
</html>
        """
        
        return html

# Convenience functions
def generate_report_from_results_file(results_file: str, config: ReportConfiguration = None) -> TestReport:
    """Generate report from saved test results JSON file"""
    reporter = TestReporter(config)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Convert back to TestResult objects (simplified)
    test_results = []
    for result_data in data.get("test_results", []):
        # This would need proper deserialization in a full implementation
        pass
    
    execution_info = data.get("execution_info", {})
    return reporter.generate_comprehensive_report(test_results, execution_info)

def generate_quick_summary(test_results: List[TestResult]) -> str:
    """Generate a quick text summary of test results"""
    if not test_results:
        return "No test results to summarize"
    
    total = len(test_results)
    successful = len([r for r in test_results if r.status == TestStatus.COMPLETED])
    failed = len([r for r in test_results if r.status == TestStatus.FAILED])
    
    summary = f"""
📊 Test Execution Summary:
• Total Tests: {total}
• Successful: {successful} ({successful/total:.1%})
• Failed: {failed} ({failed/total:.1%})
• Success Rate: {successful/total:.1%}
"""
    
    if test_results:
        durations = [r.duration_seconds for r in test_results if r.duration_seconds > 0]
        if durations:
            summary += f"• Average Duration: {statistics.mean(durations):.1f}s\n"
        
        validation_scores = [r.validation_score for r in test_results if r.validation_score > 0]
        if validation_scores:
            summary += f"• Average Validation Score: {statistics.mean(validation_scores):.2f}\n"
    
    return summary 