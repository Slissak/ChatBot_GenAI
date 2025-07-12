#!/usr/bin/env python3
"""
Phase 4 Automation Demo - Complete demonstration of the automated testing system.
Shows how to run comprehensive test suites, validate agent flows, and generate detailed reports.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import asyncio
import logging
from datetime import datetime
from typing import List
import json

# Setup logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('automation_demo.log')
    ]
)

# Import our automation systems
from tests.automation.test_runner import (
    AutomatedTestRunner, TestConfiguration, TestCase, TestPriority,
    run_single_test_async, run_comprehensive_test_suite_async
)
from tests.automation.test_reporter import (
    TestReporter, ReportConfiguration, generate_quick_summary
)
from tests.user_simulator.scenarios import ConversationScenarios
from tests.user_simulator.personas import get_all_persona_configs

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")

async def demo_single_test_execution():
    """Demonstrate executing a single test case"""
    print_section("SINGLE TEST EXECUTION DEMO")
    
    print("🚀 Running single test: Basic Information Request with Eager persona")
    
    # Configure test execution
    config = TestConfiguration(
        max_parallel_tests=1,
        test_timeout_seconds=30,
        validate_flows=True,
        capture_detailed_logs=True
    )
    
    # Run single test
    result = await run_single_test_async(
        scenario_id="info_query_interested",
        persona_name="eager",
        config=config
    )
    
    if result:
        print(f"✅ Test completed successfully!")
        print(f"   Status: {result.status.value}")
        print(f"   Duration: {result.duration_seconds:.1f}s")
        print(f"   Validation Score: {result.validation_score:.2f}")
        print(f"   Conversation ID: {result.conversation_id}")
        
        if result.errors:
            print(f"   Errors: {', '.join(result.errors)}")
        if result.warnings:
            print(f"   Warnings: {', '.join(result.warnings)}")
            
        print(f"   Metrics: {result.metrics}")
    else:
        print("❌ Test execution failed")

async def demo_batch_test_execution():
    """Demonstrate batch test execution with multiple scenarios and personas"""
    print_section("BATCH TEST EXECUTION DEMO")
    
    print("🔄 Setting up automated test runner...")
    
    config = TestConfiguration(
        max_parallel_tests=3,
        test_timeout_seconds=60,
        validate_flows=True,
        generate_reports=True,
        results_directory="demo_results"
    )
    
    runner = AutomatedTestRunner(config)
    
    # Add specific test cases
    print_subsection("Adding Test Cases")
    
    test_scenarios = [
        ("info_query_interested", "eager", TestPriority.HIGH),
        ("info_query_not_interested", "skeptical", TestPriority.NORMAL),
        ("direct_scheduling", "direct", TestPriority.HIGH),
        ("indecisive_journey", "indecisive", TestPriority.NORMAL),
        ("immediate_rejection", "disinterested", TestPriority.LOW),
        ("detailed_exploration", "detail_oriented", TestPriority.NORMAL)
    ]
    
    for scenario_id, persona_name, priority in test_scenarios:
        runner.add_scenario_persona_combination(scenario_id, persona_name, priority)
        print(f"  📋 Added: {scenario_id} + {persona_name} (Priority: {priority.name})")
    
    print(f"\n🚀 Executing {len(runner.test_queue)} test cases...")
    start_time = datetime.now()
    
    # Run the test suite
    results = await runner.run_test_suite()
    
    execution_time = (datetime.now() - start_time).total_seconds()
    print(f"⏱️ Execution completed in {execution_time:.1f} seconds")
    
    # Display results summary
    print_subsection("Execution Results")
    successful = len([r for r in results if r.status.value == "completed"])
    failed = len([r for r in results if r.status.value == "failed"])
    
    print(f"✅ Successful tests: {successful}")
    print(f"❌ Failed tests: {failed}")
    print(f"📊 Success rate: {successful/len(results):.1%}")
    
    # Show individual results
    print_subsection("Individual Test Results")
    for result in results:
        status_emoji = "✅" if result.status.value == "completed" else "❌"
        print(f"{status_emoji} {result.test_case.scenario_id} + {result.test_case.persona_name}")
        print(f"   Duration: {result.duration_seconds:.1f}s | Score: {result.validation_score:.2f}")
        if result.errors:
            print(f"   Errors: {result.errors[0]}")
    
    # Save results
    results_file = runner.save_results()
    print(f"\n💾 Results saved to: {results_file}")
    
    return results

async def demo_comprehensive_test_suite():
    """Demonstrate comprehensive test suite covering all scenarios and personas"""
    print_section("COMPREHENSIVE TEST SUITE DEMO")
    
    print("🌟 Running comprehensive test suite...")
    print("This will test all scenario-persona combinations")
    
    # Get available scenarios and personas
    scenarios = ConversationScenarios.get_all_scenarios()
    personas = get_all_persona_configs()
    
    print(f"📊 Coverage: {len(scenarios)} scenarios × {len(personas)} personas")
    print(f"   Scenarios: {', '.join(scenarios.keys())}")
    print(f"   Personas: {', '.join(personas.keys())}")
    
    config = TestConfiguration(
        max_parallel_tests=5,
        test_timeout_seconds=45,
        validate_flows=True,
        results_directory="comprehensive_results"
    )
    
    # Run comprehensive suite (using a subset for demo purposes)
    print("\n🚀 Executing comprehensive test suite (subset for demo)...")
    start_time = datetime.now()
    
    # For demo, we'll run a representative subset
    runner = AutomatedTestRunner(config)
    
    # Add strategic combinations
    demo_combinations = [
        ("info_query_interested", "eager"),
        ("info_query_not_interested", "skeptical"),
        ("detailed_exploration", "detail_oriented"),
        ("direct_scheduling", "direct"),
        ("indecisive_journey", "indecisive"),
        ("immediate_rejection", "disinterested"),
        ("full_journey_success", "eager"),
        ("skeptical_evaluation", "skeptical")
    ]
    
    for scenario_id, persona_name in demo_combinations:
        if scenario_id in scenarios:
            runner.add_scenario_persona_combination(scenario_id, persona_name)
    
    print(f"   Running {len(runner.test_queue)} strategic test combinations...")
    
    results = await runner.run_test_suite()
    
    execution_time = (datetime.now() - start_time).total_seconds()
    print(f"⏱️ Comprehensive testing completed in {execution_time:.1f} seconds")
    
    # Statistics
    stats = runner.get_execution_statistics()
    print(f"\n📈 Execution Statistics:")
    print(f"   Success Rate: {stats['success_rate']:.1%}")
    print(f"   Average Duration: {stats['average_duration_seconds']:.1f}s")
    print(f"   Average Validation Score: {stats['average_validation_score']:.2f}")
    print(f"   Tests per Minute: {stats['tests_per_minute']:.1f}")
    
    return results

def demo_comprehensive_reporting(test_results):
    """Demonstrate comprehensive reporting capabilities"""
    print_section("COMPREHENSIVE REPORTING DEMO")
    
    print("📊 Generating comprehensive test report...")
    
    # Configure reporting
    report_config = ReportConfiguration(
        include_detailed_logs=True,
        include_flow_analysis=True,
        include_statistical_analysis=True,
        include_recommendations=True,
        generate_html_report=True,
        generate_json_report=True,
        output_directory="demo_reports"
    )
    
    reporter = TestReporter(report_config)
    
    # Generate comprehensive report
    report = reporter.generate_comprehensive_report(test_results)
    
    print(f"✅ Report generated: {report.report_id}")
    
    print_subsection("Execution Summary")
    summary = report.execution_summary
    print(f"   Total Tests: {summary.get('total_tests', 0)}")
    print(f"   Success Rate: {summary.get('success_rate', 0):.1%}")
    print(f"   Average Duration: {summary.get('average_duration_seconds', 0):.1f}s")
    print(f"   Average Validation Score: {summary.get('average_validation_score', 0):.2f}")
    
    print_subsection("Top Performing Scenarios")
    top_scenarios = sorted(report.scenario_analysis, key=lambda x: x.success_rate, reverse=True)[:3]
    for scenario in top_scenarios:
        print(f"   🏆 {scenario.scenario_id}: {scenario.success_rate:.1%} success rate")
    
    print_subsection("Challenging Scenarios")
    challenging_scenarios = sorted(report.scenario_analysis, key=lambda x: x.success_rate)[:3]
    for scenario in challenging_scenarios:
        print(f"   🚨 {scenario.scenario_id}: {scenario.success_rate:.1%} success rate")
    
    print_subsection("Persona Performance")
    for persona in report.persona_analysis:
        print(f"   👤 {persona.persona_name}: {persona.success_rate:.1%} success ({persona.total_tests} tests)")
    
    print_subsection("Key Recommendations")
    for i, recommendation in enumerate(report.recommendations[:5], 1):
        print(f"   {i}. {recommendation}")
    
    # Save reports
    print_subsection("Saving Reports")
    json_file = reporter.save_json_report(report)
    html_file = reporter.save_html_report(report)
    
    print(f"   📄 JSON Report: {json_file}")
    print(f"   🌐 HTML Report: {html_file}")
    
    # Generate quick summary
    print_subsection("Quick Summary")
    quick_summary = generate_quick_summary(test_results)
    print(quick_summary)
    
    return report

def demo_statistical_insights(test_results):
    """Demonstrate statistical analysis capabilities"""
    print_section("STATISTICAL INSIGHTS DEMO")
    
    print("📈 Analyzing statistical patterns...")
    
    reporter = TestReporter()
    insights = reporter._generate_statistical_insights(test_results)
    
    if "score_distribution" in insights:
        print_subsection("Validation Score Distribution")
        dist = insights["score_distribution"]
        print(f"   🌟 Excellent (≥90%): {dist.get('excellent', 0)} tests")
        print(f"   👍 Good (80-89%): {dist.get('good', 0)} tests")
        print(f"   👌 Fair (60-79%): {dist.get('fair', 0)} tests")
        print(f"   👎 Poor (<60%): {dist.get('poor', 0)} tests")
    
    if "duration_patterns" in insights:
        print_subsection("Duration Patterns")
        patterns = insights["duration_patterns"]
        print(f"   ⚡ Very Fast (<10s): {patterns.get('very_fast', 0)} tests")
        print(f"   🏃 Fast (10-30s): {patterns.get('fast', 0)} tests")
        print(f"   🚶 Normal (30-60s): {patterns.get('normal', 0)} tests")
        print(f"   🐌 Slow (≥60s): {patterns.get('slow', 0)} tests")
    
    if "top_errors" in insights:
        print_subsection("Most Common Issues")
        for error_info in insights["top_errors"]:
            print(f"   🐛 {error_info['error']} ({error_info['count']} times, {error_info['percentage']:.1f}%)")
    
    if "performance_combinations" in insights:
        perf = insights["performance_combinations"]
        
        if perf.get("best_combinations"):
            print_subsection("Best Performing Combinations")
            for combo in perf["best_combinations"]:
                print(f"   🏅 {combo['scenario_id']} + {combo['persona_name']}: {combo['success_rate']:.1%}")
        
        if perf.get("worst_combinations"):
            print_subsection("Challenging Combinations")
            for combo in perf["worst_combinations"]:
                print(f"   ⚠️ {combo['scenario_id']} + {combo['persona_name']}: {combo['success_rate']:.1%}")

async def main():
    """Main demonstration function"""
    print_section("🤖 PHASE 4: AUTOMATED TESTING SYSTEM DEMO")
    print("This demo showcases the complete automated testing pipeline:")
    print("• Automated test execution with multiple scenarios and personas")
    print("• Agent flow validation and scoring")
    print("• Comprehensive reporting and statistical analysis")
    print("• Performance insights and recommendations")
    
    try:
        # Demo 1: Single test execution
        await demo_single_test_execution()
        
        # Demo 2: Batch test execution
        batch_results = await demo_batch_test_execution()
        
        # Demo 3: Comprehensive test suite
        comprehensive_results = await demo_comprehensive_test_suite()
        
        # Use comprehensive results for reporting demos
        results_for_reporting = comprehensive_results if comprehensive_results else batch_results
        
        if results_for_reporting:
            # Demo 4: Comprehensive reporting
            report = demo_comprehensive_reporting(results_for_reporting)
            
            # Demo 5: Statistical insights
            demo_statistical_insights(results_for_reporting)
        else:
            print("\n⚠️ No test results available for reporting demos")
        
        print_section("🎉 DEMO COMPLETED SUCCESSFULLY")
        print("Key Phase 4 capabilities demonstrated:")
        print("✅ Automated test case generation and execution")
        print("✅ Parallel test execution with configurable limits")
        print("✅ Agent flow validation and scoring")
        print("✅ Comprehensive statistical analysis")
        print("✅ Detailed HTML and JSON reporting")
        print("✅ Performance insights and recommendations")
        print("✅ Error pattern analysis")
        print("✅ Scenario and persona performance tracking")
        
        print("\n📁 Generated Files:")
        print("• automation_demo.log - Execution logs")
        print("• demo_results/ - Test execution results")
        print("• comprehensive_results/ - Comprehensive test data")
        print("• demo_reports/ - HTML and JSON reports")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main()) 