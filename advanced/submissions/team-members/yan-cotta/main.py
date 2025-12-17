#!/usr/bin/env python3
"""
FinResearch AI - Production CLI Entry Point

Command-line interface for running financial research workflows.

Usage:
    python main.py AAPL                      # Research Apple Inc
    python main.py TSLA --name "Tesla Inc"   # Research with custom name
    python main.py MSFT --sequential         # Use sequential process
    python main.py GOOGL --output report.md  # Custom output filename
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import get_settings, setup_logging
from src.crew import FinResearchCrew, SequentialFinResearchCrew, CrewExecutionResult
from src.tools.memory import MemoryTool


logger = logging.getLogger(__name__)


def validate_environment() -> bool:
    """
    Validate that required environment variables are set.
    
    Returns:
        True if environment is valid, False otherwise
    """
    settings = get_settings()
    
    if not settings.openai_api_key:
        print("\n" + "=" * 60)
        print("ERROR: OPENAI_API_KEY not configured")
        print("=" * 60)
        print("\nPlease set your OpenAI API key:")
        print("  1. Create a .env file in the project root")
        print("  2. Add: OPENAI_API_KEY=sk-your-key-here")
        print("\nOr set the environment variable:")
        print("  export OPENAI_API_KEY='sk-your-key-here'")
        print("=" * 60 + "\n")
        return False
    
    return True


def print_banner() -> None:
    """Print application banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ███████╗██╗███╗   ██╗██████╗ ███████╗███████╗███████╗ █████╗ ██╗   ║
║   ██╔════╝██║████╗  ██║██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██║   ║
║   █████╗  ██║██╔██╗ ██║██████╔╝█████╗  ███████╗█████╗  ███████║██║   ║
║   ██╔══╝  ██║██║╚██╗██║██╔══██╗██╔══╝  ╚════██║██╔══╝  ██╔══██║██║   ║
║   ██║     ██║██║ ╚████║██║  ██║███████╗███████║███████╗██║  ██║██║   ║
║   ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝   ║
║                                                                      ║
║          Multi-Agent Financial Research System v1.0.0                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="finresearch",
        description="FinResearch AI - Multi-Agent Financial Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL                          Research Apple Inc
  python main.py TSLA --name "Tesla Inc"       Research with company name
  python main.py MSFT --sequential             Use sequential process
  python main.py GOOGL -o google_report.md     Custom output filename
  python main.py NVDA --verbose                Enable verbose logging
        """
    )
    
    # Required arguments
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol to research (e.g., AAPL, TSLA, MSFT)"
    )
    
    # Optional arguments
    parser.add_argument(
        "-n", "--name",
        type=str,
        default=None,
        help="Company name (defaults to ticker symbol)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output filename for the report (default: {TICKER}_report.md)"
    )
    
    parser.add_argument(
        "-s", "--sequential",
        action="store_true",
        help="Use sequential process instead of hierarchical"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress banner and progress output"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: stdout only)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running research"
    )
    
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Clear ChromaDB memory before starting (prevents context pollution)"
    )
    
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output execution result as JSON (for UI integration)"
    )
    
    return parser


def run_research(args: argparse.Namespace) -> tuple[int, Optional[CrewExecutionResult]]:
    """
    Execute the research workflow.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple of (exit_code, execution_result)
    """
    ticker = args.ticker.strip().upper()
    company_name = args.name or ticker
    settings = get_settings()
    
    # Use consistent output filename
    output_filename = args.output or f"{ticker}_report.md"
    
    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"FINRESEARCH AI - Starting Research")
        print(f"{'='*60}")
        print(f"   Ticker: {ticker}")
        print(f"   Company: {company_name}")
        print(f"   Process: {'Sequential' if args.sequential else 'Hierarchical'}")
        print(f"   Output: {settings.output_path / output_filename}")
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
    
    logger.info(f"Starting research workflow for {ticker}")
    
    try:
        # Create appropriate crew type
        if args.sequential:
            logger.info("Using Sequential process mode")
            crew = SequentialFinResearchCrew(
                ticker=ticker,
                company_name=company_name,
                verbose=args.verbose
            )
        else:
            logger.info("Using Hierarchical process mode (Manager-led)")
            crew = FinResearchCrew(
                ticker=ticker,
                company_name=company_name,
                verbose=args.verbose
            )
        
        # Execute research
        logger.info("Executing crew workflow...")
        result = crew.run()
        
        # Save report to outputs/ directory
        report_path = crew.save_report(result, filename=output_filename)
        
        # Get execution result for potential JSON output
        execution_result = crew.get_execution_result()
        
        if not args.quiet:
            print("\n" + "=" * 60)
            print("RESEARCH COMPLETE")
            print("=" * 60)
            print(f"\nReport saved to: {report_path}")
            
            # Show validation status
            if execution_result:
                if execution_result.report_valid:
                    print("[✓] Report passed quality validation")
                else:
                    print("[!] Report has validation issues:")
                    for issue in execution_result.validation_issues:
                        print(f"    - {issue}")
                
                if execution_result.duration_seconds:
                    print(f"\nExecution time: {execution_result.duration_seconds:.1f} seconds")
            
            print("\n--- REPORT PREVIEW ---\n")
            print(result[:2000])
            if len(result) > 2000:
                print("\n... [truncated, see full report] ...")
        
        logger.info(f"Research completed successfully for {ticker}")
        return (0, execution_result)
        
    except KeyboardInterrupt:
        logger.warning("Research interrupted by user")
        print("\n\nResearch interrupted by user")
        return (1, None)
        
    except Exception as e:
        logger.exception(f"Research failed for {ticker}")
        print(f"\nERROR: {type(e).__name__}: {e}")
        return (1, None)


def reset_memory(quiet: bool = False) -> bool:
    """
    Reset ChromaDB memory to prevent context pollution between sessions.
    
    Args:
        quiet: Suppress output messages
        
    Returns:
        True if reset successful, False otherwise
    """
    try:
        memory_tool = MemoryTool()
        
        if memory_tool._collection is None:
            if not quiet:
                print("[WARNING] Memory system not available. Nothing to reset.")
            logger.warning("Memory system not available for reset")
            return True
        
        result = memory_tool._clear()
        
        if not quiet:
            print(f"[OK] {result}")
        
        logger.info("Memory reset completed successfully")
        return True
        
    except Exception as e:
        logger.exception("Failed to reset memory")
        if not quiet:
            print(f"[ERROR] Failed to reset memory: {type(e).__name__}: {e}")
        return False


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging with chain-of-thought support
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(
        level=log_level,
        log_file=args.log_file,
        log_chain_of_thought=args.verbose
    )
    
    logger.info("FinResearch AI starting up")
    logger.debug(f"Arguments: {args}")
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Validate environment
    if not validate_environment():
        return 1
    
    # Handle memory reset
    if args.reset_memory:
        if not args.quiet:
            print("\n" + "-" * 60)
            print("MEMORY RESET")
            print("-" * 60)
        
        if not reset_memory(quiet=args.quiet):
            return 1
        
        if not args.quiet:
            print("-" * 60 + "\n")
    
    # Dry run - just validate
    if args.dry_run:
        logger.info("Dry run completed - configuration valid")
        print("Configuration valid. Dry run complete.")
        return 0
    
    # Run research
    exit_code, execution_result = run_research(args)
    
    # Output JSON if requested (for UI integration)
    if args.json_output and execution_result:
        import json
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(execution_result.to_dict(), indent=2, default=str))
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
