#!/usr/bin/env python3
"""
FinResearch AI - Production CLI Entry Point

Command-line interface for running financial research workflows.
Supports both single-run and interactive conversational modes.

Usage:
    python main.py AAPL                      # Research Apple Inc
    python main.py TSLA --name "Tesla Inc"   # Research with custom name
    python main.py MSFT --sequential         # Use sequential process
    python main.py --interactive             # Enter conversational mode
"""

import argparse
import logging
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import get_settings, setup_logging
from src.crew import FinResearchCrew, SequentialFinResearchCrew, CrewExecutionResult
from src.tools.memory import MemoryTool


logger = logging.getLogger(__name__)


# =============================================================================
# Conversation History Management
# =============================================================================

class ConversationContext:
    """
    Manages conversation history and context for interactive mode.
    
    Uses ChromaDB via MemoryTool for persistent context across queries.
    """
    
    def __init__(self):
        """Initialize conversation context with memory tool."""
        self._memory_tool = MemoryTool()
        self._session_history: List[Dict[str, Any]] = []
        self._current_ticker: Optional[str] = None
        self._session_start = datetime.now()
    
    @property
    def current_ticker(self) -> Optional[str]:
        """Get the currently active ticker from conversation context."""
        return self._current_ticker
    
    @current_ticker.setter
    def current_ticker(self, ticker: str) -> None:
        """Set the currently active ticker."""
        self._current_ticker = ticker.upper() if ticker else None
    
    def add_to_history(
        self, 
        query: str, 
        response_summary: str, 
        ticker: str,
        success: bool = True
    ) -> None:
        """
        Add a query/response pair to session history and persist to memory.
        
        Args:
            query: User's original query
            response_summary: Brief summary of the response
            ticker: Stock ticker researched
            success: Whether the query succeeded
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "ticker": ticker,
            "response_summary": response_summary[:200],
            "success": success
        }
        self._session_history.append(entry)
        
        # Persist to ChromaDB for cross-session context
        memory_entry = f"conversation:{ticker}:{query[:100]}:{response_summary[:100]}"
        try:
            self._memory_tool._run(f"save:conversation:{memory_entry}")
        except Exception as e:
            logger.debug(f"Failed to persist conversation to memory: {e}")
    
    def get_context_for_ticker(self, ticker: str) -> str:
        """
        Retrieve previous research context for a ticker from ChromaDB.
        
        Args:
            ticker: Stock ticker to get context for
            
        Returns:
            Context string with previous findings, or empty string
        """
        try:
            result = self._memory_tool._run(f"retrieve:{ticker}")
            if result and "No relevant" not in result:
                return f"\nPrevious research context for {ticker}:\n{result}\n"
        except Exception as e:
            logger.debug(f"Failed to retrieve context: {e}")
        return ""
    
    def get_session_summary(self) -> str:
        """Get a summary of the current session."""
        if not self._session_history:
            return "No queries in this session yet."
        
        duration = datetime.now() - self._session_start
        tickers = set(e["ticker"] for e in self._session_history)
        successful = sum(1 for e in self._session_history if e["success"])
        
        return (
            f"Session Summary:\n"
            f"  Duration: {duration.seconds // 60}m {duration.seconds % 60}s\n"
            f"  Queries: {len(self._session_history)} ({successful} successful)\n"
            f"  Tickers researched: {', '.join(sorted(tickers))}"
        )
    
    def clear_session(self) -> None:
        """Clear session history (keeps ChromaDB context)."""
        self._session_history = []
        self._current_ticker = None
        logger.info("Session history cleared")


# =============================================================================
# Interactive Mode
# =============================================================================

def parse_interactive_query(query: str, current_ticker: Optional[str]) -> tuple[Optional[str], Optional[str], str]:
    """
    Parse user query in interactive mode to extract intent and ticker.
    
    Args:
        query: User's raw input
        current_ticker: Currently active ticker from context
        
    Returns:
        Tuple of (ticker, company_name, action_type)
        action_type: "research", "followup", "context", "help", "exit", "unknown"
    """
    query_lower = query.lower().strip()
    
    # Exit commands
    if query_lower in ("exit", "quit", "q", "bye", "goodbye"):
        return (None, None, "exit")
    
    # Help commands
    if query_lower in ("help", "?", "commands"):
        return (None, None, "help")
    
    # Session/context commands
    if query_lower in ("status", "session", "summary"):
        return (None, None, "status")
    
    if query_lower in ("clear", "reset", "new"):
        return (None, None, "clear")
    
    if query_lower.startswith("context"):
        ticker = query_lower.replace("context", "").strip().upper() or current_ticker
        return (ticker, None, "context")
    
    # Research commands - explicit ticker patterns
    # Pattern: "research AAPL", "analyze TSLA", "look up MSFT"
    research_patterns = [
        r"^(?:research|analyze|look\s*up|check|get|report\s*on)\s+([A-Za-z]{1,5})(?:\s+(.+))?$",
        r"^([A-Za-z]{1,5})\s+(?:research|analysis|report)$",
        r"^([A-Za-z]{1,5})$",  # Just a ticker
    ]
    
    for pattern in research_patterns:
        match = re.match(pattern, query.strip(), re.IGNORECASE)
        if match:
            ticker = match.group(1).upper()
            company_name = match.group(2) if match.lastindex >= 2 else None
            return (ticker, company_name, "research")
    
    # Follow-up queries (use current context)
    if current_ticker:
        followup_keywords = [
            "more", "detail", "explain", "why", "what about",
            "compare", "news", "price", "metrics", "risk"
        ]
        if any(kw in query_lower for kw in followup_keywords):
            return (current_ticker, None, "followup")
    
    # Try to extract any ticker-like pattern from the query
    ticker_match = re.search(r"\b([A-Z]{1,5})\b", query.upper())
    if ticker_match:
        return (ticker_match.group(1), None, "research")
    
    return (None, None, "unknown")


def print_interactive_help() -> None:
    """Print help message for interactive mode."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    INTERACTIVE MODE - COMMANDS                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  RESEARCH:                                                           ║
║    AAPL              - Research a ticker                             ║
║    research TSLA     - Research Tesla                                ║
║    analyze MSFT      - Analyze Microsoft                             ║
║                                                                      ║
║  FOLLOW-UP (uses current ticker):                                    ║
║    more details      - Get more information                          ║
║    what about risks  - Ask follow-up questions                       ║
║                                                                      ║
║  CONTEXT:                                                            ║
║    context           - Show context for current ticker               ║
║    context AAPL      - Show context for specific ticker              ║
║    status            - Show session summary                          ║
║                                                                      ║
║  SESSION:                                                            ║
║    clear             - Start fresh session                           ║
║    exit / quit       - Exit interactive mode                         ║
║    help              - Show this help                                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def run_interactive_mode(args: argparse.Namespace) -> int:
    """
    Run the interactive conversational interface.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code
    """
    context = ConversationContext()
    settings = get_settings()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    FINRESEARCH AI - INTERACTIVE MODE                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  Enter a ticker symbol or command. Type 'help' for commands.         ║
║  Type 'exit' or 'quit' to leave.                                     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    while True:
        try:
            # Show current context in prompt
            prompt_context = f"[{context.current_ticker}]" if context.current_ticker else ""
            user_input = input(f"\nfinresearch{prompt_context}> ").strip()
            
            if not user_input:
                continue
            
            # Parse the query
            ticker, company_name, action = parse_interactive_query(
                user_input, context.current_ticker
            )
            
            # Handle different actions
            if action == "exit":
                print(f"\n{context.get_session_summary()}")
                print("\nGoodbye! Thank you for using FinResearch AI.")
                return 0
            
            elif action == "help":
                print_interactive_help()
                continue
            
            elif action == "status":
                print(f"\n{context.get_session_summary()}")
                if context.current_ticker:
                    print(f"  Current focus: {context.current_ticker}")
                continue
            
            elif action == "clear":
                context.clear_session()
                # Also clear ChromaDB memory
                try:
                    memory_tool = MemoryTool()
                    memory_tool._run("clear")
                except Exception:
                    pass
                print("\n[OK] Session cleared. Ready for new research.")
                continue
            
            elif action == "context":
                if ticker:
                    ctx = context.get_context_for_ticker(ticker)
                    if ctx:
                        print(ctx)
                    else:
                        print(f"\nNo previous context found for {ticker}.")
                else:
                    print("\nNo ticker specified. Use 'context AAPL' or research a ticker first.")
                continue
            
            elif action == "followup":
                # For follow-up queries, retrieve context and provide guidance
                ctx = context.get_context_for_ticker(ticker)
                print(f"\n[Using context for {ticker}]")
                if ctx:
                    print(ctx[:500])
                print(f"\nTo get updated research, type: research {ticker}")
                continue
            
            elif action == "research" and ticker:
                # Run full research
                print(f"\n{'='*60}")
                print(f"Starting research for {ticker}...")
                print(f"{'='*60}")
                
                # Check for previous context
                prev_context = context.get_context_for_ticker(ticker)
                if prev_context:
                    print(f"\n[Note: Found previous research context for {ticker}]")
                
                # Update current ticker
                context.current_ticker = ticker
                
                try:
                    # Create and run crew
                    if args.sequential:
                        crew = SequentialFinResearchCrew(
                            ticker=ticker,
                            company_name=company_name or ticker,
                            verbose=args.verbose
                        )
                    else:
                        crew = FinResearchCrew(
                            ticker=ticker,
                            company_name=company_name or ticker,
                            verbose=args.verbose
                        )
                    
                    result = crew.run()
                    output_filename = f"{ticker}_report.md"
                    report_path = crew.save_report(result, filename=output_filename)
                    
                    # Add to conversation history
                    context.add_to_history(
                        query=user_input,
                        response_summary=f"Generated report for {ticker}",
                        ticker=ticker,
                        success=True
                    )
                    
                    exec_result = crew.get_execution_result()
                    
                    print("\n" + "=" * 60)
                    print("RESEARCH COMPLETE")
                    print("=" * 60)
                    print(f"\nReport saved to: {report_path}")
                    
                    if exec_result:
                        if exec_result.report_valid:
                            print("[✓] Report passed quality validation")
                        else:
                            print("[!] Report has validation issues:")
                            for issue in exec_result.validation_issues:
                                print(f"    - {issue}")
                        
                        if exec_result.duration_seconds:
                            print(f"\nExecution time: {exec_result.duration_seconds:.1f} seconds")
                    
                    print("\n--- REPORT PREVIEW ---\n")
                    print(result[:1500])
                    if len(result) > 1500:
                        print("\n... [truncated, see full report] ...")
                    
                except Exception as e:
                    logger.exception(f"Research failed for {ticker}")
                    context.add_to_history(
                        query=user_input,
                        response_summary=f"Failed: {str(e)[:100]}",
                        ticker=ticker,
                        success=False
                    )
                    print(f"\n[ERROR] Research failed: {type(e).__name__}: {e}")
            
            else:
                print(f"\n[?] I didn't understand that. Try 'help' for commands.")
                print(f"    Or enter a ticker symbol like 'AAPL' or 'research TSLA'")
        
        except KeyboardInterrupt:
            print(f"\n\n{context.get_session_summary()}")
            print("\nInterrupted. Goodbye!")
            return 0
        
        except EOFError:
            print("\nGoodbye!")
            return 0
    
    return 0


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
  python main.py --interactive                 Start conversational mode
  python main.py -i                            Start conversational mode (short)
        """
    )
    
    # Positional argument (optional when using --interactive)
    parser.add_argument(
        "ticker",
        type=str,
        nargs="?",  # Make optional for interactive mode
        default=None,
        help="Stock ticker symbol to research (e.g., AAPL, TSLA, MSFT)"
    )
    
    # Interactive mode flag
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive conversational mode"
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
    
    # Interactive mode
    if args.interactive:
        logger.info("Starting interactive mode")
        return run_interactive_mode(args)
    
    # Single-run mode requires a ticker
    if not args.ticker:
        parser.print_help()
        print("\n[ERROR] Ticker required for single-run mode.")
        print("        Use --interactive (-i) for conversational mode.")
        return 1
    
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
