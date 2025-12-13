import argparse
import sys
import os

# Add src to python path to ensure imports work correctly when running from any location
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph import create_graph

def main():
    parser = argparse.ArgumentParser(description="FinResearch AI - Week 1")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker to research (e.g. AAPL)")
    args = parser.parse_args()
    
    ticker = args.ticker
    print(f"Starting FinResearch AI for {ticker}...")
    
    app = create_graph()
    
    initial_state = {
        "ticker": ticker,
        "messages": [],
        "research_summary": "",
        "financial_data": {},
        "next_step": ""
    }
    
    # Run the graph
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"Finished Node: {key}")
            # print(f"Output: {value}") # Verbose logging if needed
            
    print("\n--- Workflow Completed ---")

if __name__ == "__main__":
    main()
