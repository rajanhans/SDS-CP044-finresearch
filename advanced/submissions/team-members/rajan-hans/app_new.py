# app_new.py
# Streamlit application for FinResearch AI
# Uses pure CrewAI with auto sector detection
# This is the APPLICATION ENTRY POINT - load_dotenv() is called here
# ---------------------------------------------

import os
import streamlit as st
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables ONCE at application entry point
load_dotenv()

from app_service_new import run_full_research
from charts import (
    create_score_radar_chart,
    create_score_gauges_row,
    create_sector_comparison_chart,
    create_final_score_display,
    create_technical_chart
)
from tradingview_widgets import (
    render_advanced_chart,
    render_technical_analysis,
    render_symbol_overview,
    render_mini_chart,
    detect_exchange,
    get_exchange_prefix,
)


def get_secret(key: str, default: str = None) -> str:
    """
    Get a secret/environment variable from multiple sources.
    
    Priority order:
    1. os.environ (includes .env via load_dotenv)
    2. st.secrets (Streamlit Cloud secrets.toml)
    3. default value
    
    Args:
        key: The environment variable name
        default: Default value if not found anywhere
    
    Returns:
        The secret value or default
    """
    # First try os.environ (includes .env file)
    value = os.environ.get(key)
    if value:
        return value
    
    # Then try Streamlit secrets
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            # Also set in os.environ so other modules can access it
            value = st.secrets[key]
            os.environ[key] = value
            return value
    except Exception:
        # st.secrets may not be available in all contexts
        pass
    
    return default


def load_secrets_to_environ():
    """
    Load all Streamlit secrets into os.environ.
    This ensures compatibility with modules that use os.getenv().
    
    Call this once at startup after load_dotenv().
    """
    try:
        if hasattr(st, 'secrets'):
            for key in st.secrets:
                # Only set if not already in environ (don't override .env)
                if key not in os.environ:
                    os.environ[key] = str(st.secrets[key])
    except Exception:
        # st.secrets may not be available
        pass


# Load Streamlit secrets into environment (for Streamlit Cloud)
load_secrets_to_environ()


def validate_environment():
    """
    Validate required and optional environment variables at startup.
    Checks both .env file and Streamlit secrets.
    Shows warnings/errors in Streamlit UI.
    """
    # Required API keys
    required_keys = ["OPENAI_API_KEY"]
    missing_required = [k for k in required_keys if not get_secret(k)]

    if missing_required:
        st.error(
            f"⚠️ Missing required API keys: **{', '.join(missing_required)}**\n\n"
            "Please set these in one of the following locations:\n"
            "- **Local development**: `.env` file in project root\n"
            "- **Streamlit Cloud**: Settings → Secrets (TOML format)\n\n"
            "Example secrets.toml format:\n"
            "```toml\n"
            'OPENAI_API_KEY = "sk-your-key-here"\n'
            "```"
        )
        st.stop()

    # Optional API keys (for news search)
    optional_keys = ["TAVILY_API_KEY", "SERPAPI_API_KEY"]
    missing_optional = [k for k in optional_keys if not get_secret(k)]

    if missing_optional:
        st.sidebar.warning(
            f"Optional keys not configured: {', '.join(missing_optional)}\n\n"
            "News search will use free sources only."
        )


def render_glossary():
    """Render the static definitions page."""

    glossary = {
        "Valuation Metrics": {
            "P/E (Price-to-Earnings)": "Ratio of a company's share price to its earnings per share. High P/E suggests investors expect higher growth.",
            "Forward P/E": "Price-to-Earnings ratio using forecasted earnings for the next 12 months.",
            "PEG Ratio": "P/E ratio divided by the growth rate of its earnings. A PEG < 1.0 is often considered undervalued.",
            "P/S (Price-to-Sales)": "Ratio of a company's stock price to its revenues. Useful for valuing unprofitable companies.",
            "P/B (Price-to-Book)": "Ratio of the market value of equity to the book value of equity (assets minus liabilities).",
            "EV/EBITDA": "Enterprise Value divided by Earnings Before Interest, Taxes, Depreciation, and Amortization. A capital-structure-neutral valuation metric.",
            "FCF Yield": "Free Cash Flow per share divided by the share price. Represents the cash return on investment.",
        },
        "Profitability Ratios": {
            "Gross Margin": "Percentage of revenue exceeding the cost of goods sold (COGS).",
            "Operating Margin": "Percentage of revenue remaining after paying for variable costs of production (wages, raw materials, etc.).",
            "Net Margin": "Percentage of revenue remaining after all expenses, including taxes and interest, have been paid.",
            "ROE (Return on Equity)": "Net income divided by shareholders' equity. Measures how effectively management is using a company’s assets to create profits.",
            "ROA (Return on Assets)": "Net income divided by total assets. Indicates how efficient a company's management is at using its assets to generate earnings.",
        },
        "Financial Health": {
            "Debt-to-Equity": "Total liabilities divided by shareholder equity. High values indicate higher financial leverage and risk.",
            "Current Ratio": "Current assets divided by current liabilities. Measures the ability to pay short-term obligations.",
            "Interest Coverage": "EBIT divided by interest expense. Indicates how easily a company can pay interest on its outstanding debt.",
        },
        "Technical Indicators": {
            "SMA (Simple Moving Average)": "Average price over a specific period (e.g., 20, 50, 200 days). Used to identify trends.",
            "RSI (Relative Strength Index)": "Momentum oscillator measuring the speed and change of price movements. >70 is overbought, <30 is oversold.",
            "MACD (Moving Average Convergence Divergence)": "Trend-following momentum indicator showing the relationship between two moving averages of a security’s price.",
            "Bollinger Bands": "Volatility bands placed above and below a moving average. When bands widen, volatility is high; when they contract, volatility is low.",
            "ATR (Average True Range)": "Market volatility indicator derived from the greatest of the 3 values: current high-low, abs(current high-prev close), abs(current low-prev close).",
            "Stochastic Oscillator": "Momentum indicator comparing a closing price to a range of prices over a given time period. Used to identify turning points.",
            "Max Drawdown": "The largest peak-to-trough decline in a stock price over a specified period (typically 1 year). Measures downside risk - a 20% max drawdown means the stock fell 20% from its highest point. Lower values indicate less risk.",
            "Volatility (1Y)": "Annualized standard deviation of daily returns over the past year. Higher volatility means larger price swings. Expressed as a percentage (e.g., 25% means price typically moves within plus or minus 25% annually).",
            "Trend": "The general direction of price movement. Uptrend: higher highs and higher lows. Downtrend: lower highs and lower lows. Sideways: price moving within a horizontal range.",
        },
        "Growth Metrics": {
            "Revenue CAGR": "Compound Annual Growth Rate of revenue over a specific period (usually 3 years).",
            "EPS CAGR": "Compound Annual Growth Rate of earnings per share over a specific period.",
        },
    }

    st.markdown("### Key Financial & Technical Definitions")
    st.markdown("---")

    for category, terms in glossary.items():
        st.subheader(category)
        for term, definition in terms.items():
            st.markdown(f"**{term}**")
            st.markdown(f"{definition}")
            st.markdown("")  # Spacing
        st.markdown("---")


def render_scoring_methodology():
    """Render the scoring methodology page."""
    
    st.markdown("### 📊 Scoring Methodology")
    st.markdown("---")
    
    st.markdown("""
    The investment scores are calculated using **deterministic rules** based on financial data.
    No AI/LLM is involved in the score calculations - only in report generation.
    """)
    
    # The 5 Factor Scores
    st.subheader("The 5 Factor Scores")
    
    factor_scores_data = {
        "Factor": ["Valuation", "Growth", "Profitability", "Financial Health", "Technical"],
        "Weight": ["25%", "25%", "15%", "15%", "20%"],
        "Key Metrics": [
            "P/E ratio, PEG ratio, sector comparison",
            "Revenue CAGR (3Y), EPS CAGR (3Y)",
            "ROE, Operating Margin",
            "Debt/Equity, Interest Coverage",
            "RSI, Trend label, Max Drawdown"
        ],
        "Data Source": [
            "Yahoo Finance (.info)",
            "Financial Statements",
            "Financial Statements",
            "Balance Sheet",
            "Price History"
        ]
    }
    
    st.table(factor_scores_data)
    
    st.markdown("---")
    
    # Detailed Score Breakdowns
    st.subheader("Score Calculation Details")
    
    with st.expander("📈 Valuation Score (25% weight)", expanded=False):
        st.markdown("""
**Base Score from P/E Ratio:**
| P/E Range | Score | Interpretation |
|-----------|-------|----------------|
| < 10 | 85 | Very cheap |
| 10-15 | 75 | Cheap |
| 15-20 | 65 | Reasonable |
| 20-25 | 55 | Slightly expensive |
| 25-35 | 40 | Expensive |
| > 35 | 25 | Very expensive |

**Sector Comparison Adjustment:**
- P/E < 70% of sector median: +15 points
- P/E 70-90% of sector: +8 points
- P/E 110-130% of sector: -8 points
- P/E > 130% of sector: -15 points

**PEG Ratio Adjustment:**
- PEG < 1.0: +10 points (good value for growth)
- PEG 1.0-1.5: +5 points
- PEG 2.0-2.5: -5 points
- PEG > 2.5: -10 points
        """)
    
    with st.expander("📊 Growth Score (25% weight)", expanded=False):
        st.markdown("""
**Revenue CAGR (3-Year):**
| Revenue CAGR | Score |
|--------------|-------|
| ≥ 20% | 95 |
| 12-20% | 80 |
| 6-12% | 65 |
| 0-6% | 45 |
| < 0% (declining) | 25 |

**EPS CAGR (3-Year):**
| EPS CAGR | Score |
|----------|-------|
| ≥ 25% | 95 |
| 15-25% | 80 |
| 8-15% | 65 |
| 0-8% | 45 |
| < 0% | 20 |

**Growth Score** = (Revenue Score + EPS Score) / 2

**Acceleration Bonus:** +5 if YoY growth > 3Y CAGR, -5 if decelerating
        """)
    
    with st.expander("💰 Profitability Score (15% weight)", expanded=False):
        st.markdown("""
**ROE (Return on Equity):**
| ROE | Score |
|-----|-------|
| ≥ 25% | 95 |
| 15-25% | 80 |
| 10-15% | 65 |
| 5-10% | 45 |
| < 5% | 25 |

**Operating Margin:**
| Operating Margin | Score |
|------------------|-------|
| ≥ 30% | 95 |
| 20-30% | 80 |
| 10-20% | 60 |
| 0-10% | 40 |
| < 0% (loss) | 20 |

**Profitability Score** = (ROE Score + Margin Score) / 2
        """)
    
    with st.expander("🏦 Financial Health Score (15% weight)", expanded=False):
        st.markdown("""
**Debt/Equity Ratio** (lower is better):
| Debt/Equity | Score |
|-------------|-------|
| ≤ 0.3 | 95 |
| 0.3-0.6 | 80 |
| 0.6-1.0 | 60 |
| 1.0-2.0 | 40 |
| > 2.0 | 20 |

**Interest Coverage** (higher is better):
| Interest Coverage | Score |
|-------------------|-------|
| ≥ 10x | 95 |
| 6-10x | 80 |
| 3-6x | 60 |
| 1-3x | 35 |
| < 1x | 15 |

**Health Score** = (D/E Score + Interest Coverage Score) / 2
        """)
    
    with st.expander("📉 Technical Score (20% weight)", expanded=False):
        st.markdown("""
**RSI (14-day):**
| RSI Range | Score | Interpretation |
|-----------|-------|----------------|
| 40-60 | 80 | Neutral - healthy |
| 60-70 | 70 | Slightly overbought |
| 30-40 | 55 | Slightly oversold |
| > 70 | 40 | Overbought - caution |
| < 30 | 35 | Oversold - distressed |

**Trend:**
| Trend | Score |
|-------|-------|
| Uptrend | 85 |
| Sideways | 55 |
| Downtrend | 25 |

**Max Drawdown (1Y):**
| Drawdown | Score |
|----------|-------|
| < 10% | 90 |
| 10-20% | 70 |
| 20-35% | 45 |
| > 35% | 25 |

**Technical Score** = (RSI Score + Trend Score + Drawdown Score) / 3
        """)
    
    st.markdown("---")
    
    # Final Score Formula
    st.subheader("Final Score Formula")
    
    st.code("""
Final Score = (0.25 × Valuation) + (0.25 × Growth) + (0.15 × Profitability) 
            + (0.15 × Financial Health) + (0.20 × Technical) 
            + Sentiment Adjustment (-5 to +5)
    """, language="text")
    
    st.markdown("---")
    
    # Rating Thresholds
    st.subheader("Rating Thresholds")
    
    rating_data = {
        "Score Range": ["80-100", "65-79", "45-64", "30-44", "0-29"],
        "Rating": ["STRONG BUY", "BUY", "HOLD", "REDUCE", "SELL"],
        "Interpretation": [
            "Excellent investment opportunity across most factors",
            "Good investment with favorable risk/reward",
            "Fairly valued, wait for better entry or hold existing position",
            "Consider reducing position, unfavorable risk/reward",
            "Significant concerns, consider exiting position"
        ]
    }
    
    st.table(rating_data)
    
    st.markdown("---")
    
    # Example Calculation
    st.subheader("Example Calculation")
    
    with st.expander("📝 See Example Score Calculation", expanded=False):
        st.markdown("""
**Example Stock Data:**
- P/E = 18, Sector P/E = 25, PEG = 1.2
- Revenue CAGR = 15%, EPS CAGR = 20%
- ROE = 22%, Operating Margin = 25%
- Debt/Equity = 0.5, Interest Coverage = 8x
- RSI = 55, Trend = Uptrend, Max Drawdown = 15%

**Score Calculations:**
```
Valuation:    65 (base for PE 18) + 8 (below sector) + 5 (PEG 1.2) = 78
Growth:       (80 + 80) / 2 = 80
Profitability: (80 + 80) / 2 = 80
Health:       (80 + 80) / 2 = 80
Technical:    (80 + 85 + 70) / 3 = 78.3

Final = 0.25×78 + 0.25×80 + 0.15×80 + 0.15×80 + 0.20×78.3
      = 19.5 + 20 + 12 + 12 + 15.66
      = 79.16
```

**Result:** Score of 79.16 → **Rating: "BUY"**
        """)


# --------------------------------
# Page config
# --------------------------------
st.set_page_config(
    page_title="FinResearch AI - Pure CrewAI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Validate environment on startup
validate_environment()

# --------------------------------
# Initialize Session State
# --------------------------------
# This ensures results persist even if sidebar widgets cause a re-run
if "research_result" not in st.session_state:
    st.session_state.research_result = None

# --------------------------------
# Sidebar Inputs
# --------------------------------
st.sidebar.title("FinResearch AI")
st.sidebar.caption("Pure CrewAI with Auto Sector Detection")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="AAPL",
    help="Enter a valid stock ticker symbol (e.g., AAPL, GOOG, MSFT)",
)

force_refresh = st.sidebar.checkbox(
    "Force refresh (bypass cache)",
    value=False,
    help="Re-run all agents even if cached data exists",
)

st.sidebar.markdown("---")

# Advanced options
with st.sidebar.expander("Advanced Options"):
    show_execution_log = st.checkbox(
        "Show execution timeline",
        value=True,
        help="Display execution time and sector detection details",
    )

    show_debug_tabs = st.checkbox(
        "Show raw agent outputs",
        value=True,
        help="Show 'Raw Analysis' and 'Debug Output' tabs",
    )

    # --- CLEAR CACHE BUTTON ---
    # Manual tool to delete bad records from ChromaDB
    if st.button("Clear Cache for Ticker"):
        from tools.chroma_client_tools import get_chroma_client

        try:
            client = get_chroma_client()
            # Deletes ALL records associated with this ticker
            client.delete(where={"ticker": ticker})
            st.sidebar.success(
                f"Cache cleared for {ticker}! Re-run to fetch fresh data."
            )
            # Clear session state if it matches the deleted ticker
            if (
                st.session_state.research_result
                and st.session_state.research_result.get("ticker") == ticker
            ):
                st.session_state.research_result = None
        except Exception as e:
            st.sidebar.error(f"Error clearing cache: {e}")

run_btn = st.sidebar.button("Run Research", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Architecture:**
    - Analyst Agent (quantitative)
    - Researcher Agent (news/sentiment)
    - Reporter Agent (synthesis)
    
    Uses CrewAI's **sequential process** with **async tasks** for parallel execution.
    """
)

# --------------------------------
# Main page
# --------------------------------
st.title("FinResearch AI - Equity Research Assistant")
st.markdown(
    """
    AI-driven multi-agent equity research using **pure CrewAI** with automatic sector detection.
    The system coordinates parallel execution of Analyst and Researcher agents,
    then synthesizes their outputs into a professional investment report.
    """
)

# Disclaimer
st.warning(
    "**Disclaimer: This tool is for informational and demonstration purposes only. "
    "Not financial advice. Consult a licensed financial advisor before making investment decisions.**",
    icon="⚠️",
)

# --------------------------------
# Run Research Logic
# --------------------------------
if run_btn:
    if not ticker.strip():
        st.error("Please enter a valid stock ticker.")
    else:
        ticker = ticker.upper().strip()

        # Progress container
        progress_container = st.container()

        with progress_container:
            st.info(f"Analyzing **{ticker}** (sector will be auto-detected)...")
            progress_bar = st.progress(0, text="Initializing agents...")

            # Run the research
            try:
                result = asyncio.run(
                    run_full_research(
                        ticker=ticker,
                        force_refresh=force_refresh,
                    )
                )

                # Check for errors in result
                if "error" in result:
                    progress_bar.progress(100, text="Analysis failed")
                    st.error(f"**Error:** {result['error']}")
                    if "execution_info" in result:
                        st.json(result["execution_info"])
                    st.stop()
                else:
                    progress_bar.progress(100, text="Analysis complete!")
                    # SAVE RESULT TO SESSION STATE
                    st.session_state.research_result = result

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.exception(e)
                st.stop()

        # Clear progress indicators
        progress_container.empty()

# --------------------------------
# Render Results (from Session State)
# --------------------------------
# Only render if we have a valid result in session state
if st.session_state.research_result:
    result = st.session_state.research_result

    # Get sector info
    detected_sector = result.get("sector", "Unknown")
    raw_sector = result.get("raw_sector", detected_sector)

    # --------------------------------
    # Execution Status Block
    # --------------------------------
    if show_execution_log and "execution_info" in result:
        st.subheader("Execution Status")
        exec_info = result["execution_info"]

        col_time, col_sector = st.columns(2)

        col_time.metric(
            "Execution Time",
            f"{exec_info.get('execution_time_seconds', 'N/A')}s",
        )

        col_sector.info(f"Detected Sector: {detected_sector}")

    # --------------------------------
    # Investment Snapshot
    # --------------------------------
    st.subheader("Investment Snapshot")

    col1, col2, col3, col4 = st.columns(4)

    rating = result.get("rating_5_tier", "N/A")
    final_score = result.get("final_score")

    col1.metric("Rating", rating)
    col2.metric(
        "Final Score",
        f"{final_score} / 100" if final_score else "N/A",
    )
    col3.metric("Confidence", result.get("confidence", "N/A"))
    col4.metric("Risk Level", result.get("risk_level", "N/A"))

    # --------------------------------
    # Report Header Info
    # --------------------------------
    st.markdown("---")

    h_col1, h_col2, h_col3 = st.columns(3)

    with h_col1:
        st.markdown(f"**Ticker:** {ticker}")

    with h_col2:
        st.markdown(f"**Date:** {result.get('as_of_date', 'N/A')}")

    with h_col3:
        st.markdown(f"**Sector:** {detected_sector}")
        if raw_sector != detected_sector:
            st.caption(f"(Raw: {raw_sector})")

    st.markdown("---")

    # --------------------------------
    # Target Price (if available)
    # --------------------------------
    target_price = result.get("target_price", {})
    if target_price and target_price.get("status") == "Computed":
        st.subheader("Target Price Range")
        st.write(
            f"**Range:** {target_price.get('range_low')} - {target_price.get('range_high')}"
        )
        st.caption(f"Method: {target_price.get('method', 'N/A')}")

    # --------------------------------
    # Tabs for detailed content
    # --------------------------------
    # Define available tabs
    tab_names = ["Full Report", "Charts", "News & Sentiment"]

    # Add Debug tabs if selected
    if show_debug_tabs:
        tab_names.extend(["Raw Analysis", "Debug Output"])

    # Always add Scoring and Definitions tabs at the end
    tab_names.append("Scoring")
    tab_names.append("Definitions")

    tabs = st.tabs(tab_names)

    # Tab 1: Full Report
    with tabs[0]:
        report_md = result.get("report_markdown", "")
        if report_md:
            st.markdown(report_md)
            st.download_button(
                label="Download Report (Markdown)",
                data=report_md,
                file_name=f"{ticker}_equity_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
            )
        else:
            st.warning("Report markdown not available.")

    # Tab 2: Charts
    with tabs[1]:
        st.subheader("📊 Visual Analysis")
        
        # Get data for charts
        scores = result.get("scores", {})
        analyst_data = result.get("analyst", {})
        
        # If scores not at top level, try to get from analyst data
        if not scores and analyst_data:
            scores = analyst_data.get("scores", {})
        
        # Determine color theme based on Streamlit theme (default to light)
        tv_theme = "light"  # Change to "dark" if you add dark mode support
        
        # Get full symbol with exchange prefix for TradingView
        tv_symbol = get_exchange_prefix(ticker, detect_exchange(ticker))
        
        # =================================================================
        # TRADINGVIEW SECTION: Live Charts
        # =================================================================
        st.markdown("### 📈 Live Market Data")
        
        # Symbol Overview - Shows live price with mini chart
        try:
            render_symbol_overview(ticker, height=180, color_theme=tv_theme)
        except Exception as e:
            st.warning(f"Could not load TradingView symbol overview: {e}")
        
        st.markdown("---")
        
        # Two-column layout: Advanced Chart + Technical Analysis
        tv_col1, tv_col2 = st.columns([2, 1])
        
        with tv_col1:
            st.markdown("### Interactive Price Chart")
            
            # Chart interval selector
            chart_interval = st.selectbox(
                "Select Timeframe",
                options=["1", "5", "15", "60", "D", "W", "M"],
                format_func=lambda x: {
                    "1": "1 Minute", 
                    "5": "5 Minutes", 
                    "15": "15 Minutes",
                    "60": "1 Hour", 
                    "D": "Daily", 
                    "W": "Weekly", 
                    "M": "Monthly"
                }.get(x, x),
                index=4,  # Default to Daily
                key="tv_chart_interval"
            )
            
            try:
                render_advanced_chart(
                    symbol=ticker,
                    height=450,
                    interval=chart_interval,
                    color_theme=tv_theme,
                    studies=[
                        "MASimple@tv-basicstudies",  # Moving Average
                        "RSI@tv-basicstudies",       # RSI indicator
                    ],
                )
            except Exception as e:
                st.warning(f"Could not load TradingView chart: {e}")
                # Fallback message
                st.info("📊 TradingView chart requires internet connection.")
        
        with tv_col2:
            st.markdown("### Technical Analysis")
            st.caption("TradingView's Buy/Sell Gauge")
            try:
                render_technical_analysis(
                    symbol=ticker,
                    height=450,
                    color_theme=tv_theme,
                    interval="1D",
                )
            except Exception as e:
                st.warning(f"Could not load TradingView technical analysis: {e}")
        
        st.markdown("---")
        
        # =================================================================
        # OUR SCORES SECTION: Proprietary Analysis
        # =================================================================
        st.markdown("### 🎯 FinAI Investment Scores")
        st.caption("Our proprietary scoring based on fundamentals, technicals, and sector comparison")
        
        # Final Score Display
        final_score = result.get("final_score") or scores.get("final_score", 50)
        rating = result.get("rating_5_tier") or scores.get("rating", "HOLD")
        confidence = result.get("confidence", "MEDIUM")
        
        if final_score:
            try:
                fig_final = create_final_score_display(final_score, rating, confidence)
                st.plotly_chart(fig_final, width='stretch')
            except Exception as e:
                st.warning(f"Could not render final score chart: {e}")
        
        st.markdown("---")
        
        # Score Gauges Row
        if scores:
            st.markdown("### Factor Score Breakdown")
            try:
                fig_gauges = create_score_gauges_row(scores)
                st.plotly_chart(fig_gauges, width='stretch')
            except Exception as e:
                st.warning(f"Could not render gauge charts: {e}")
            
            st.markdown("---")
            
            # Radar Chart
            st.markdown("### Factor Score Radar")
            try:
                fig_radar = create_score_radar_chart(scores, ticker)
                st.plotly_chart(fig_radar, width='stretch')
            except Exception as e:
                st.warning(f"Could not render radar chart: {e}")
        else:
            st.info("Score data not available for charts.")
        
        st.markdown("---")
        
        # Technical Indicators Chart (our calculations)
        technical_indicators = result.get("technical_indicators", {})
        if not technical_indicators and analyst_data:
            technical_indicators = analyst_data.get("technical_indicators", {})
        
        if technical_indicators:
            st.markdown("### Our Technical Indicators")
            st.caption("Calculated from historical price data")
            try:
                fig_tech = create_technical_chart(technical_indicators, ticker)
                st.plotly_chart(fig_tech, width='stretch')
            except Exception as e:
                st.warning(f"Could not render technical chart: {e}")
        
        st.markdown("---")
        
        # Sector Comparison Chart
        sector = result.get("sector", "Technology")
        sector_benchmarks = result.get("sector_benchmarks", {})
        fundamental_metrics = result.get("fundamental_metrics", {})
        
        if not sector_benchmarks and analyst_data:
            sector_benchmarks = analyst_data.get("sector_benchmarks", {})
        if not fundamental_metrics and analyst_data:
            fundamental_metrics = analyst_data.get("fundamental_metrics", {})
        
        # Merge valuation data for comparison
        stock_metrics = {}
        if fundamental_metrics:
            stock_metrics.update(fundamental_metrics.get("profitability", {}))
            stock_metrics.update(fundamental_metrics.get("valuation", {}))
        
        # Add PE from scores if available
        if scores:
            stock_metrics["pe_ttm"] = scores.get("pe_ttm") or fundamental_metrics.get("pe_ttm")
            stock_metrics["peg_ratio"] = scores.get("peg_ratio") or fundamental_metrics.get("peg_ratio")
        
        if stock_metrics and sector_benchmarks:
            st.markdown(f"### {ticker} vs {sector} Sector")
            try:
                # Get valuation benchmarks
                valuation_benchmarks = sector_benchmarks.get("valuation", {})
                profitability_benchmarks = sector_benchmarks.get("profitability", {})
                combined_benchmarks = {**valuation_benchmarks, **profitability_benchmarks}
                
                fig_sector = create_sector_comparison_chart(
                    stock_metrics, combined_benchmarks, ticker, sector
                )
                if fig_sector:
                    st.plotly_chart(fig_sector, width='stretch')
            except Exception as e:
                st.warning(f"Could not render sector comparison chart: {e}")

    # Tab 3: News & Sentiment
    with tabs[2]:
        st.subheader("Researcher Agent Output")
        researcher_data = result.get("researcher", {})

        if not researcher_data:
            researcher_data = {
                "headline_summary": result.get("headline_summary", []),
                "sentiment": result.get("sentiment", {}),
                "key_risks": result.get("key_risks", []),
                "key_tailwinds": result.get("key_tailwinds", []),
            }

        if researcher_data.get("headline_summary"):
            st.markdown("**Recent Headlines:**")
            for headline in researcher_data.get("headline_summary", []):
                st.markdown(f"- {headline}")

            sentiment = researcher_data.get("sentiment", {})
            if sentiment:
                st.markdown(
                    f"**Sentiment:** {sentiment.get('label', 'N/A')} "
                    f"(score: {sentiment.get('score', 0)})"
                )

            if researcher_data.get("key_risks"):
                st.markdown("**Key Risks:**")
                for risk in researcher_data.get("key_risks", []):
                    st.markdown(f"- {risk}")

            if researcher_data.get("key_tailwinds"):
                st.markdown("**Key Tailwinds:**")
                for tailwind in researcher_data.get("key_tailwinds", []):
                    st.markdown(f"- {tailwind}")
        else:
            st.info("News data not available.")

    # Handle Debug Tabs if enabled
    current_tab_index = 3

    if show_debug_tabs:
        # Tab 4: Raw Analysis
        with tabs[current_tab_index]:
            st.subheader("Analyst Agent Output")
            analyst_data = result.get("analyst", {})
            if not analyst_data:
                analyst_data = {
                    "scores": result.get("scores", {}),
                    "technical_indicators": result.get("technical_indicators", {}),
                    "fundamental_metrics": result.get("fundamental_metrics", {}),
                    "sector_benchmarks": result.get("sector_benchmarks", {}),
                }
            st.json(analyst_data)
        current_tab_index += 1

        # Tab 5: Debug Output
        with tabs[current_tab_index]:
            st.subheader("Full CrewAI Output")
            st.json(result)
            st.download_button(
                label="Download Raw JSON",
                data=json.dumps(result, indent=2, default=str),
                file_name=f"{ticker}_raw_output.json",
                mime="application/json",
            )
        current_tab_index += 1

    # Scoring Tab
    with tabs[current_tab_index]:
        render_scoring_methodology()
    current_tab_index += 1

    # Final Tab: Definitions (Always the last index)
    with tabs[current_tab_index]:
        render_glossary()

else:
    # Landing state (Only show if no result is in session)
    st.info("Enter a ticker and click **Run Research** to begin.")

    with st.expander("Try these sample tickers"):
        st.markdown("**Technology:** AAPL, MSFT, NVDA")
        st.markdown("**Healthcare:** JNJ, PFE, MRK")
        st.markdown("**Financial Services:** JPM, BAC, GS")
        st.markdown("**Consumer Cyclical:** AMZN, TSLA, MCD")
        st.markdown("**Energy:** XOM, CVX, SLB")
