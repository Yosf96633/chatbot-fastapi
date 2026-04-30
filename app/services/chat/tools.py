# app/services/chat/tools.py
import os
import math
import numexpr
import requests
from langchain_core.tools import tool
from tavily import TavilyClient
from langgraph.types import interrupt
# ─── Web Search ────────────────────────────────────────────────────────────────


@tool
def web_search(query: str) -> str:
    """Search the web for current information about any topic."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query=query, max_results=5)
    if not results.get("results"):
        return "No results found."
    return "\n\n".join(
        f"**{r['title']}**\n{r['content']}\nSource: {r['url']}"
        for r in results["results"]
    )


# ─── Weather ───────────────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Input should be a city name like 'London' or 'New York'."""
    api_key = os.getenv("WEATHER_API_KEY")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        return f"Could not fetch weather for '{city}'. Error: {resp.json().get('message', 'Unknown error')}"
    data = resp.json()
    weather = data["weather"][0]["description"].capitalize()
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    wind = data["wind"]["speed"]
    return (
        f"Weather in {data['name']}, {data['sys']['country']}:\n"
        f"  Condition : {weather}\n"
        f"  Temperature: {temp}°C (feels like {feels_like}°C)\n"
        f"  Humidity  : {humidity}%\n"
        f"  Wind speed: {wind} m/s"
    )


# ─── Stock Price ───────────────────────────────────────────────────────────────

@tool
def get_stock_price(symbol: str) -> str:
    """Get the latest stock price for a ticker symbol like AAPL, TSLA, MSFT."""
    api_key = os.getenv("STOCK_API_KEY")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol.upper(),
        "apikey": api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json().get("Global Quote", {})
    if not data or not data.get("05. price"):
        return f"Could not fetch stock data for '{symbol}'. Check the ticker symbol."
    return (
        f"Stock: {data['01. symbol']}\n"
        f"  Price    : ${float(data['05. price']):.2f}\n"
        f"  Change   : {data['09. change']} ({data['10. change percent']})\n"
        f"  High     : ${float(data['03. high']):.2f}\n"
        f"  Low      : ${float(data['04. low']):.2f}\n"
        f"  Volume   : {int(data['06. volume']):,}\n"
        f"  Last updated: {data['07. latest trading day']}"
    )


# ─── Calculator ────────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, exp, pi, e, etc.
    Examples: '2 ** 10', 'sqrt(144)', 'sin(pi/2)', '(3 + 4) * 5'
    """
    try:
        # numexpr handles safe math eval with numpy functions
        result = numexpr.evaluate(expression.replace("^", "**"))
        # result is a numpy scalar — convert to Python type
        result = result.item() if hasattr(result, "item") else result
        return f"{expression} = {result}"
    except Exception:
        # fallback: try with math constants exposed
        try:
            safe_locals = {k: getattr(math, k)
                           for k in dir(math) if not k.startswith("_")}
            safe_locals["pi"] = math.pi
            safe_locals["e"] = math.e
            result = eval(expression, {"__builtins__": {}}, safe_locals)  # noqa: S307
            return f"{expression} = {result}"
        except Exception as exc:
            return f"Could not evaluate '{expression}': {exc}"

# ─── Email sender ────────────────────────────────────────────────────────────────


@tool
def send_email(subject: str, body: str) -> dict[str, str]:
    """
    Sends an email to the user.
    Args:
        subject: The subject line of the email.
        body: The main content/body of the email.
    """
    # ✅ interrupt MUST be outside try/except
    decision = interrupt({
        "question": "Do you want to send this email?",
        "subject": subject,
        "body": body,
    })

    if decision.lower() != "yes":
        return {"status": "cancelled", "message": "Cancelled by user."}

    try:
        # your actual email sending logic here
        return {"status": "success", "message": f"Email '{subject}' sent successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─── Exported list ─────────────────────────────────────────────────────────────

all_tools = [web_search, get_weather, get_stock_price, calculator, send_email]
