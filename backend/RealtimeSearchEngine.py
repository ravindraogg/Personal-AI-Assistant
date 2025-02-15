from groq import Groq
from json import load, dump
import json
import datetime
import os
from googlesearch import search
from dotenv import dotenv_values
import requests.adapters
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

env_vars = dotenv_values(os.path.join(os.path.dirname(__file__), "../.env"))
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")

client = Groq(api_key=GroqAPIKey)

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)

# System prompt
System = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which has real-time up-to-date information from the internet. 
*** Provide Answers In a Professional Way, make sure to add full stops, commas, question marks, and use proper grammar.*** 
*** Just answer the question from the provided data in a professional way. ***"""

# Initialize chat history
def init_chat_history():
    try:
        with open(os.path.join("Data", "ChatLog.json"), "r") as f:
            return load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(os.path.join("Data", "ChatLog.json"), "w") as f:
            dump([], f)
        return []

def GoogleSearch(query, max_retries=3, timeout=10):
    """Enhanced Google search with retry mechanism and longer timeout"""
    for attempt in range(max_retries):
        try:
            results = list(search(query, advanced=True, num_results=5, timeout=timeout))
            Answer = f"The search results for '{query}' are:\n[start]\n"
            
            for i in results:
                Answer += f"Title: {i.title}\nDescription: {i.description}\n\n"
            
            Answer += "[end]"
            return Answer
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt == max_retries - 1:
                return f"Search failed after {max_retries} attempts. Error: {str(e)}"
            continue

def AnswerModifier(Answer):
    """Clean up response formatting"""
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)

def get_current_time_info():
    """Get formatted current time information"""
    current_date_time = datetime.datetime.now()
    return {
        'day': current_date_time.strftime("%A"),
        'date': current_date_time.strftime("%d"),
        'month': current_date_time.strftime("%B"),
        'year': current_date_time.strftime("%Y"),
        'time': current_date_time.strftime("%H:%M:%S")
    }

def RealtimeSearchEngine(prompt):
    """Main search engine function with improved error handling"""
    try:
        messages = init_chat_history()
        messages.append({"role": "user", "content": prompt})
        
        # Get search results
        search_results = GoogleSearch(prompt)
        
        # Prepare system messages
        SystemChatBot = [
            {"role": "system", "content": System},
            {"role": "system", "content": search_results},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello, how can I help you?"}
        ]
        
        # Create chat completion
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=SystemChatBot + messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )
        
        # Process response
        Answer = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content
        
        # Update chat history
        messages.append({"role": "assistant", "content": Answer})
        with open(os.path.join("Data", "ChatLog.json"), "w") as f:
            dump(messages, f, indent=4)
        
        return AnswerModifier(Answer)
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    while True:
        try:
            prompt = input("Enter your query: ")
            print(RealtimeSearchEngine(prompt))
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")