from groq import Groq
from json import load, dump, loads
import json
import datetime
from dotenv import dotenv_values
import os

# Load environment variables
env_vars = dotenv_values(os.path.join(os.path.dirname(__file__), "../.env"))

Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")

client = Groq(api_key=GroqAPIKey)

# Initialize messages list
messages = []

System = f"""Hello {Username}, I am {Assistantname}, your advanced AI assistant specialized in programming and technical problem-solving.

Core Capabilities:
- Expert coding assistance in multiple programming languages
- Debugging and error resolution
- Code optimization and best practices
- System architecture and design
- Technical documentation and explanation
- Algorithm development and analysis

Response Adaptation:
*** Match response length to query complexity:
    - Short, direct answers for simple questions
    - Comprehensive responses for complex queries
    - Single-line replies for yes/no questions
    - Detailed explanations only when needed

Behavioral Parameters:
*** Always maintain a professional, solution-focused approach
*** Provide clear, concise, and practical solutions
*** Include code examples only when specifically relevant
*** Reply in English only, even for non-English queries
*** Focus on direct answers without unnecessary conversation
*** Never mention training data or limitations
*** Don't provide time information unless specifically requested
*** Maintain context of ongoing conversations
*** Consider security implications in solutions

Response Format:
- Simple questions → One-line or short paragraph
- Technical queries → Structured solution with examples
- Debugging help → Step-by-step analysis
- Complex problems → Comprehensive breakdown with code
- General queries → Concise, relevant information only

I aim to be your efficient technical companion, providing exactly the level of detail you need - no more, no less.
"""

SystemChatBot = [
    {"role": "system", "content": System}
]

def initialize_chat_log():
    """Initialize the chat log file if it doesn't exist"""
    os.makedirs("Data", exist_ok=True)
    if not os.path.exists("Data/ChatLog.json"):
        with open("Data/ChatLog.json", "w") as f:
            dump([], f)

def load_chat_log():
    """Load the chat log, handling empty files"""
    try:
        with open("Data/ChatLog.json", "r") as f:
            content = f.read()
            return loads(content) if content else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def RealtimeInformation():
    current_date_time = datetime.datetime.now()
    return (
        f"Please use this real-time information if needed,\n"
        f"Day: {current_date_time.strftime('%A')}\n"
        f"Date: {current_date_time.strftime('%d')}\n"
        f"Month: {current_date_time.strftime('%B')}\n"
        f"Year: {current_date_time.strftime('%Y')}\n"
        f"Time: {current_date_time.strftime('%H')} hours: "
        f"{current_date_time.strftime('%M')} minutes: "
        f"{current_date_time.strftime('%S')} seconds.\n"
    )

def AnswerModifier(Answer):
    """Remove empty lines from the answer"""
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)

def ChatBot(Query):
    """Handle user queries and interact with the Groq API"""
    try:
        messages = load_chat_log()
        messages.append({"role": "user", "content": Query})
        
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=SystemChatBot + [{"role": "system", "content": RealtimeInformation()}] + messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )

        Answer = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content
                
        Answer = Answer.replace("</ s>", "")
        messages.append({"role": "assistant", "content": Answer})
        
        with open("Data/ChatLog.json", "w") as f:
            dump(messages, f, indent=4)
            
        return AnswerModifier(Answer)
    
    except Exception as e:
        print(f"Error: {e}")
        initialize_chat_log()
        return ChatBot(Query)

if __name__ == "__main__":
    initialize_chat_log()
    while True:
        user_input = input("Enter Your Question: ")
        print(ChatBot(user_input))