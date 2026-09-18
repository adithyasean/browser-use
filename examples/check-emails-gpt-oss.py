import asyncio
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from browser_use import Agent
from browser_use.llm import ChatOllama

load_dotenv()

# Initialize the model with Ollama using GPT-OSS
llm = ChatOllama(
    model='gpt-oss:20b',  # Use the GPT-OSS model
    host='http://localhost:11434',
    keep_alive='30m',
    options={
        'temperature': 0.1,  # Lower temperature for more deterministic output
        'top_p': 0.8,
        'num_ctx': 8192,
        'num_thread': 8,
        'repeat_penalty': 1.1,
    }
)

# Define the task for checking Gmail
task = '''
Goal: Check Gmail inbox and provide a summary of important and recent emails.

Steps:
1. Navigate to Gmail and handle login if needed
2. Check inbox for:
   - Unread emails from the last 24 hours
   - Important emails (starred or marked important)
   - Emails from key contacts or with important subjects
3. Summarize findings including:
   - Sender, subject, and brief content summary for important emails
   - Any urgent action items or deadlines
   - Total count of unread and important emails

Success criteria:
- Successfully access Gmail inbox
- Identify and summarize important/recent emails
- Save summary to email_summary.txt
'''

# Initialize the agent with credentials
agent = Agent(
    username = os.getenv('GOOGLE_USERNAME')
    password = os.getenv('GOOGLE_PASSWORD')
    task=task,
    llm=llm,
    # Configure agent behavior
    validate_output=True,
    use_vision=True,
    use_thinking=True,
    include_tool_call_examples=True,
    # Configure timeouts
    llm_timeout=180,
    step_timeout=240,
    # Configure retries and actions
    max_actions_per_step=3,
    max_failures=3,
    retry_delay=3
)

async def main():
    print(f"Using Ollama model: {llm.model} @ {llm.host}")
    
    try:
        # Run the agent with a maximum number of steps
        history = await agent.run(max_steps=15)  # Fewer steps for simpler model
        
        # Check if task was completed successfully
        if history.is_successful():
            print("\n✅ Successfully checked Gmail")
            if result := history.final_result():
                print("\nSummary:", result)
        else:
            print("\n❌ Failed to complete the task")
            
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        
    finally:
        # Clean up
        await agent.close()

if __name__ == '__main__':
    asyncio.run(main())
