# benchmark01-baseline-GPT-4o, with prompting to name domain

############################################################
# 0. Setup & imports                                       #
############################################################
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Make sure your .env has OPENAI_API_KEY=<your-key>
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

############################################################
# 1. Helper: read CSV into a trimmed string                #
############################################################
def csv_to_markdown(file_path: str, max_rows: int = 15) -> str:
    """Load a CSV and return (up to) the first `max_rows` rows as markdown.
    This keeps the prompt small while giving GPT-4o a feel for the schema."""
    df = pd.read_csv(file_path)
    sample = df.head(max_rows)
    return sample.to_markdown(index=False)

############################################################
# 2. Build the chain                                       #
############################################################
# --- 2.1  Prompt template ---------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Analyze this file and identify the domain of the data, and then based on the domain knowledge, analyze the data with insights. Your response must be in JSON format."),
        # The user's question or instruction will arrive at run-time
        ("human", "{user_prompt}\n\nHere's a preview of the dataset:\n{csv_md}")
    ]
)

# --- 2.2  Chat model --------------------------------------
llm = ChatOpenAI(
    model_name="gpt-4o", 
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# --- 2.3  Chain object ------------------------------------
chain = prompt | llm  # (Prompt → ChatOpenAI)

############################################################
# 3. Convenience wrapper                                   #
############################################################
def analyze_csv_with_insights_domain(csv_path: str, user_prompt: str) -> str:
    csv_md = csv_to_markdown(csv_path)
    response = chain.invoke({"user_prompt": user_prompt, "csv_md": csv_md})
    return response.content











