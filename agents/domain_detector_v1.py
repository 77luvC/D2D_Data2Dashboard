############################################################
# 0. Imports & GPT‑4 model                                 #
############################################################
import json
import pandas as pd
import numpy as np
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, START, END

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

llm = ChatOpenAI(
    model_name="gpt-4o", 
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

############################################################
# 1. LLM‑assisted DataProfiler                             #
############################################################
PROFILE_PROMPT = PromptTemplate(
    input_variables=["raw_preview"],
    template=(
        "You are a data–profiling assistant. Carefully inspect the raw preview of a CSV table and think step‑by‑step to extract *structural facts* that an LLM can later use.\n"
        "--- RAW PREVIEW START ---\n"
        "{raw_preview}\n"
        "--- RAW PREVIEW END ---\n\n"
        "**Think step by step**:\n"
        "1. Count how many rows and columns.\n"
        "2. List every column with an *inferred type* (numeric / categorical / datetime / text).\n"
        "3. For each column give: • three example values  • min / max (if numeric) • detected unit symbol (currency, %, kWh, etc.).\n"
        "4. Detect *functional dependencies* – e.g. if Revenue − Cost ≈ Profit.\n"
        "5. Detect *hierarchical or time‑series structure*: quarter columns, year‑month fields, category → sub‑category columns.\n"
        "6. Identify candidate *primary keys* (columns where all values are unique) and possible *foreign‑key* pairs.\n"
        "7. Note any columns/rows that look like totals or subtotals.\n\n"
        "After reasoning, output a JSON object with keys:\n"
        "  rows, cols, columns (list with {{name, type, examples, unit, min, max}}),\n"
        "  formulas (list of strings),\n"
        "  hierarchy (free text),\n"
        "  time_series (true/false),\n"
        "  candidate_pk (list of column names),\n"
        "  possible_fk (list of tuple strings),\n"
        "  subtotal_cols (list), subtotal_rows (true/false)\n"
        "Use double quotes for all JSON keys. Do **not** wrap the JSON in markdown."
    )
)
profile_chain = PROFILE_PROMPT | llm

def _raw_stats(df: pd.DataFrame, n: int = 5) -> Dict[str, Any]:
    meta = {"n_rows": len(df), "n_cols": df.shape[1], "columns": {}}
    for c in df.columns:
        s = df[c]
        meta["columns"][c] = {
            "dtype": str(s.dtype),
            "unique_ratio": round(s.nunique() / len(s), 4),
            "sample": s.head(n).astype(str).tolist(),
        }
        if pd.api.types.is_numeric_dtype(s):
            meta["columns"][c].update({
                "min": s.min(), "max": s.max(), "mean": round(s.mean(), 4), "std": round(s.std(), 4)
            })
    return meta

def build_profile(csv_path: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    raw = _raw_stats(df)
    llm_response = profile_chain.invoke({"raw_preview": json.dumps(raw, cls=NumpyEncoder)})
    # Extract content from AIMessage
    if hasattr(llm_response, 'content'):
        llm_enriched = llm_response.content
    else:
        llm_enriched = str(llm_response)
    return {"raw": raw, **json.loads(llm_enriched)}

profile = build_profile("https://drive.google.com/uc?export=download&id=1JhsgpIulCv8Q9NPTZGhrz5-y_RUufMoO")  # ← replace with real path

############################################################
# 2. Prompt templates                                      #
############################################################
DOMAIN_PROMPT = PromptTemplate(
    input_variables=["profile", "memory"],
    template=(
        "Dataset profile (JSON):\n{profile}\n"
        "{memory}\n"
        "Determine the most precise domain / industry label (Wikipedia terminology).\n"
        "Return JSON response in this format: {{ 'domain':<string>, 'definition':<one sentence>, 'wiki_url':<url|''> }}"
    )
)
domain_chain = DOMAIN_PROMPT | llm

CONCEPT_PROMPT = PromptTemplate(
    input_variables=["profile", "domain_info", "memory"],
    template=(
        "Profile: {profile}\nDomain: {domain_info}\n"
        "{memory}\n"
        "List 4‑6 concept names (English) that should be analysed for this domain.\n"
        "Return the list in JSON format."
    )
)
concept_chain = CONCEPT_PROMPT | llm

ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["profile", "domain_info", "concepts", "memory"],
    template=(
        "Profile: {profile}\nDomain: {domain_info}\nCore concepts: {concepts}\n"
        "{memory}\n"
        "Produce a JSON response exactly in this shape:\n"
        "{{ 'domain': <string>, 'core_concepts': [...], 'analysis': {{ 'descriptive':<paragraph>, 'predictive':<paragraph>, 'domain_related':<paragraph> }} }}\n"
        "Ensure strong **insightfulness** and **novelty**—surface hidden patterns, non‑obvious relationships, or actionable hypotheses."
    )
)
analysis_chain = ANALYSIS_PROMPT | llm

# Evaluator prompt – separate criteria groups
EVAL_PROMPT = PromptTemplate(
    input_variables=["domain_info", "concepts", "analysis", "profile"],
    template=(
        "You are an evaluation agent.\n\n"
        "**Part A – Domain & Concepts**\n"
        "• correctness : is the domain label factually accurate for this dataset?\n"
        "• relevance   : do the concepts correspond to real columns / metrics present?\n"
        "• coverage    : do the concepts cover the major elements of the table?\n\n"
        "**Part B – Analysis JSON**\n"
        "• insightfulness : does the analysis provide meaningful, actionable understanding (combines relevance, uniqueness, actionability)?\n"
        "• novelty        : does it reveal non‑obvious or deeper patterns beyond simple column descriptions?\n\n"
        "Return a JSON response exactly in this format:\n"
        "{{ 'reason':<brief text>,\n  'scores': {{ 'correctness':#, 'relevance':#, 'coverage':#, 'insightfulness':#, 'novelty':# }},\n  'domain_ok': <bool correctness==4>,\n  'concepts_ok': <bool relevance>=3 and coverage>=3> }}"
    )
)
eval_chain = EVAL_PROMPT | llm

REFLECT_PROMPT = PromptTemplate(
    input_variables=["evaluation", "memory"],
    template=(
        "Evaluation JSON: {evaluation}\n"
        "{memory}\n"
        "For every dimension with score ≤3, write ONE bullet‑point self‑critique:\n"
        "  – state what was missing or wrong, and how to improve (e.g. 'Consider column YYYY …').\n"
        "Return ≤5 bullets only, in a valid JSON list format."
    )
)
reflect_chain = REFLECT_PROMPT | llm

############################################################
# 3. Graph node functions                                  #
############################################################

# Helper function to extract content from LLM response
def _extract_content(response):
    if hasattr(response, 'content'):
        return response.content
    return str(response)

def domain_node(state):
    if state.get("domain_fixed"):
        return {"profile": state["profile"]}
    memory_text = f"Reflection: {state.get('memory')}" if state.get("memory") and state.get("memory") != "None" else ""
    
    response = domain_chain.invoke({"profile": json.dumps(state["profile"], cls=NumpyEncoder), "memory": memory_text})
    
    out = _extract_content(response)
    
    # Handle possible JSON parsing errors
    try:
        domain_info = json.loads(out)
    except json.JSONDecodeError:
        # Create a default structure if parsing fails
        domain_info = {
            "domain": "Unknown",
            "definition": "Unable to determine domain from the data",
            "wiki_url": ""
        }
    
    return {"domain_info": domain_info, "profile": state["profile"]}

def concept_node(state):
    memory_text = f"Reflection: {state.get('memory')}" if state.get("memory") and state.get("memory") != "None" else ""
    
    response = concept_chain.invoke({
        "profile": json.dumps(state["profile"], cls=NumpyEncoder),
        "domain_info": json.dumps(state["domain_info"]),
        "memory": memory_text
    })
    
    out = _extract_content(response)
    return {
        "concepts": out.strip(),
        "profile": state["profile"],
        "domain_info": state["domain_info"]
    }

def analysis_node(state):
    memory_text = f"Please fix / include: {state.get('memory')}" if state.get("memory") and state.get("memory") != "None" else ""
    
    response = analysis_chain.invoke({
        "profile": json.dumps(state["profile"], cls=NumpyEncoder),
        "domain_info": json.dumps(state["domain_info"]),
        "concepts": state["concepts"],
        "memory": memory_text
    })
    
    out = _extract_content(response)
    return {
        "analysis": out,
        "profile": state["profile"],
        "domain_info": state["domain_info"],
        "concepts": state["concepts"]
    }

def eval_node(state):
    response = eval_chain.invoke({
        "domain_info": json.dumps(state["domain_info"]),
        "concepts": state["concepts"],
        "analysis": state["analysis"],
        "profile": json.dumps(state["profile"], cls=NumpyEncoder)
    })
    
    out = _extract_content(response)
    
    # Handle possible JSON parsing errors
    try:
        ev = json.loads(out)
    except json.JSONDecodeError:
        # Create a default evaluation if parsing fails
        ev = {
            "reason": "Unable to parse evaluation response",
            "scores": {
                "correctness": 2,
                "relevance": 2, 
                "coverage": 2,
                "insightfulness": 2,
                "novelty": 2
            },
            "domain_ok": False,
            "concepts_ok": False
        }
    
    return {
        "evaluation": ev["reason"],
        "scores": ev["scores"],
        "domain_ok": ev["domain_ok"],
        "concepts_ok": ev["concepts_ok"],
        "profile": state["profile"],
        "domain_info": state["domain_info"],
        "concepts": state["concepts"],
        "analysis": state["analysis"]
    }

def reflect_node(state):
    # Build evaluation payload for reflector
    eval_payload = {
        "reason": state["evaluation"],
        "scores": state["scores"]
    }
    memory_text = f"Previous reflection: {state.get('memory')}" if state.get("memory") and state.get("memory") != "None" else ""
    
    response = reflect_chain.invoke({
        "evaluation": json.dumps(eval_payload),
        "memory": memory_text
    })
    
    out = _extract_content(response)
    
    # Update the iteration count
    iteration = state.get("iteration", 0) + 1
    
    return {
        "memory": out.strip(),
        "domain_fixed": state["domain_ok"],
        "profile": state["profile"],
        "domain_info": state["domain_info"],
        "concepts": state["concepts"],
        "analysis": state["analysis"],
        "evaluation": state["evaluation"],
        "scores": state["scores"],
        "iteration": iteration
    }

############################################################
# 4. Build state graph                                     #
############################################################

def success(scores: Dict[str, int]) -> bool:
    return all(v >= 4 for v in scores.values())

MAX_ITERS = 3
builder = StateGraph(dict)
builder.add_node("domain", domain_node)
builder.add_node("concept", concept_node)
builder.add_node("analysis", analysis_node)
builder.add_node("eval", eval_node)
builder.add_node("reflect", reflect_node)

builder.add_edge(START, "domain")
builder.add_edge("domain", "concept")
builder.add_edge("concept", "analysis")
builder.add_edge("analysis", "eval")

# Conditional branching after eval
def decide_next(state):
    scores = state.get("scores", {})
    iteration = state.get("iteration", 0)
    
    # Force stop after MAX_ITERS iterations
    if iteration >= MAX_ITERS:
        print(f"Reached max iterations ({MAX_ITERS}), stopping.")
        return END
    
    # Stop if all scores are good enough
    if all(v >= 4 for v in scores.values()):
        print("All scores are excellent, stopping.")
        return END
    
    # Otherwise continue with reflect
    return "reflect"

builder.add_conditional_edges("eval", decide_next)
builder.add_edge("reflect", "concept")  # skip domain if domain_fixed true inside concept logic

graph = builder.compile()

############################################################
# 5. Run the loop                                          #
############################################################
if __name__ == "__main__":
            initial_state = {
                "profile": profile,
                "memory": "None",
                "iteration": 0,
                "domain_fixed": False
            }
            result = graph.invoke(initial_state)
            print("Scores:", result["scores"])
            print("Analysis JSON:\n", result["analysis"])