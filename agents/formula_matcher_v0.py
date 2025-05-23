import re
import json
import asyncio
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain

# ────────────────────────────────────────────────────────────────
# 0.  Helper — shared async run for notebooks that may already
#     have a running event‑loop (e.g. Jupyter, VSCode)
# ----------------------------------------------------------------
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    # ok if the package is missing in pure‑script execution
    pass


def run_async(coro):
    """Safe `await` helper that works in both scripts & notebooks."""
    try:
        return asyncio.run(coro)
    except RuntimeError as err:
        if "already running" in str(err):
            return asyncio.get_event_loop().run_until_complete(coro)
        raise

# ────────────────────────────────────────────────────────────────
# 1.  Initialise the LLM 
# ----------------------------------------------------------------
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0

llm_matcher = ChatOpenAI(
    model_name      = MODEL_NAME,
    temperature     = TEMPERATURE,
    max_tokens      = 256,
    openai_api_key  = None,      # `None` → pull from env var automatically
    # the API itself can insist on JSON if you have gpt‑4o/3.5‑turbo‑0125+:
    model_kwargs   = {"response_format": {"type": "json_object"}}
)

# ────────────────────────────────────────────────────────────────
# 2.  Wrap the two external tools as tiny LLM chains
# ----------------------------------------------------------------
FUZZY_PROMPT = PromptTemplate(
    input_variables=["var", "columns"],
    template=(
        "You are a helper that aligns a short **variable** name to **one** of the "
        "candidate *column* names.\n\n"
        "Variable: {var}\n"
        "Candidates: {columns}\n\n"
        "Which candidate is the **best semantic match**?  Reply **JSON only**:\n"
        '{{"match":"<column_name>", "score": <0.0-1.0>}}'
    ),
)

fuzzy_chain = LLMChain(
    llm=llm_matcher,
    prompt=FUZZY_PROMPT,
    output_parser=StrOutputParser(),
)

DERIVE_PROMPT = PromptTemplate(
    input_variables=["var", "columns"],
    template=(
        "You are given a variable `{var}` and these table columns: {columns}.\n\n"
        "Can `{var}` be **computed** from the columns (via arithmetic or ratios)?\n"
        "If yes → output **exactly** the JSON:\n"
        '{{"expr":"<python-or-pandas-expression>"}} \n'
        "If impossible →  {{\"expr\": null}}"
    ),
)

derive_chain = LLMChain(
    llm=llm_matcher,
    prompt=DERIVE_PROMPT,
    output_parser=StrOutputParser(),
)

# ────────────────────────────────────────────────────────────────
# 3. ReAct‑style agent operating *per formula*
# ----------------------------------------------------------------
class FormulaAgent:
    """Single‑formula matcher that records every ReAct step."""

    _VAR_REGEX = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

    def __init__(
        self, formula: str, columns: List[str], *, threshold: float = 0.8
    ) -> None:
        self.formula = formula
        self.columns = columns
        self.threshold = threshold
        self.vars = self._extract_vars(formula)

        # ReAct trace & outputs
        self.log: List[Dict[str, Any]] = []
        self.mapping: Dict[str, str] = {}
        self.status: str = "pending"
        self.failure_reason: Optional[str] = None

    # ------------------------------------------------------------
    @staticmethod
    def _extract_vars(expr: str) -> List[str]:
        # crude: all identifiers that *start* with a letter/underscore
        return list(dict.fromkeys(FormulaAgent._VAR_REGEX.findall(expr)))

    # ------------------------------------------------------------
    async def _fuzzy(self, var: str) -> Dict[str, Any]:
        """Call fuzzy_match tool and parse JSON."""
        raw = await fuzzy_chain.arun(var=var, columns=self.columns)
        return json.loads(raw)

    async def _derive(self, var: str) -> Dict[str, Any]:
        raw = await derive_chain.arun(var=var, columns=self.columns)
        return json.loads(raw)

    # ------------------------------------------------------------
    async def run(self) -> Dict[str, Any]:
        for v in self.vars:
            # THOUGHT
            self.log.append({"step": "thought", "text": f"Match variable '{v}'"})

            # ACTION 1: fuzzy_match
            res = await self._fuzzy(v)
            self.log.append({"step": "action", "tool": "fuzzy_match", "input": v, "output": res})

            if res.get("score", 0) >= self.threshold:
                self.mapping[v] = res["match"]
                continue

            # THOUGHT again
            self.log.append({"step": "thought", "text": f"Try derive '{v}'"})

            # ACTION 2: can_be_derived
            der = await self._derive(v)
            self.log.append({"step": "action", "tool": "can_be_derived", "input": v, "output": der})

            if der.get("expr"):
                self.mapping[v] = der["expr"]
                continue

            # FAILURE — give up for this formula
            self.status = "fail"
            self.failure_reason = "derive_fail"
            break
        else:
            # completed loop without break ⇒ success!
            self.status = "saved"

        return {
            "formula": self.formula, #useless comment
            "mapping": self.mapping,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "log": self.log,
        }

# ────────────────────────────────────────────────────────────────
# 4.  Async batch helper
# ----------------------------------------------------------------
async def batch_match(formulas: List[str], columns: List[str]) -> List[Dict[str, Any]]:
    agents = [FormulaAgent(f, columns) for f in formulas]
    return await asyncio.gather(*[a.run() for a in agents])
