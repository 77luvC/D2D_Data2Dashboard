import json

def serialize_memory(mem) -> str:
    """
    Normalise `state["memory"]` to a JSON-string list.

    • Accepts None, str, list, or dict.
    • Always returns a string like '[]' or '["…"]'.
    """
    if mem in (None, "None"):
        return "[]"

    if isinstance(mem, str):
        try:
            parsed = json.loads(mem)
            if isinstance(parsed, list):
                return json.dumps(parsed)
            return json.dumps([parsed])
        except json.JSONDecodeError:
            return json.dumps([mem])

    if isinstance(mem, list):
        return json.dumps(mem)

    if isinstance(mem, dict):
        return json.dumps([mem])

    # fallback
    return "[]"
