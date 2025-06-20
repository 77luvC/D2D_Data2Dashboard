import os
import yaml
from functools import lru_cache

@lru_cache()
def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file and environment variables."""
    data = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
    llm = data.setdefault('llm', {})
    llm['backend'] = os.getenv('LLM_BACKEND', llm.get('backend', 'openai'))
    llm['model'] = os.getenv('LLM_MODEL', llm.get('model', 'gpt-4o'))
    llm['temperature'] = float(os.getenv('LLM_TEMPERATURE', llm.get('temperature', 0)))
    llm.setdefault('openai_api_key', os.getenv('OPENAI_API_KEY'))
    llm.setdefault('google_api_key', os.getenv('GOOGLE_API_KEY'))
    llm.setdefault('hf_token', os.getenv('HF_TOKEN'))
    llm.setdefault('ollama_host', os.getenv('OLLAMA_HOST'))
    think_env = os.getenv('OLLAMA_THINK')
    if think_env is not None:
        llm['think'] = think_env.lower() in ('1', 'true', 'yes')
    else:
        llm.setdefault('think', False)
    llm.setdefault('lmstudio_url', os.getenv('LMSTUDIO_API_URL', 'http://localhost:1234/v1'))
    return data

def get_default_llm(**kwargs):
    cfg = load_config()
    llm_cfg = cfg.get('llm', {})
    backend = llm_cfg.get('backend', 'openai')
    model = llm_cfg.get('model', 'gpt-4o')
    from agents.llm_factory import get_llm
    options = {
        'openai_api_key': llm_cfg.get('openai_api_key'),
        'google_api_key': llm_cfg.get('google_api_key'),
        'hf_token': llm_cfg.get('hf_token'),
    }
    if backend == 'lmstudio':
        options['base_url'] = llm_cfg.get('lmstudio_url')
    if backend == 'ollama':
        if llm_cfg.get('ollama_host'):
            options['base_url'] = llm_cfg['ollama_host']
        options['think'] = llm_cfg.get('think')
    return get_llm(backend=backend, model_name=model, temperature=llm_cfg.get('temperature', 0), **{**options, **kwargs})
