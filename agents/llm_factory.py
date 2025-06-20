import os
from langchain_openai import ChatOpenAI
# You'll need to install appropriate libraries for other backends
# For example:
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from .ollama_thinking import ChatOllamaThinking
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(backend: str, model_name: str, **kwargs):
    """
    Returns a Langchain ChatModel instance based on the specified backend.

    Args:
        backend: The backend to use ('openai', 'ollama', 'lmstudio', 'huggingface', 'google-edge').
        model_name: The name of the model to use.
        **kwargs: Additional arguments for the specific ChatModel.

    Returns:
        A Langchain ChatModel instance.

    Raises:
        ValueError: If the specified backend is not supported.
    """
    # Ensure required environment variables are loaded
    from dotenv import load_dotenv
    load_dotenv()

    if backend == 'openai':
        # Requires OPENAI_API_KEY environment variable set unless provided
        api_key = kwargs.pop("openai_api_key", os.getenv("OPENAI_API_KEY"))
        return ChatOpenAI(model=model_name, api_key=api_key, **kwargs)
    elif backend == 'ollama':
        # Requires Ollama to be installed and running, and the specified model pulled (e.g., 'ollama pull llama2').
        # You can customise the base_url via OLLAMA_HOST or `base_url` kwarg.
        # Optionally enable thinking mode via the `think` kwarg or OLLAMA_THINK env variable.
        base_url = kwargs.pop("base_url", os.getenv("OLLAMA_HOST"))
        think = kwargs.pop("think", None)
        return ChatOllamaThinking(model=model_name, base_url=base_url, think=think, **kwargs)
    elif backend == 'lmstudio':
        # Example implementation for LM Studio (OpenAI-compatible API)
        # Requires LM Studio to be running with an OpenAI-compatible server.
        # The model_name should match the "Model" shown in LM Studio's server tab (often "local-model").
        # You can override the API URL with the LMSTUDIO_API_URL environment variable if not using localhost:1234.
        # LM Studio exposes an OpenAI-compatible API. Override the URL if needed.
        base_url = kwargs.pop("base_url", os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1"))
        return ChatOpenAI(
            base_url=base_url,
            api_key="not-needed",
            model=model_name,
            **kwargs,
        )
    elif backend == 'huggingface':
        # Uses HuggingFace's `transformers` pipeline to load a local or hub model.
        # You can override the `pipeline_task` or provide additional
        # `pipeline_kwargs` via **kwargs. For example:
        #   get_llm('huggingface', 'mistralai/Mistral-7B-Instruct-v0.2',
        #          pipeline_task='text-generation',
        #          pipeline_kwargs={'max_new_tokens': 256})
        pipeline_task = kwargs.pop("pipeline_task", "text-generation")
        pipeline_kwargs = kwargs.pop("pipeline_kwargs", {})
        hf_token = kwargs.pop("hf_token", os.getenv("HF_TOKEN"))
        pipe = pipeline(pipeline_task, model=model_name, token=hf_token, **pipeline_kwargs)
        return HuggingFacePipeline(pipeline=pipe, **kwargs)
    elif backend == 'google-edge':
        # Uses Google Generative AI (Gemini) via `langchain-google-genai`.
        # Requires a valid GOOGLE_API_KEY environment variable or provide
        # `google_api_key` in **kwargs.
        # Example:
        #   get_llm('google-edge', 'gemini-pro')
        api_key = kwargs.pop("google_api_key", os.getenv("GOOGLE_API_KEY"))
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

if __name__ == '__main__':
    # Example Usage:
    # To use OpenAI:
    # os.environ["OPENAI_API_KEY"] = "your-api-key" # Set your API key
    # openai_llm = get_llm('openai', 'gpt-4o', temperature=0)
    # print(openai_llm)

    # To use Ollama (assuming it's running with a model named 'llama2'):
    # ollama_llm = get_llm('ollama', 'llama2')
    # print(ollama_llm)

    # To use LM Studio (assuming it's running on localhost:1234):
    # lmstudio_llm = get_llm('lmstudio', 'local-model') # 'local-model' is often the default name
    # print(lmstudio_llm)

    # To use Hugging Face (example with a text generation model):
    # hf_llm = get_llm('huggingface', 'gpt2')
    # print(hf_llm)

    # To use Google Edge:
    # google_edge_llm = get_llm('google-edge', 'gemini-edge-model')
    # print(google_edge_llm)

    pass
