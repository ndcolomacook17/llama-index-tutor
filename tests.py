from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"

if __name__ == '__main__':
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        # model_url=model_url,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path="/Users/ianmclaughlin/Downloads/llava-v1.5-7b-Q4_K.gguf",
        # temperature=0.1,
        # max_new_tokens=256,
        # # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        # context_window=3900,
        # # kwargs to pass to __call__()
        # generate_kwargs={},
        # # kwargs to pass to __init__()
        # # set to at least 1 to use GPU
        # # model_kwargs={"n_gpu_layers": 1},
        # # transform inputs into Llama2 format
        # messages_to_prompt=messages_to_prompt,
        # completion_to_prompt=completion_to_prompt,
        # verbose=True,
    )
    response = llm.stream_complete("Give me a list of 10 hotdogs")
    for r in response:
        print(r.delta, end="", flush=True)