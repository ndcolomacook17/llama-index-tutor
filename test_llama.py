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

from llama_index import set_global_tokenizer
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

# use Huggingface embeddings
from llama_index.embeddings import HuggingFaceEmbedding

MODELS_PATH = "/Users/ianmclaughlin/Downloads"
DOCUMENTS_PATH = "/Users/ianmclaughlin/PycharmProjects/llama-index-tutor/documents"

if __name__ == "__main__":
  load_dotenv()

  llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url="/Users/nicocoloma-cook/Desktop/model-00002-of-00002.safetensors",
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=f"{MODELS_PATH}/llava-v1.5-7b-Q4_K.gguf",
    # temperature=0.1,
    # max_new_tokens=256,
    # # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    # context_window=3900,
    # # kwargs to pass to __call__()
    # generate_kwargs={},
    # # kwargs to pass to __init__()
    # # set to at least 1 to use GPU
    # model_kwargs={"n_gpu_layers": 1},
    # # transform inputs into Llama2 format
    # messages_to_prompt=messages_to_prompt,
    # completion_to_prompt=completion_to_prompt,
    # verbose=True,
  )

  # response = llm.complete("Give me a list of 3 random foods")
  # print(response.text)

  # response_iter = llm.stream_complete("Can you write me a poem about fast horses?")
  # for response in response_iter:
  #   print(response.delta, end="", flush=True)

  set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
  )

  embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

  # create a service context
  service_context = ServiceContext.from_defaults(
      llm=llm,
      embed_model=embed_model,
  )

  # load documents
  data_dir_path = "/Users/nicocoloma-cook/Desktop/llama_data"
  documents = SimpleDirectoryReader(data_dir_path).load_data()

  # create vector store index
  index = VectorStoreIndex.from_documents(
      documents, service_context=service_context
  )

  # set up query engine
  query_engine = index.as_query_engine()
  raw_query = "Give me a summary of these notes/ this pdf"

  response = query_engine.query(raw_query)
  print(response)