from flask import Flask, request, jsonify
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding
from transformers import AutoTokenizer

app = Flask(__name__)
MODELS_PATH = "/Users/ianmclaughlin/Downloads"
# DOCUMENTS_PATH = "/Users/ianmclaughlin/PycharmProjects/llama-index-tutor/documents"
DOCUMENTS_PATH = "/Users/ianmclaughlin/PycharmProjects/financial-provider-services"
MODEL_NAME="llava-v1.5-7b-Q4_K.gguf"

llm = LlamaCPP(model_path=f"{MODELS_PATH}/llava-v1.5-7b-Q4_K.gguf",
               temperature=.1,
               context_window=8000
               )


tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
documents = SimpleDirectoryReader(DOCUMENTS_PATH).load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    raw_query = data['query']
    response = query_engine.query(raw_query)
    return_data = {"model_response": str(response)}
    return jsonify(return_data)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)
