from flask import Flask, request, jsonify
from llama_index import ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os
from llama_index import SQLDatabase
from llama_index.indices.struct_store import NLSQLTableQueryEngine
from sqlalchemy import create_engine

app = Flask(__name__)
load_dotenv()
models_path, model_name, data_path = os.getenv("MODELS_PATH"), os.getenv("MODEL_NAME"), os.getenv("DATA_PATH")
mysql_username, mysql_password = os.getenv("MYSQL_USERNAME"), os.getenv("MYSQL_PASSWORD")
host, port = os.getenv("HOST"), os.getenv("PORT")
database = os.getenv("DATABASE")

# Create db engine
engine = create_engine(f"mysql://{mysql_username}:{mysql_password}@{host}:{port}/{database}")

llm = LlamaCPP(model_path=f"{models_path}/{model_name}",
            temperature=.1,
            context_window=4096
            # can pull model directly from HuggingFace
            # model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
            # max_new_tokens=256,
            # # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            # #    kwargs to pass to __call__()
            # generate_kwargs={},
            # # kwargs to pass to __init__()
            # # set to at least 1 to use GPU
            # model_kwargs={"n_gpu_layers": 1},
            # # transform inputs into Llama2 format
            # messages_to_prompt=messages_to_prompt,
            # completion_to_prompt=completion_to_prompt,
            # verbose=True,
            )

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
# set_global_tokenizer(AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf"))
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# TODO: add tables to context window
tables_to_include = []

sql_database = SQLDatabase(engine, include_tables=tables_to_include)

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=tables_to_include,
    service_context=service_context
)
query_str = "How many records are in each table?"
response = query_engine.query(query_str)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    raw_query = data['query']
    response = query_engine.query(raw_query)
    return_data = {"model_response": str(response)}
    return jsonify(return_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)