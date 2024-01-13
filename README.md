# llama-index-tutor
Llama Index for natural language to SQL querying. This `structued_data.py` example uses a local llama LLM for the model (download [here](https://huggingface.co/jartine/llava-v1.5-7B-GGUF/tree/main)) but any local or hosted model can be dropped in.


## Install on M3:
I cloned the llamacpp repo and did this:
```commandline
arch -arm64 pip install .  --upgrade --force-reinstall --no-cache-dir
``` 

## Install dependencies
```commandline
pip install -r requirements.txt
```

## Config
Touch a `.env` file
```commandline
touch .env
```
Ensure you define the following env vars:
```python
# dir where your local llms are
MODELS_PATH

# local llm model file (must be .gguf)
MODEL_NAME

# dir for local data to be indexed
DATA_PATH

# all database related creds (default mysql)
MYSQL_USERNAME, MYSQL_PASSWORD, HOST, PORT, DATABASE
```

## Usage
For natural language to SQL querying, this will spin up a flask server with a query input:
```commandline
python3 structured_data.py
```
Run:
```
http POST http://127.0.0.1:5050/query query="Some query about your in-context tables"
```
