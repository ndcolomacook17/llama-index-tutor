# llama-index-tutor
Llama Index for education resource querying
TESTSTSTST


## Install on M3:
I cloned the llamacpp repo and did this:
```commandline
arch -arm64 pip install .  --upgrade --force-reinstall --no-cache-dir
``` 

## Install dependencies
```commandline
pip install -r requirements.txt
```

## Usage
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
```
