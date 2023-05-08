# Document Q&A on Wikipedia articles
Run [document Q&A](https://python.langchain.com/en/latest/use_cases/question_answering.html) on Wikipedia articles. Use [Wikipedia-API](https://pypi.org/project/Wikipedia-API/) to search/retrieve/beautify Wikipedia articles, [LangChain](https://python.langchain.com/en/latest/index.html) for the Q&A framework, and OpenAI & [HuggingFace](https://huggingface.co/) models for embeddings and LLMs.

For the accompanying blog post, see [https://georgesung.github.io/ai/llm-qa-eval-wikipedia/](https://georgesung.github.io/ai/llm-qa-eval-wikipedia/)

## Instructions
For a batch run over different LLMs and embedding models, see the [Colab notebook](https://colab.research.google.com/drive/1p0cKg6LWzfuHLAeK4x-YBYjc0VBW6XoC?usp=sharing)

To run an interactive Gradio app, do the following:
* `pip install -r requirements.txt`
* If you're using OpenAI ada embeddings and/or GPT 3.5, then `cp template.env .env`, and edit `.env` to include your OpenAI API key
* `python gradio_app.py`

The meat of the code is in `WikipediaQA.py`.

## Results with different LLMs and embeddings
For detailed results and analysis, see the full blog post [here](https://georgesung.github.io/ai/llm-qa-eval-wikipedia/)
