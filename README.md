# Document Q&A on Wikipedia articles
Run [document Q&A](https://python.langchain.com/en/latest/use_cases/question_answering.html) on Wikipedia articles. Use [Wikipedia-API](https://pypi.org/project/Wikipedia-API/) to search/retrieve/beautify Wikipedia articles, [LangChain](https://python.langchain.com/en/latest/index.html) for the Q&A framework, and OpenAI & [HuggingFace](https://huggingface.co/) models for embeddings and LLMs.

For the accompanying blog post, see [https://georgesung.github.io/ai/llm-qa-eval-wikipedia/](https://georgesung.github.io/ai/llm-qa-eval-wikipedia/)

## Instructions
For a batch run over different LLMs and embedding models, you can run the notebook `WikipediaQA_batch_runs.ipynb` in your own compute instance, or run the same notebook on Colab:

<a target="_blank" href="https://colab.research.google.com/github/georgesung/LLM-WikipediaQA/blob/main/WikipediaQA_batch_runs.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

To run an interactive Gradio app, do the following:
* `pip install -r requirements.txt`
* If you're using OpenAI ada embeddings and/or GPT 3.5, then `cp template.env .env`, and edit `.env` to include your OpenAI API key
* `python gradio_app.py`

The meat of the code is in `WikipediaQA.py`.

## Results with different LLMs and embeddings
For detailed results and analysis, see the full blog post [here](https://georgesung.github.io/ai/llm-qa-eval-wikipedia/)
