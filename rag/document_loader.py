import bs4
from langchain_community.document_loaders import WebBaseLoader

def load_web_documents(url: str):
    """Loads and parses web documents from a given URL."""
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    return docs
