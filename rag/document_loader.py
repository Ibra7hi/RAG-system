from langchain_community.document_loaders import PyPDFLoader

def load_pdf_document(file_path: str):
    """Loads and parses a PDF document from a given file path"""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs
