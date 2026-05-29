from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_and_index(vector_store,docs, chunk_size=500,overlap=70):

    text_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=overlap)
    all_splites = text_split.split_documents(docs)
    _=vector_store.add_documents(documents=all_splites)
    return all_splites