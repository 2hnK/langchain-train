from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./test.pdf')
docs = loader.load()

print(docs)