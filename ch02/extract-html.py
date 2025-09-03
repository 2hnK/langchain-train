from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader('https://www.naver.com')
docs = loader.load()

print(docs)