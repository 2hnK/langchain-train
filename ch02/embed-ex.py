from langchain_openai import OpenAIEmbeddings

model = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = model.embed_documents([
    "Hi there!",
    "Hey, how are you?",
    "I'm doing great, thank you!",
    "What's your name?",
    "My name is John Doe.",
])

print(embeddings)