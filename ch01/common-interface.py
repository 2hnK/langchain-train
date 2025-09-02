from config import get_model

model = get_model()

completion = model.invoke("Hello, world! 안녕")
print(completion.content)

completions = model.batch(["Hello", "world!", "안녕"])
for completion in completions:
    print(completion.content)

for token in model.stream("Hello, world! 안녕"):
    print(token.content)