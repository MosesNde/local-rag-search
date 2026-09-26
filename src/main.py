from ollama import chat

response = chat(
	model="qwen3:8b",
	messages=[
		{
			"role":"user",
			"content":"What is retrieval-augmented generation?"
		}
	],
)
print(response.message.content)