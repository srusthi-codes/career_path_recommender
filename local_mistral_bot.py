from llama_cpp import Llama

llm = Llama(model_path="C:/Users/Arvind/Downloads/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

while True:
    user_input = input("You: ")
    result = llm(user_input, max_tokens=256, temperature=0.7)
    print("Bot:", result["choices"][0]["text"].strip())


