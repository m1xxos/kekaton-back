import ollama

client = ollama.Client(host='http://localhost:11434')  # Замените на нужный хост
response = client.chat(model='llama3.1', messages=[{"role": "user", "content": "Привет"}])

print(response)