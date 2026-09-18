import ollama

def main():
    # Example conversation with streaming response
    messages = [{
        'role': 'user',
        'content': 'Write a short poem about artificial intelligence.'
    }]
    
    print("Asking Gemma to write a poem...\n")
    
    # Stream the response
    for chunk in ollama.chat(
        model='gemma3:12b',
        messages=messages,
        stream=True,
        options={
            'temperature': 0.7
        }
    ):
        if 'message' in chunk:
            print(chunk['message']['content'], end='')
    
    print("\n\nDone!")

if __name__ == "__main__":
    main()
