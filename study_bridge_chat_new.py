import chromadb
from huggingface_hub import login
from pathlib import Path

from auto_gptq import AutoGPTQForCausalLM
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

login_token="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"
mistral_models_path = Path.home().joinpath('Final Structure', 'Mistral', 'CapybaraHermes-GPTQ')
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
message_history = []
prompt_context = """You are StudyBridge, a specialized assistance chatbot for the Technical University of Applied Sciences Bingen (TH-Bingen). Your purpose is to help prospective and current students navigate information about the university.

Core responsibilities:
- Provide accurate information based solely on the provided knowledge base
- If you're unsure or information is missing, acknowledge this clearly
- Keep responses concise and focused on the user's specific question
- When discussing deadlines or important dates, emphasize their significance
- For complex application processes, break down information into clear steps
- Use a friendly, professional tone appropriate for academic communication

Guidelines:
- Only answer the most recent user message in the chat history
- Base your responses on the provided related information
- If information seems outdated or contradictory, mention this to the user
- For questions outside your knowledge base, direct users to the university's official contact channels

Language:
- Respond in the same language as the user's question
- Use clear, straightforward language avoiding academic jargon unless necessary"""


def get_related_chunks(input_query, n_chunks):
    persistent_client = chromadb.PersistentClient(path="./chroma_db")
    collection = persistent_client.get_collection(name="th_bingen_collection")
    embedding_model = SentenceTransformer(embedding_model_name)
    
    question_embedding = embedding_model.encode(input_query)
    
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_chunks, 
        include=['documents', 'metadatas']  
    )
    
    related_results = []
    for idx, (document, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        related_results.append(f"Source: {metadata['source']}; Content: {document}")

    return related_results


def invoke_llm(input_string):
    global tokenizer, model
    
    inputs = tokenizer(input_string, 
                      return_tensors="pt",
                      add_special_tokens=True).to("cuda")
    attention_mask = inputs['input_ids'].ne(tokenizer.pad_token_id).long()
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=200,  # Instead of max_length, more precise control
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # Helps prevent repetitive text
        )

    full_response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    # Clean up the response - remove any "Answer:" prefix if present
    assistant_response = full_response
    prefixes_to_remove = ["Output:", "Answer:", "Assistant:", "Response:"]
    for prefix in prefixes_to_remove:
        assistant_response = assistant_response.replace(prefix, "").strip()
    
    # Add to history
    message_history.append({"role": "assistant", "content": assistant_response})
    
    # Print only the new response
    print_bot_message(assistant_response)

    
def print_user_message():
    print(f"\n👤  User:")
    
    
def print_bot_message(message):
    print(f"\n🤖  Assistant:")
    print(f">>> {message}")
    print("\n" + "─" * 50)

    
def start_chat():
    while True:
        print_user_message()
        user_input = input(">>> ")
        if user_input == "/bye":
            break
        elif user_input == "/clear":
            message_history.clear()
            print("[*] Chat history has been cleared.")
            continue
        related_data = get_related_chunks(user_input, 3)
        message_history.append({"role": "user", "content": user_input})
        # print(message_history)
        combined_query = (
            f"{prompt_context}\n\n"
            f"Related information:\n{related_data}\n\n"
            f"Current conversation:\n{message_history}\n\n"
            f"Instructions: Provide a single, direct response to the user's last message. Do not generate additional questions or responses."
        )
        invoke_llm(combined_query)
    
    
def init():
    global tokenizer, model
    
    login(token=login_token)
    print("[*] Successfully logged into Huggingface Hub")
    
    model = AutoModelForCausalLM.from_pretrained(
        mistral_models_path,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(mistral_models_path)
    # Ensure the model knows about the pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("\n\n")
    print("╔" + "═"*48 + "╗")
    print("║" + " "*11 + "Welcome to Study Bridge" + " "*14 + "║")
    print("║" + " "*8 + "International Student Assistant" + " "*9 + "║")
    print("║" + " "*10 + "Your Gateway to Information" + " "*11 + "║")
    print("╚" + "═"*48 + "╝")
    print("[*] Model loaded")
    print("[*] Starting Chat ...")
    print("[*] Type '/bye' to exit")
    print("[*] Type '/clear' to reset the history")
    print("\n" + "─" * 50)

    
if __name__ == "__main__":
    init()
    start_chat()
    