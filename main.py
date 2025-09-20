from dotenv import load_dotenv
import os
from DocumentManager import DocumentManager
from EmbeddingManager import EmbeddingManager
from LangGraphRAGAgent import LangGraphRAGAgent

load_dotenv()

def main():
    # Initialising and loading documents
    doc_manager = DocumentManager()  # Will use DOCUMENTS_DIRECTORY environment variable
    doc_manager.load_documents()
    doc_manager.split_documents()

    # Creation and persistence of embeddings
    embed_manager = EmbeddingManager(doc_manager.all_sections)
    embed_manager.create_and_persist_embeddings()

    # Setup and use of LangGraph RAG agent
    bot = LangGraphRAGAgent(embed_manager.vectordb)
    
    print("LangGraph RAG System with Local Ollama Model Ready!")
    print("Ask me questions about your documents. Type 'bye' to exit.")
    print("Type 'stream' followed by your question to see processing steps.\n")
    
    # Interactive conversation loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == "bye":
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Check if user wants to see streaming steps
            if user_input.lower().startswith("stream "):
                question = user_input[7:].strip()  # Remove "stream " prefix
                if question:
                    print("Bot: Processing with streaming steps...")
                    print("\n--- Processing Steps ---")
                    for step in bot.stream_question(question):
                        print(f"Step: {step}")
                        print("---")
                    print("--- End Processing ---\n")
                else:
                    print("Please provide a question after 'stream'")
                continue
                
            print("Bot: ", end="")
            response = bot.ask_question(user_input)
            print(response)
            print()  # Add blank line for readability
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")
if __name__ == "__main__":
    main()