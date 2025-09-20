from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
import os

load_dotenv()


class ConversationalRetrievalAgent:
    # Initialize the ConversationalRetrievalAgent with a vector database and a temperature for the Ollama model
    def __init__(self, vectordb, temperature=0.5):
        self.vectordb = vectordb
        self.llm = ChatOllama(
            model="gpt-oss:20b",
            base_url="http://localhost:11434",
            temperature=temperature
        )
        self.chat_history = []

    # Method to get the chat history as a string
    def get_chat_history(self, inputs):
        res = []
        for human, ai in inputs:
            res.append(f"Human:{human}\nAI:{ai}")
        return "\n".join(res)

    # Method to set up the bot
    def setup_bot(self):
        # Create a retriever from the vector database with more documents for better retrieval
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 10})
        # Create a ConversationalRetrievalChain from the Ollama model and the retriever
        self.bot = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever,
            return_source_documents=True,
            get_chat_history=self.get_chat_history,
        )

    def generate_prompt(self, question):
        if not self.chat_history:
            # Enhanced prompt for first question with better context for person queries
            prompt = f"""You are a helpful assistant for Dematic employees. Use the following pieces of retrieved context to answer the question about people, projects, or company information.

If the question is about a specific person (like "Who is [Name]?"), look for information about their role, department, team, and responsibilities in the context.

If you don't know the answer based on the provided context, just say that you don't know.

Question: {question}
Context: 
Answer:"""
        else:
            # Context-aware prompt for follow-up questions
            context_entries = [f"Question: {q}\nAnswer: {a}" for q, a in self.chat_history[-3:]]
            context = "\n\n".join(context_entries)
            prompt = f"Using the context from recent conversations and the retrieved documents, answer the new question concisely and informatively.\n\nRecent conversation context:\n{context}\n\nNew question: {question}\n\nAnswer:"
        
        return prompt
    
    # Method to ask a question to the bot
    def ask_question(self, query):
        prompt = self.generate_prompt(query)
        # Invoke the bot with the question and the chat history
        result = self.bot.invoke({"question": prompt, "chat_history": self.chat_history})
        # Append the question and the bot's answer to the chat history
        self.chat_history.append((query, result["answer"]))
        
        # Return the bot's answer
        return result["answer"]
