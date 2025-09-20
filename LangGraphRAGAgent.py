from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from dotenv import load_dotenv
import os

load_dotenv()


class State(TypedDict):
    """State for the RAG application."""
    question: str
    context: List[Document]
    answer: str


class LangGraphRAGAgent:
    """RAG Agent using LangGraph StateGraph pattern."""
    
    def __init__(self, vectordb, temperature=0.5):
        """Initialize the RAG agent with a vector database."""
        self.vectordb = vectordb
        self.llm = ChatOllama(
            model="gpt-oss:20b",
            base_url="http://localhost:11434",
            temperature=temperature
        )
        
        # Create the RAG prompt template
        self.prompt = self._create_rag_prompt()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _create_rag_prompt(self):
        """Create the RAG prompt template."""
        template = """You are a helpful assistant for Dematic employees. Use the following pieces of retrieved context to answer the question about people, projects, or company information.

If the question is about a specific person (like "Who is [Name]?"), look for information about their role, department, team, and responsibilities in the context.

If you don't know the answer based on the provided context, just say that you don't know.

Context: {context}

Question: {question}

Answer:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def retrieve(self, state: State) -> dict:
        """Retrieve relevant documents based on the question."""
        retrieved_docs = self.vectordb.similarity_search(
            state["question"], 
            k=10  # Retrieve top 10 most relevant documents
        )
        return {"context": retrieved_docs}
    
    def generate(self, state: State) -> dict:
        """Generate an answer based on the retrieved context."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        # Use the prompt template
        messages = self.prompt.format_messages(
            question=state["question"],
            context=docs_content
        )
        
        response = self.llm.invoke(messages)
        return {"answer": response.content}
    
    def _build_graph(self):
        """Build the LangGraph StateGraph."""
        # Create the graph builder
        graph_builder = StateGraph(State)
        
        # Add the sequence of nodes
        graph_builder.add_sequence([self.retrieve, self.generate])
        
        # Add the edge from START to retrieve
        graph_builder.add_edge(START, "retrieve")
        
        # Compile the graph
        return graph_builder.compile()
    
    def ask_question(self, question: str) -> str:
        """Ask a question and get an answer using the RAG system."""
        # Create initial state
        initial_state = {"question": question}
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return result["answer"]
    
    def stream_question(self, question: str):
        """Stream the processing steps for a question."""
        initial_state = {"question": question}
        
        for step in self.graph.stream(initial_state, stream_mode="updates"):
            yield step
