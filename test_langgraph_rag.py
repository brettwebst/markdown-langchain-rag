#!/usr/bin/env python3
"""
Test script for the LangGraph RAG implementation.
This script tests the basic functionality without requiring the full document loading process.
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from LangGraphRAGAgent import LangGraphRAGAgent

load_dotenv()

def create_test_vectorstore():
    """Create a test vector store with sample documents."""
    # Sample documents for testing
    test_docs = [
        Document(
            page_content="Dematic is a leading supplier of integrated automated technology, software and services for supply chain optimization.",
            metadata={"source": "company_info.md", "section": "overview"}
        ),
        Document(
            page_content="The company specializes in automated material handling systems, warehouse management software, and supply chain solutions.",
            metadata={"source": "company_info.md", "section": "services"}
        ),
        Document(
            page_content="Dematic serves customers in manufacturing, distribution, and e-commerce industries worldwide.",
            metadata={"source": "company_info.md", "section": "customers"}
        )
    ]
    
    # Create embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store
    vectordb = Chroma.from_documents(
        documents=test_docs, 
        embedding=embedding, 
        persist_directory='test_db'
    )
    
    return vectordb

def test_langgraph_rag():
    """Test the LangGraph RAG agent."""
    print("Creating test vector store...")
    vectordb = create_test_vectorstore()
    
    print("Initializing LangGraph RAG agent...")
    try:
        agent = LangGraphRAGAgent(vectordb)
        print("✓ LangGraph RAG agent initialized successfully!")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        return False
    
    # Test questions
    test_questions = [
        "What does Dematic do?",
        "What industries does Dematic serve?",
        "What kind of software does Dematic provide?"
    ]
    
    print("\nTesting questions...")
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        try:
            answer = agent.ask_question(question)
            print(f"Answer: {answer}")
            print("✓ Question answered successfully!")
        except Exception as e:
            print(f"✗ Failed to answer question: {e}")
            return False
    
    print("\nTesting streaming functionality...")
    try:
        question = "What is Dematic's main business?"
        print(f"Streaming question: {question}")
        for step in agent.stream_question(question):
            print(f"Step: {step}")
        print("✓ Streaming functionality works!")
    except Exception as e:
        print(f"✗ Streaming failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The LangGraph RAG implementation is working correctly.")
    return True

if __name__ == "__main__":
    success = test_langgraph_rag()
    if not success:
        print("\n❌ Some tests failed. Please check the implementation.")
        exit(1)
