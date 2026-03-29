import os
from operator import itemgetter
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

def process_data(refined_df):
    """
    Process the refined dataset and create the vector store.
    """
    refined_df['combined_info'] = refined_df.apply(lambda row: f"Product ID: {row['pid']}. Product URL: {row['product_url']}. Product Name: {row['product_name']}. Primary Category: {row['primary_category']}. Retail Price: ${row['retail_price']}. Discounted Price: ${row['discounted_price']}. Primary Image Link: {row['primary_image_link']}. Description: {row['description']}. Brand: {row['brand']}. Gender: {row['gender']}", axis=1)

    loader = DataFrameLoader(refined_df, page_content_column="combined_info")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore

def save_vectorstore(vectorstore, directory):
    vectorstore.save_local(directory)

def load_vectorstore(directory, embeddings):
    vectorstore = FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def format_docs(docs):
    """Helper to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def display_product_recommendation(refined_df):
    st.header("Product Recommendation")

    vectorstore_dir = 'vectorstore'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(vectorstore_dir):
        vectorstore = load_vectorstore(vectorstore_dir, embeddings)
    else:
        vectorstore = process_data(refined_df)
        save_vectorstore(vectorstore, vectorstore_dir)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # --- AI Chatbot Recommendation (Pure modern LCEL RAG) ---
    chatbot_system_prompt = """
    You are a friendly, conversational retail shopping assistant that helps customers find products that match their preferences.
    Use the following pieces of retrieved context and chat history to assist customers in finding what they are looking for.
    For each question, suggest three products, including their category, price, and current stock quantity.
    Sort the result by the cheapest product.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context:
    {context}
    """
    
    chatbot_prompt = ChatPromptTemplate.from_messages([
        ("system", chatbot_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Pure LCEL RAG Chain
    retriever = vectorstore.as_retriever()
    
    # This building block is the foundation of modern LangChain v1.x
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history")
        }
        | chatbot_prompt
        | llm
        | StrOutputParser()
    )

    # Initialize chat history in Streamlit session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # UI Inputs
    department = st.text_input("Product Department")
    category = st.text_input("Product Category")
    brand = st.text_input("Product Brand")
    price = st.text_input("Maximum Price Range")

    col1, col2 = st.columns([1, 4])

    with col1:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    if st.button("Get Recommendations", type="primary"):
        if not (department or category or brand or price):
            st.warning("Please enter at least one field for a recommendation.")
        else:
            question = f"Suggest three products in {department} category {category} from {brand} under {price}"
            
            # Invoke pure LCEL chain
            with st.spinner("Searching for the best matches in our inventory..."):
                response = rag_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history
                })
            
            # Update history
            st.session_state.chat_history.append(HumanMessage(content=question))
            st.session_state.chat_history.append(AIMessage(content=response))
            
            st.write(response)

    # Display Chat History if it exists
    if st.session_state.chat_history:
        st.divider()
        st.subheader("Recent Recommendations")
        for message in reversed(st.session_state.chat_history):
            if isinstance(message, AIMessage):
                st.info(message.content)
            elif isinstance(message, HumanMessage):
                st.caption(f"Search: {message.content}")