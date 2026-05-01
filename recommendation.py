import os
from operator import itemgetter
from dotenv import load_dotenv
import streamlit as st
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
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

    embeddings = MistralAIEmbeddings()
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

SUGGESTIONS = {
    "🏝️ Beach outfit ideas?": "Suggest some cool beach outfit ideas for summer",
    "👟 Best running shoes?": "I need durable running shoes under $100",
    "⌚ Formal watches?": "Show me some elegant formal watches for men",
}

@st.dialog("Legal disclaimer")
def show_disclaimer_dialog():
    st.caption("""
            This AI chatbot is powered by Mistral AI and public e-commerce data. Answers may be inaccurate, inefficient, or biased. Any use or decisions based on such answers should include reasonable practices including human oversight to ensure they are safe, accurate, and suitable for your intended purpose. We are not liable for any actions, losses, or damages resulting from the use of the chatbot. Do not enter any private, sensitive, personal, or regulated data.
        """)

def display_product_recommendation(refined_df):
    vectorstore_dir = 'vectorstore'
    embeddings = MistralAIEmbeddings()

    if os.path.exists(vectorstore_dir):
        try:
            vectorstore = load_vectorstore(vectorstore_dir, embeddings)
        except Exception as e:
            st.info("🔄 Refreshing Assistant memory...")
            vectorstore = process_data(refined_df)
            save_vectorstore(vectorstore, vectorstore_dir)
    else:
        vectorstore = process_data(refined_df)
        save_vectorstore(vectorstore, vectorstore_dir)

    llm = ChatMistralAI(model="mistral-large-latest")

    # --- Structured System Prompt ---
    chatbot_system_prompt = """
    You are a premium, expert Retail Shopping Assistant with a focus on luxury and high-end consumer experiences. 
    Your goal is to provide highly organized, visually appealing, and professional product recommendations that make shopping effortless.

    ### Tone and Style:
    - Use a sophisticated yet friendly tone.
    - Be concise but descriptive.
    - Treat every interaction as a high-end concierge service.

    ### Response Structure:
    1.  **### [Main Category Heading]**: A clear, bold heading for the category.
    2.  **Introduction**: A brief (1-2 sentences) personalized opening based on the user's request.
    3.  **The Selection**: For each recommended product, use this format:
        *   **[Product Name]** ([Brand]) — [Buy Now]({{product_url}})
            *   **Price**: ~~$[{{retail_price}}]~~ **$[{{discounted_price}}]** (Save $[{{savings}}]!)
            *   **Highlight**: A short, punchy sentence about what makes this specific item perfect for them.
    4.  **Why We Recommend This**: A brief explanation of the curation logic for this set of products.
    5.  **Pro Tip**: A small piece of expert advice related to the category (e.g., "Pair this watch with a leather strap for a more formal look").

    ### Guidelines:
    - ALWAYS place the product link (Buy Now) directly next to the product name.
    - ALWAYS use the modern price format: ~~$Retail Price~~ **$Discounted Price**.
    - If a price is the same, just show **$Price**.
    - Calculate the savings if possible.
    - Use high-quality markdown formatting for readability.
    - If no relevant products are found, offer a helpful alternative or ask for more details.

    Retrieved Context:
    {context}
    """
    
    chatbot_prompt = ChatPromptTemplate.from_messages([
        ("system", chatbot_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # RAG Chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
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

    # -----------------------------------------------------------------------------
    # Draw the UI.

    st.html('<div style="font-size: 5rem; line-height: 1">🛍️</div>')

    title_row = st.container()

    with title_row:
        st.title(
            "Seek Shop",
            anchor=False,
        )

    user_just_asked_initial_question = (
        "initial_question" in st.session_state and st.session_state.initial_question
    )

    user_just_clicked_suggestion = (
        "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
    )

    user_first_interaction = (
        user_just_asked_initial_question or user_just_clicked_suggestion
    )

    has_message_history = (
        "chat_history" in st.session_state and len(st.session_state.chat_history) > 0
    )

    # Show a different UI when the user hasn't asked a question yet.
    if not user_first_interaction and not has_message_history:
        st.session_state.chat_history = []

        with st.container():
            st.chat_input("Ask a question...", key="initial_question")

            selected_suggestion = st.pills(
                label="Examples",
                label_visibility="collapsed",
                options=SUGGESTIONS.keys(),
                key="selected_suggestion",
            )

        st.button(
            ":gray[:material/balance: Legal disclaimer]",
            type="tertiary",
            on_click=show_disclaimer_dialog,
        )

        st.stop()

    # Show chat input at the bottom when a question has been asked.
    user_message = st.chat_input("Ask a follow-up or search for products...")

    if not user_message:
        if user_just_asked_initial_question:
            user_message = st.session_state.initial_question
        if user_just_clicked_suggestion:
            user_message = SUGGESTIONS[st.session_state.selected_suggestion]

    with title_row:
        def clear_conversation():
            st.session_state.chat_history = []
            if "initial_question" in st.session_state:
                del st.session_state["initial_question"]
            if "selected_suggestion" in st.session_state:
                del st.session_state["selected_suggestion"]

        st.button(
            "Restart",
            icon=":material/refresh:",
            on_click=clear_conversation,
        )

    # Display chat messages from history as speech bubbles.
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🛍️"):
                st.container()  # Fix ghost message bug.
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    if user_message:
        # When the user posts a message...

        # Streamlit's Markdown engine interprets "$" as LaTeX code (used to
        # display math). The line below fixes it.
        user_message_escaped = user_message.replace("$", r"\$")

        # Display message as a speech bubble.
        with st.chat_message("user"):
            st.text(user_message_escaped)

        # Display assistant response as a speech bubble.
        with st.chat_message("assistant", avatar="🛍️"):
            with st.spinner("Analyzing inventory..."):
                response = rag_chain.invoke({
                    "input": user_message,
                    "chat_history": st.session_state.chat_history
                })

            # Put everything after the spinners in a container to fix the
            # ghost message bug.
            with st.container():
                st.markdown(response)

                # Add messages to chat history.
                st.session_state.chat_history.append(HumanMessage(content=user_message))
                st.session_state.chat_history.append(AIMessage(content=response))