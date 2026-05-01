# 🛍️ Seek Shop

A state-of-the-art, concierge-style E-commerce recommendation engine powered by **Generative AI** and **RAG (Retrieval-Augmented Generation)**. This system provides a premium, chat-first experience that understands semantic intent and natural language queries to offer highly relevant product suggestions.

### Live Link - https://seekshop.streamlit.app

### <img width="400" height="350" alt="Screenshot 2026-05-01 at 10 21 41 PM" src="https://github.com/user-attachments/assets/e45fb13c-54d1-402d-a124-2e2a6b2c9148" />


## 🚀 Key Features

- **Chat-First Experience:** A modern, minimalist interface centered around a sticky chat input for natural, conversational shopping.
- **Semantic AI Search (RAG):** Utilizes Retrieval-Augmented Generation to match user intent with product descriptions, moving beyond simple keyword matching.
- **Intelligent Landing Page:** Includes pill-shaped suggestion buttons to help users jump-start their search with common queries.
- **Premium Response Formatting:** AI suggestions are organized with professional headings, feature highlights, and clear pricing comparisons.
- **Local Vector Database:** Uses **FAISS** with **Mistral AI Embeddings** for blazing-fast, semantic product retrieval.
- **Performance Optimized:** Automatically samples the top 1,000 products for testing, reducing vector store generation time from 30 minutes to under 90 seconds.
- **Concierge Logic:** Context-aware chat history allows users to refine their search (e.g., "Show me these but in a different color").

## 🛠️ Tech Stack

- **Large Language Model:** Mistral Large Latest
- **Orchestration:** LangChain v1.x (LCEL)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** Mistral AI
- **Backend/Frontend:** Python, Streamlit
- **Data Science:** Pandas

## 📂 Project Structure

```text
Ecommerce_recommendation/
├── app.py                # Main Streamlit entry point
├── recommendation.py     # Premium UI, Chat Logic, and RAG Implementation
├── data_processing.py    # Intelligent data cleaning and 1k row sampling
├── vectorstore/          # Local FAISS index (generated on first run)
├── requirements.txt      # Project dependencies
├── .env                  # Environment secrets (API Keys)
└── .gitignore            # Git exclusion rules
```

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/Ecommerce_recommendation.git
   cd Ecommerce_recommendation
   ```

2. **Set up Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install faiss-cpu  # Required for vector search
   ```

4. **Configure API Keys:**
   Create a `.env` file in the root directory:
   ```env
   MISTRAL_API_KEY=your_mistral_api_key
   ```

## 🚀 Usage

1. **Start the Application:**
   ```bash
   streamlit run app.py
   ```
2. **First Run:** The system will process the dataset and generate the local FAISS vector store. Thanks to recent optimizations, this will take less than 2 minutes.
3. **Get Recommendations:**
   - Use the **Suggestion Pills** on the landing page for quick results.
   - Type naturally in the **Chat Input** at the bottom (e.g., "Suggest a formal outfit for a wedding").
   - Refine your search by asking follow-up questions.

## 📊 Dataset

The system is optimized for the **Flipkart E-commerce dataset**, providing rich metadata such as names, descriptions, and current pricing.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for new features or optimizations.

## 📜 License

This project is licensed under the MIT License.
