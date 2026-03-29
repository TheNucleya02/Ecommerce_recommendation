# 🛍️ E-commerce GenAI Recommendation System

A state-of-the-art E-commerce product recommendation engine powered by **Generative AI** and **RAG (Retrieval-Augmented Generation)**. This system goes beyond traditional filtering by understanding semantic meaning and natural language queries to provide highly relevant product suggestions.

![Product Recommendation Demo](https://via.placeholder.com/800x450.png?text=GIF+Placeholder+-+Insert+Demo+GIF+Here)
*Note: Add your demo GIF here to showcase the interactive UI!*

## 🚀 Key Features

- **Semantic AI Search (RAG):** Utilizes Retrieval-Augmented Generation to match user intent with product descriptions, not just keywords.
- **Modern LangChain Orchestration:** Built using **LangChain v1.x** and **LCEL (LangChain Expression Language)** for modular and efficient AI chains.
- **Hybrid Recommendations:** Combine specific filters (Department, Category, Brand, Price) with natural language processing.
- **Local Vector Database:** Uses **FAISS** with **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) for blazing-fast, persistent product retrieval.
- **Google Gemini Integration:** Leverages **Gemini 2.0 Flash** for conversational and intelligent product summaries.
- **Interactive Streamlit UI:** A clean, responsive dashboard for exploring data and getting recommendations with persistent session-based chat history.
- **Automated Data Processing:** Intelligent cleaning and tokenization of the Flipkart e-commerce dataset.

## 🛠️ Tech Stack

- **Large Language Model:** Google Gemini 2.0 Flash
- **Orchestration:** LangChain v1.x (LCEL)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** HuggingFace (Sentence Transformers)
- **Backend/Frontend:** Python, Streamlit
- **Data Science:** Pandas, Matplotlib, Seaborn

## 📂 Project Structure

```text
Ecommerce_recommendation/
├── app.py                # Main Streamlit entrance
├── recommendation.py     # AI Logic, LCEL Chains, and RAG Implementation
├── data_processing.py    # Data cleaning and exploratory analysis
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
   ```

4. **Configure API Keys:**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_gemini_api_key
   ```

## 🚀 Usage

1. **Start the Application:**
   ```bash
   streamlit run app.py
   ```
2. **First Run:** The system will process the dataset and generate the local FAISS vector store automatically.
3. **Get Recommendations:**
   - Navigate to the "Product Recommendation" section.
   - Enter your preferences (Department, Brand, Price).
   - The AI will fetch the best matches and provide a conversational summary of the top 3 products.

## 📊 Dataset

The system is optimized for the **Flipkart E-commerce dataset**, which includes rich product metadata such as names, descriptions, pricing, and image links.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for new features or optimizations.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
