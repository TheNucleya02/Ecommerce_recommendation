import streamlit as st
from data_processing import load_data, preprocess_data
from recommendation import display_product_recommendation

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Retail AI Assistant", page_icon="🛍️")

    dataset_path = 'flipkart_com-ecommerce_sample.csv'
    df = load_data(dataset_path)
    
    if df is not None:
        refined_df = preprocess_data(df)
        # Directly display product recommendation
        display_product_recommendation(refined_df)

if __name__ == '__main__':
    main()