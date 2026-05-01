import pandas as pd
import streamlit as st
from ast import literal_eval

@st.cache_data
def load_data(dataset_path):
    """
    Load the dataset from the specified path.
    
    Args:
        dataset_path (str): Path to the dataset file.
        
    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    try:
        return pd.read_csv(dataset_path)
    except pd.errors.EmptyDataError:
        st.error("The dataset file is empty.")
        return None
    except pd.errors.ParserError:
        st.error("There was an error parsing the dataset file.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {str(e)}")
        return None

def extract_primary_category(product_category_tree):
    """
    Extract the primary category from the product category tree.
    
    Args:
        product_category_tree (str): Product category tree string.
        
    Returns:
        str: Primary category if found, None otherwise.
    """
    try:
        return literal_eval(product_category_tree)[0].split('>>')[0].strip()
    except (ValueError, SyntaxError, IndexError):
        return None

def extract_primary_image(image_str):
    """
    Extract the primary image URL from the image string.
    
    Args:
        image_str (str): Image string containing image URLs.
        
    Returns:
        str: Primary image URL if found, None otherwise.
    """
    try:
        images = literal_eval(image_str)
        if images and isinstance(images, list):
            return images[0]
    except (ValueError, SyntaxError):
        return None

def determine_gender(product_name, description):
    """
    Determine the gender based on the product name and description.
    
    Args:
        product_name (str): Product name.
        description (str): Product description.
        
    Returns:
        str: Gender category ('Women', 'Men', or 'Unisex').
    """
    keywords_women = ['women', 'woman', 'female', 'girls', 'girl', 'ladies', 'lady']
    keywords_men = ['men', 'man', 'male', 'boys', 'boy', 'gentlemen', 'gentleman']

    name_desc = f"{str(product_name).lower()} {str(description).lower()}"
    if any(keyword in name_desc for keyword in keywords_women):
        return 'Women'
    elif any(keyword in name_desc for keyword in keywords_men):
        return 'Men'
    else:
        return 'Unisex'

def preprocess_data(df):
    """
    Preprocess the dataset by extracting relevant information and cleaning the data.
    
    Args:
        df (pd.DataFrame): Input dataset DataFrame.
        
    Returns:
        pd.DataFrame: Preprocessed dataset DataFrame.
    """
    df['primary_category'] = df['product_category_tree'].apply(extract_primary_category)
    df['primary_image_link'] = df['image'].apply(extract_primary_image)
    # It adds a new column called 'gender' to the pandas DataFrame df by applying the determine_gender function to each row.
    df['gender'] = df.apply(lambda x: determine_gender(x['product_name'], x['description']), axis=1)

    columns_of_interest = ['pid', 'product_url', 'product_name', 'primary_category',
                           'retail_price', 'discounted_price', 'primary_image_link',
                           'description', 'brand', 'gender']
    refined_df = df[columns_of_interest]
    refined_df = refined_df.dropna(subset=['primary_category', 'retail_price', 'discounted_price'])
    return refined_df