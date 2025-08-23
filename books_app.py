import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Data Loading and Model Building ---
@st.cache_data
def load_and_prepare_data():
    """
    Dataset ko load karta hai aur required columns prepare karta hai.
    """
    try:
        books = pd.read_csv('data.csv', on_bad_lines='skip')
    except FileNotFoundError:
        st.error("data.csv file not found. Please place the file in the same directory.")
        return pd.DataFrame()
    
    # Required columns check karo
    required_columns = ['title', 'authors', 'categories', 'description']
    if not all(col in books.columns for col in required_columns):
        st.error(f"Required columns {required_columns} not found in the dataset.")
        return pd.DataFrame()
        
    # Sirf necessary columns select karo aur empty values remove karo
    books = books[required_columns]
    books.dropna(inplace=True)
    
    # Sabhi text features ko ek string me combine karo
    books['features'] = books['title'] + ' ' + books['authors'] + ' ' + books['categories'] + ' ' + books['description']
    
    return books

@st.cache_data
def get_cosine_sim_matrix(books_df):
    """
    TF-IDF vector aur cosine similarity matrix calculate karta hai.
    """
    if books_df.empty:
        return None
        
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books_df['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# --- Recommendation Function ---
def get_recommendations(title, cosine_sim, books_df):
    """
    Book title ke basis par similar books recommend karta hai.
    """
    if title not in books_df['title'].values:
        return pd.DataFrame()
    
    book_index = books_df[books_df['title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[book_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_books_indices = [i[0] for i in similarity_scores[1:6]]
    recommended_books = books_df.iloc[top_books_indices]
    
    return recommended_books

# --- Streamlit UI Section ---
def main():
    """
    Ye main function Streamlit app ka UI run karta hai.
    """
    st.title('📚 Book Recommendation System')
    st.markdown("""
    _Select a book from the dropdown menu to get recommendations!_
    """)

    # Data aur model load karo
    books_df = load_and_prepare_data()
    
    if books_df.empty:
        st.stop()
        
    cosine_sim = get_cosine_sim_matrix(books_df)
    
    if cosine_sim is None:
        st.error("Failed to build the recommendation model.")
        st.stop()

    book_list = books_df['title'].tolist()
    
    selected_book = st.selectbox('Choose a book:', book_list)

    if st.button('Show Recommendations'):
        recommendations = get_recommendations(selected_book, cosine_sim, books_df)
        if not recommendations.empty:
            st.subheader(f"Here are some books similar to '{selected_book}':")
            for index, row in recommendations.iterrows():
                st.write(f"**{row['title']}** by {row['authors']}")
        else:
            st.warning(f"Sorry, we couldn't find recommendations for '{selected_book}'.")

if __name__ == "__main__":
    main()
