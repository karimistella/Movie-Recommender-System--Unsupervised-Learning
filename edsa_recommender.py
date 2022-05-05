"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from matplotlib.cbook import Stack
import streamlit as st
from PIL import Image

# library to evaluate strings containing python literals
from ast import literal_eval

# Data handling dependencies
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import re

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

#Visualitation Libraries
import seaborn as sns


movies_meta = 'resources/data/movies_metadata.csv'
# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview", "Meet The Team", "About Us"]
    my_gif = Image.open('resources/imgs/Bowls_logo.jpeg')
    st.sidebar.image(my_gif, width = 200)
    
    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    

    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")
        st.header("Movies Data")
    

        #EDA
        if st.checkbox("Data Preview"):
            def eda_data(dataset):
                df = pd.read_csv(os.path.join(dataset))
                return df

            #Previwing Dataset
            st.markdown("""
            To see the first five rows of the dataset select "Head" and 
            to see the last 5 rows of the dataset select "Tail"
            """)
            if st.checkbox("Raw Uncleaned Data"):
                data = eda_data(movies_meta)
                options = st.radio("", ("Head", "Tail"))
                if options == "Head":
                    st.write(data.head())
                elif options == "Tail":
                    st.write(data.tail())

            #Cleaning Data

            meta_copy = eda_data(movies_meta).copy(deep=True)
            meta_copy.drop_duplicates(inplace=True)

            # Agenda 1: Column Belongs_to_collection should be converted to a boolean field (True or False)
            meta_copy['belongs_to_collection'] = meta_copy['belongs_to_collection'].apply(lambda row: True if type(row) != float else False )
            
            x = "[{'id': 12, 'name': 'Adventure'},{'id': 14, 'name': 'Fantasy'},{'id': 10751, 'name': 'Family'}]"
            x = literal_eval(x)

            # Agenda 2: Extract the genres from column genres as they are store in dictionaries in a list

            def decompose(text, key='name'):
                
                try:
                    # check if text is np.nan 
                    if type(text) == float:
                        decomposed_text = text
                        
                    #apply literal_exal to each row of data
                    eval_text = literal_eval(text)

                    # get the name key of each dictionary in the list
                    # store extracted name in a list
                    # join each item in the list into a string
                    decomposed_text = ' '.join([dictionary[key].replace(' ', '_') for dictionary in eval_text]).lower()
                
                except (ValueError, TypeError):
                    decomposed_text = np.nan
                    
                return decomposed_text

            # apply the decompose function on the genres column.
            meta_copy['genres'] = meta_copy['genres'].apply(decompose)



            # Agenda 3: Extract the digits from the imdb_id column and rename the column to imdbId

            re_pattern = '\d+' # regex to extract 1 or more digits

            # applying regex, search through every row text, find every digits
            # in the text then return the group of texts if the row is not 'null'
            meta_copy['imdb_id'] = meta_copy['imdb_id'].apply(lambda row: re.search(re_pattern, row).group() if type(row) == str else np.nan).astype('float32') # convert the column to dtype 'int32'

            # rename the 'imdb_id' to 'imdbId' 
            meta_copy = meta_copy.rename(columns={'imdb_id': 'ImdbId'})

            # Agenda 4: Convert popularity to float32
            meta_copy['popularity'] = pd.to_numeric(meta_copy['popularity'], downcast='float', errors='coerce')
            # Agenda 5: Extract production_companies and production_countries

            boolean = np.where((meta_copy['production_companies'] == True) | (meta_copy['production_companies'] == 'True') | (meta_copy['production_companies'] == False) | (meta_copy['production_companies'] == 'False') | (meta_copy['production_companies'] == 'nan') | (meta_copy['production_companies'] == np.nan)) 

            data = meta_copy.iloc[boolean]
            meta_copy['production_companies'] = meta_copy['production_companies'].apply(lambda x: np.nan if x in ('', ' ', 'True', True, 'False', False) else x).apply(decompose)

            meta_copy['production_countries'] = meta_copy['production_countries'].apply(lambda x: np.nan if x in ('', ' ', 'True', True, 'False', False) else x).apply(decompose)
            
            # Agenda 7: Extract language from spoken_languages

            # here, we change the key of the decompose function
            # because we want the encoding of the language
            # not the language name itself
            meta_copy['spoken_languages'] = meta_copy['spoken_languages'].apply(decompose, args=('iso_639_1',))

            # Agenda 8: Replace Zeros in budget with np.nan and make column a numerical column

            meta_copy['budget'] = meta_copy['budget'].apply(lambda row: np.nan if row == '0' else row)
            meta_copy['budget'] = pd.to_numeric(meta_copy['budget'], downcast='float', errors='coerce')

            # Agenda 9: Convert title to lowercase

            # Convert title column to lowercase
            meta_copy['title'] = meta_copy['title'].str.lower()

            if st.checkbox("Cleaned Data"):
                data2 = meta_copy
                options2 = st.radio("", ("First Five", "Last Five"))
                if options2 == "First Five":
                    st.write(data2.head())
                elif options2 == "Last Five":
                    st.write(data2.tail())
            
            if st.checkbox("Data Summary"):
                st.markdown("""
                Below is a brief summary of our movie dataset. 
                """)
                st.write(meta_copy.describe())

            
            st.header("Data Visualization")

            st.markdown("""
            We are displaying movies grouped by budget.
            """)

            fig= plt.figure(figsize=(9, 7))

            sns.pairplot(
                meta_copy[['budget', 'revenue', 'runtime']], corner=True
            )

            st.pyplot(fig)
            
            




            #axes = df.plot.bar(x='original_title', y='budget')


    
    if page_selection == "Meet The Team":
        st.title("Meet The Team")
        image = Image.open('resources/imgs/meettheteam.jpg')
        st.image(image)
        st.markdown("""
        At Bowls Analytics we have five talented data scientists who are not afraid to get their hands dirty cleaning the data and building
        best performing unsupervised models.

        These are :
        1. Michael Dairo 
        2. Stella Njuki 
        3. Winfred Mutinda 
        4. Lungisa Nhlakanipho Joctrey 
        5. Odutayo Odufuwa

        """ )
    if page_selection == "About Us":
        st.title("About Us")
        my_gif = Image.open('resources/imgs/Bowls_logo.jpeg')
        st.image(my_gif)
        st.markdown("""
        Bowls Analytic is a leading Data Science firm in Africa. Our main goal is to build proplem solving algorithims and models 
        to make thew world a better place to live in a nd to make life a little easier to enjoy.

        Contact: info@bowlsanalytic.com
         
        """)


        
        
        
       

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
