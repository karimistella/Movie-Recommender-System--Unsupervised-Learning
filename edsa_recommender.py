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
import streamlit as st
from PIL import Image

# Data handling dependencies
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

#Visualitation Libraries
import seaborn as sns


movies_meta = 'resources/data/movies.csv'
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
        st.header("Movies")
        TopRated = st.radio("Top Rated", 
        ('Movies', 'Directors', 'Actors'))

        #EDA
        if TopRated == "Movies":
            def eda_data(dataset):
                df = pd.read_csv(os.path.join(dataset))
                return df
            if st.checkbox("Preview Dataset"):
                data = eda_data(movies_meta)
                if st.button("Head"):
                    st.write(data.head())

            

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
        st.image(my_gif, use_column_width = 'always')
        st.markdown("""
        Bowls Analytic is a leading Data Science firm in Africa. Our main goal is to build proplem solving algorithims and models 
        to make thew world a better place to live in a nd to make life a little easier to enjoy.

        Contact: info@bowlsanalytic.com
         
        """)


        
        
        
       

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
