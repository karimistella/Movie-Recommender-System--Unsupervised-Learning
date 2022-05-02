"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# SCRIPT DEPENDENCIES

# for pattern searching and extraction 
import re

# libraries for data analysis and manipulation
import pandas as pd
import numpy as np

# libraries for numerical efficiencies
import scipy as sp
from scipy import stats

# libraries for string matching
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# libraries for stopwords
from wordcloud import STOPWORDS

# library to evaluate strings containing python literals
from ast import literal_eval

# libraries for natural language processing
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# libraries for entity featurization and similarity computation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


STOP_WORDS = set(STOPWORDS) 

# LOADING DATA
imdb = pd.read_csv('resources/data/imdb_data.csv')
movies = pd.read_csv('resources/data/movies.csv')
meta_data = pd.read_csv('resources/data/movies_metadata.csv')
genome_scores = pd.read_csv('resources/data/genome_scores.csv')
genome_tags = pd.read_csv('resources/data/genome_tags.csv')
train = pd.read_csv('resources/data/train.csv')
links = pd.read_csv('resources/data/links.csv')
tags = pd.read_csv('resources/data/tags.csv')
scores_n_tags = pd.read_csv('resources/data/scores_tags.csv')


def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """

    def convert_columns(data):
        """
        This function takes in a dataset and converts the 
        dtype of each column to a lesser version to reduce
        the size of the dataset for further operations.
        """
        
        for col in data.columns: # iterate over the columns in the dataset
            
            if data[col].dtype == 'object':
                data[col] = data[col].astype('category') # convert objects to categories
            
            if data[col].dtype == 'int64':
                data[col] = data[col].astype('int32') # convert int64 to int32
            
            if data[col].dtype == 'float64':
                data[col] = data[col].astype('float32') # convert float64 to float32
            
        return data # return converted data

    def clean_text(text):

        # split text on '|'
        text_split = text.split('|')
        
        # replace the space between the actors first name and
        # lastname with an underscore, convert to lowercase
        # and then join into a string.
        cleaned_text = ' '.join([x.replace(' ', '_') if len(x) > 0 else '' for x in text_split]).lower()
        
        # return transformed text
        return cleaned_text
    
    def remove_stopwords(text):
        split_text = text.split()
        text = ' '.join([x for x in split_text if x not in STOP_WORDS])  
        return text 
    
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
    
    def tag_to_str(tag_list):
        
        try:
            tag_str = ' '.join(tag_list)
        except (TypeError, ValueError):
            tag_str = str(tag_list)
            
        return tag_str

    # lets define our tokenizer
    def tokenize(text):

        return text.split(' ')

    # Lemming and stemming function
    def transform(text):

        # tokenize words
        words = tokenize(text)
        
        # define both Lemmatizer and stemmer
        lemmer = WordNetLemmatizer()
        stemmer = SnowballStemmer(language='english')
        
        # lemmatize and stem words
        lemmatized = [lemmer.lemmatize(x) for x in words]
        stemmed = [stemmer.stem(x) for x in lemmatized]
        
        return ' '.join(stemmed)
    
    def fill_na(df):
        """
        This function fills columns with null values 
        with the appropriate values depending on the 
        datatype of data stored by the column
        """
        for column in df.columns:
            # impute based on the type of column
            if df[column].dtype == 'object':
                # fill categorical columns with blanks
                df[column] = df[column].fillna('')
            elif df[column].dtype == 'float32' or df[column].dtype == 'float64':
                # fill numerical columns with the mean value of the column
                df[column] = df[column].fillna(df[column].mean())
    
        return df

    # Apply convert columns to downcast the dtypes of columns
    genome_scores = convert_columns(genome_scores)
    genome_tags= convert_columns(genome_tags)
    train = convert_columns(train)
    links = convert_columns(links)
    tags = convert_columns(tags)

    #### DATA CLEANING ####

    # IMDB DATA

    imdb_copy = imdb.copy(deep=True)
    imdb_copy['title_cast'] = imdb_copy['title_cast'].fillna('')
    imdb_copy['director'] = imdb_copy['director'].fillna('')
    imdb_copy['plot_keywords'] = imdb_copy['plot_keywords'].fillna('')

    # apply clean_text function to each row
    imdb_copy['title_cast'] = imdb_copy['title_cast'].apply(clean_text) 

    # replace the space between the directors' first name and 
    # last names with an underscore, and convert to lowercase
    imdb_copy['director'] = imdb_copy['director'].apply(lambda row: row.replace(' ', '_').lower())

    # pick column and use .apply() with the lambda function to replace "|" character
    # with a space.
    imdb_copy['plot_keywords'] = imdb_copy['plot_keywords']\
                                .apply(lambda row: row.replace('|', ' ')) 

    # remove stopwords from plot_keywords
    imdb_copy['plot_keywords'] = imdb_copy['plot_keywords'].apply(remove_stopwords)


    # MOVIES DATA

    # make a copy of movies data
    movies_copy = movies.copy(deep=True)
    
    movies_copy['year'] = movies_copy['title'].apply(lambda x: x[-7:].replace('(', '').replace(')', ''))

    #convert year to a numeric column
    movies_copy['year'] = pd.to_numeric(movies_copy['year'], errors='coerce', downcast='float')

    movies_copy['title'] = movies_copy['title'].apply(lambda x: x[:-7].strip().lower())

    # replace '|' with an empty space
    movies_copy['genres'] = movies_copy['genres'].apply(lambda row: row.replace('|', ' ').lower())


    # META DATA

    # make a copy
    meta_copy = meta_data.copy(deep=True)

    # apply the decompose function on the genres column.
    meta_copy['genres'] = meta_copy['genres'].apply(decompose)

    re_pattern = '\d+' # regex to extract 1 or more digits

    # applying regex, search through every row text, find every digits
    # in the text then return the group of texts if the row is not 'null'
    meta_copy['imdb_id'] = meta_copy['imdb_id']\
                            .apply(lambda row: re.search(re_pattern, row)\
                                .group() if type(row) == str else np.nan)\
                                    .astype('float32') # convert the column to dtype 'int32'

    # rename the 'imdb_id' to 'imdbId' 
    meta_copy = meta_copy.rename(columns={'imdb_id': 'imdbId'})

    # convert popularity to float32
    meta_copy['popularity'] = pd.to_numeric(meta_copy['popularity'], downcast='float', errors='coerce')

    # extract production_companies from the dictionary of production companies
    # while replacing all forms of boolean and blank values with np.nan
    meta_copy['production_companies'] = meta_copy['production_companies']\
                                    .apply(lambda x: np.nan if x in ('', ' ', 'True', True, 'False', False) else x)\
                                    .apply(decompose)

    # extract production_countries from the dictionary of production countries
    # while replacing all forms of boolean and blank values with np.nan
    meta_copy['production_countries'] = meta_copy['production_countries']\
                                    .apply(lambda x: np.nan if x in ('', ' ', 'True', True, 'False', False) else x)\
                                    .apply(decompose)

    # extract language from spoken_languages
    # here, we change the key of the decompose function
    # because we want the encoding of the language
    # not the language name itself
    meta_copy['spoken_languages'] = meta_copy['spoken_languages']\
                                        .apply(decompose, args=('iso_639_1',))

    # Convert title column to lowercase
    meta_copy['title'] = meta_copy['title'].str.lower()


    #### FEATURE ENGINEERING ####
    
    imdb_columns_of_interest = ['movieId', 'title_cast', 'director', 'runtime', 'plot_keywords']
    meta_columns_of_interest = ['imdbId', 'title', 'spoken_languages', 'overview', 'popularity',
                            'production_companies', 'production_countries',
                            'tagline', 'vote_average', 'vote_count']

    # merge links and meta_data
    meta_link = links.merge(meta_copy[meta_columns_of_interest], on='imdbId')
    
    # engineer new feature call movie_description by adding the overview and tagline columns together
    meta_link['movie_description'] = meta_link['overview'].str.lower() + " " + meta_link['tagline'].str.lower()

    #convert title column to lowercase
    meta_link['title'] = meta_link['title'].str.lower()

    # drop unwanted columns
    columns_to_drop = ['tmdbId', 'overview', 'tagline']
    meta_link.drop(columns_to_drop, axis=1, inplace=True)

    # merge meta_link and imdb dataframes using a left join
    # to keep all the rows of data in meta_link
    imdb_meta = meta_link.merge(imdb_copy[imdb_columns_of_interest], how='left', on='movieId')

    # merge imdb_meta and movies dataframes
    all_movies = movies_copy.merge(imdb_meta.drop('title', axis=1), on='movieId')

    # drop imdbId
    all_movies.drop('imdbId', axis=1, inplace=True)

    # group tag df on movie id, squeezing tag into a list
    tags_grouped = tags.drop('userId',axis=1).groupby(['movieId'])['tag'].apply(list).reset_index()

    # make tag column a string of tags separated by a space (' ')
    tags_grouped['tag'] = tags_grouped['tag'].apply(tag_to_str)

    # set relevance threshold for genome tags
    threshold = 0.80 # 80% relevance at least

    # filter dataframe based on threshold
    relevant_genomes = genome_scores[genome_scores['relevance'] >= threshold]

    # merge genome_tags and relevant_genomes
    scores_tags = genome_tags.merge(relevant_genomes, on='tagId')

    # group counts of movieId by tags
    movie_count = scores_tags.groupby('tag')['movieId'].count()\
                .reset_index().rename(columns={'movieId':'movieId_counts'})

    # STRING MATCHING USING FUZZY WUZZY
    # initialize empty lists to store words that match given words
    # and by how much they match.
    # this process is time consuming. 
    
#     match_list = []
#     ratio_list = []

#     # set the bad tags as the entire tags in the scores_tags df
#     bad_tags = scores_tags['tag'].values

#     # set the good tags as tags with occurrences greater than 100
#     good_tags = movie_count.tag[movie_count.movieId_counts > 100].values

#     # set a threshold for the match ratio
    tag_threshold = 80

#     # loop through the bad_tags
#     for b_tag in bad_tags:

#         # extract the best matching words and the ratio 
#         process_extract = process.extractOne(b_tag, good_tags, scorer=fuzz.token_sort_ratio)

#         # append words to match list
#         match_list.append(process_extract[0])

#         # append ratios to ratio list
#         ratio_list.append(process_extract[1])
        

    # add new columns in the scores_tags df for the matches
    # and their respective ratios
#     scores_tags['matches'] = match_list
#     scores_tags['match_ratio'] = ratio_list

    # the FUZZYWUZZY process takes a lot of time and
    # may impact our algorithm negatively
    # the result of the process was saved into a csv file
    # and has been read.
    scores_tags = scores_n_tags
    

    # filter the df based on set threshold
    filtered_tags_threshold = scores_tags[scores_tags.match_ratio >= tag_threshold]

    # adjust the ags df accordingly
    adjusted_tags = pd.merge(scores_tags[['movieId', 'tag']], \
                            filtered_tags_threshold[['tag', 'matches']], on='tag')

    # group count of tags by movieId and matches
    cleaned_tags = adjusted_tags.groupby(['movieId', 'matches'])['tag'].count()\
                            .reset_index().rename(columns={'tag':'tag_count', 'matches':'tag'})

    # group by movieId while turning tags in a list of tags
    # per movieId
    cleaned_tags = cleaned_tags.groupby('movieId')['tag'].agg(list).reset_index()

    # convert list of tags into a string
    cleaned_tags['tag'] = cleaned_tags['tag'].apply(tag_to_str)

    # Merge cleaned_tags and tag_grouped
    tags_merged = tags_grouped.merge(cleaned_tags, how='left', on='movieId')
    tags_merged = tags_merged.fillna('') # fillna with blank

    # create new column 'tag' which is a string concatenation of 
    # the tags from tags_grouped df and cleaned_tags df
    tags_merged['tag'] = tags_merged['tag_x'] + " " + tags_merged['tag_y'] 

    # drop unwanted columns
    tags_merged.drop(['tag_x', 'tag_y'], axis=1, inplace=True)

    # from the train(ratings) df, get the mean rating for each
    # movie and the number of users that rated the movie.
    ave_ratings = train.groupby(['movieId'])\
                            .agg({'rating':'mean', 'userId':'count'}).reset_index()\
                            .rename(columns={'rating':'ave_rating', 'userId':'rating_count'})

    # merge ave_ratings and all_movies
    full_movie = ave_ratings.merge(all_movies, how='right', on='movieId')

    # rearrange the columns of the dataset
    columns = ['movieId', 'title', 'genres', 'title_cast', 'director', 'production_companies',
            'production_countries', 'movie_description', 'plot_keywords', 'spoken_languages', 'runtime', 'ave_rating', 'rating_count', 'vote_average', 
            'vote_count', 'popularity']
    full_movie = full_movie[columns]

    # merge full movie and tags_merged
    full_movies = full_movie.merge(tags_merged, how='left', on='movieId')
    full_movies['tag'] = full_movies['tag'].fillna('').str.lower()

    # TOKENIZE, LEMMATIZE AND STEM

    # fillna with blanks
    full_movies[['movie_description', 'plot_keywords', 'tag']] = full_movies[['movie_description', 'plot_keywords', 'tag']].fillna('')
    
    # apply the transform function which does the tokenization, lemmatization 
    # and stemming on all the necessary columns
    full_movies['movie_description'] = full_movies['movie_description'].apply(transform)
    full_movies['plot_keywords'] = full_movies['plot_keywords'].apply(transform)
    full_movies['tag'] = full_movies['tag'].apply(transform)

    # apply fill_na function
    full_movies = fill_na(full_movies)

    # remove stopwords
    full_movies['movie_description'] = full_movies['movie_description'].apply(remove_stopwords)
    full_movies['tag'] = full_movies['tag'].apply(remove_stopwords)

    # apply convert columns function to downcast the 
    # datatype of each column, to reduce the size of the 
    # dataset
    full_movies = convert_columns(full_movies)

    # Subset of the data
    movies_subset = full_movies[:subset_size]

    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    data = data_preprocessing(27000)


    #### IMDB TOP 250 MOVIES ####
    
    # calculate the weighted average for a movie, using IMDB's formula
    data['imdb_wr'] = ((data['vote_count']/(data['vote_count']+25000)) * data['ave_rating'] +\
                        (25000/(data['vote_count']+25000)) * 7.0)

    # sort values from highest to lowest
    top_250_movies = data.sort_values(by='imdb_wr').head(250)

    # convert movie titles to list.
    top_250_movies = top_250_movies['title'].tolist()


    #### TOP RATED 100 MOVIES ####

    top_rated = data.sort_values(by='rating_count', ascending=False).head(100)
    
    # sort values by average rating 
    top_rated_100 = top_rated.sort_values(by='ave_rating', ascending=False)

    # convert to list
    top_rated_100 = top_rated_100['title'].tolist()


    #### MOST POPULAR 100 ####
    
    # sort by popularity
    most_popular_100 = data.sort_values(by='popularity', ascending=False).head(100)

    # convert to list
    most_popular_100 = most_popular_100['title'].tolist()

    
    def get_similarity(series):

        tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 1))
        tfidf_matrix = tf.fit_transform(series)

        tfidf_matrix = tfidf_matrix.astype('float32')
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        return cosine_sim

    def content(category):

        cosine_sim = get_similarity(category)
        return cosine_sim
    
    def get_recommendations(category):
        
        # Initializing the empty list of recommended movies
        recommended_movies = []
    
        cosine_sim = content(category)
        
        movies = data[['movieId', 'title']]
        movies  = movies.reset_index()
#         titles = movies['title']
        indices = pd.Series(movies['title'], index=movies.index)

#         indices = pd.Series(data['title'])
        try:
            # Getting the index of the movie that matches the title
            idx_1 = indices[indices == movie_list[0]].index[0]
            idx_2 = indices[indices == movie_list[1]].index[0]
            idx_3 = indices[indices == movie_list[2]].index[0]
        except:
            pass

        # Creating a Series with the similarity scores in descending order
        try:
            rank_1 = cosine_sim[idx_1]
            rank_2 = cosine_sim[idx_2]
            rank_3 = cosine_sim[idx_3]
        except:
            pass

         # Calculating the scores
        try:
            score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
            score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
            score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
        except:
            pass

        # Getting the indexes of the 10 most similar movies
        try:
            listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)
        except:
            return "No movies to recommend"
        # Store movie names
        recommended_movies = []
        # Appending the names of movies
        top_50_indexes = list(listings.iloc[1:50].index)
        # Removing chosen movies
        top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
        for i in top_indexes[:top_n]:
            recommended_movies.append(list(movies['title'])[i])
        return recommended_movies


    #### RECOMMENDATIONS BASED ON CAST AND CREW ####
    
    # increasing the appearance of the director to give 
    # it more weight
    cast_n_crew = data['title_cast'].astype('O') + " " + \
                    data['director'].astype('O').apply(lambda x: ' '.join([x, x, x]))

    based_on_cast_crew = get_recommendations(cast_n_crew)


    #### RECOMMENDATION BASED ON KEYWORDS AND MOVIE DESCRIPTION ####
    
    full_description = data['plot_keywords'].astype('object') + " " +\
                        data['movie_description'].astype('object')

    based_on_full_desc = get_recommendations(full_description)


    #### RECOMMENDATION BASED ON TAGS ####
    
    tags = data['tag']

    based_on_tags = get_recommendations(tags)

    #### RECOMMENDATION BASED ON PRODUCTION COMPANIES AND COUNTRIES ####
    
    production = data['production_companies'].astype('object') + " " +\
                 data['production_countries'].astype('object')
    
    based_on_production = get_recommendations(production)


    return (top_250_movies, top_rated_100, most_popular_100, based_on_cast_crew, 
            based_on_full_desc, based_on_tags, based_on_production)
