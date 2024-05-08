#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ## Function Overview: `classify_time_of_day`
# 
# **Purpose**:
# - Classifies a given timestamp into a time of day category based on the hour.
# 
# **Parameters**:
# - `timestamp`: A datetime object containing the time information.
# 
# **Returns**:
# - `1` for morning (4 AM to 11:59 AM).
# - `2` for afternoon to evening (12 PM to 7:59 PM).
# - `3` for night (8 PM to 3:59 AM).
# 
# **Usage**:
# This function can be used to categorize activities or events into time-of-day segments for analysis or reporting.
# 

# In[2]:


def classify_time_of_day(timestamp):
    hour = timestamp.hour
    if 4 <= hour < 12:
        return 1
    elif 12 <= hour < 20:
        return 2
    else:
        return 3


# ## Data Handling Process
# 
# 1. **Load Metadata**:
#    - The metadata is loaded from `id_metadata.csv`, using tab (`'\t'`) as the delimiter, into the `metadata_df` DataFrame.
# 
# 2. **Load Listening History**:
#    - Listening history data is loaded from `lh_reduced.csv` into the `df` DataFrame.
# 
# 3. **Convert Timestamps**:
#    - The `timestamp` column in `df` is converted to datetime format to facilitate time-based operations.
# 
# 4. **Rename Columns**:
#    - In `metadata_df`, the column named `id` is renamed to `song` to align with the naming in `df`, facilitating easier merging or joining operations in future steps.
# 
# 5. **Classify Time of Day**:
#    - The `timestamp` column in `df` is used to apply the `classify_time_of_day` function, which categorizes timestamps into three time of day segments (morning, afternoon to evening, and night). The result is stored in a new column `time_of_day`.
# 
# 6. **Output DataFrame**:
#    - The final DataFrame `df`, now enhanced with the `time_of_day` categorization, is displayed or utilized for subsequent analysis.
# 

# In[3]:


metadata_path = '../data/id_metadata.csv'
listening_history_path = '../data/listening_history.csv'
metadata_df = pd.read_csv(metadata_path, delimiter='\t')
df = pd.read_csv(listening_history_path, delimiter='\t')

df['timestamp'] = pd.to_datetime(df['timestamp'])
metadata_df.rename(columns={'id': 'song'}, inplace=True)
df['time_of_day'] = df['timestamp'].apply(classify_time_of_day)
print(df)


# ## Data Filtering Process Based on Recent Dates
# 
# 1. **Determine Latest Date**:
#    - Calculate the most recent date (`latest_date`) in the `timestamp` column of the DataFrame `df`.
# 
# 2. **Compute Date for One Week Ago**:
#    - Subtract 7 days from the `latest_date` using `pd.Timedelta`, resulting in the date `one_week_ago`.
# 
# 3. **Filter Recent Data**:
#    - Restrict `df` to only include rows where the `timestamp` is on or after `one_week_ago`, effectively filtering the data to the last week.
# 
# 4. **Output Filtered DataFrame**:
#    - The resulting DataFrame `df` now contains only the records from the past week, ready for analysis or further processing.
# 

# In[4]:


latest_date = df['timestamp'].max()
one_week_ago = latest_date - pd.Timedelta(days=7)
df = df[df['timestamp'] >= one_week_ago]
print(df)


# ## Data Processing 
# 
# 1. **Merge DataFrames**:
#    - `df` is merged with selected columns from `metadata_df` based on the 'song' column. The selected columns include song attributes such as release year, popularity, danceability, energy, key, mode, valence, and tempo.
# 
# 2. **Standardize Numeric Data**:
#    - Identify numeric columns in the merged DataFrame using `select_dtypes`.
#    - Standardize these numeric columns using `StandardScaler` to normalize the data, aiding in model performance and stability.
# 

# In[5]:


df = pd.merge(df, metadata_df[
    ['song', 'release', 'popularity', 'danceability', 'energy', 'key', 'mode', 'valence', 'tempo']], on='song')

numeric_cols_df = df.select_dtypes(include=np.number).columns
sscaler = StandardScaler()
df[numeric_cols_df] = sscaler.fit_transform(df[numeric_cols_df])
print(df)


# ## DataFrame Operations Overview
# 
# 1. **Check for Missing Values**:
#    - `df.isnull().sum()` calculates the total number of missing values in each column of the DataFrame.
#    
# 2. **Identify Unique Entries**:
#    - `df.song.unique()` retrieves an array of unique song IDs from the `song` column.
#    - `df.user.unique()` retrieves an array of unique user IDs from the `user` column.
# 
# 3. **Calculate Song Popularity**:
#    - The popularity of each song is calculated as the frequency of the song's appearance in the DataFrame divided by the total number of unique songs.
#    - This calculated popularity is then mapped back to the `song` column of `df` and stored in a new column `song_popularity`.
# 
# The final DataFrame `df` is enhanced with a new `song_popularity` column which provides a relative measure of how frequently each song appears in the dataset, adjusted by the number of unique songs.
# 

# In[6]:


df.isnull().sum()


# In[7]:


unique_names_song = df.song.unique()
unique_names_user = df.user.unique()
print(unique_names_song.shape, unique_names_user.shape)


# In[8]:


df.isnull().values.any()
song_popularity = df['song'].value_counts() / len(unique_names_song)
df['song_popularity'] = df['song'].map(song_popularity)
print(df)


# ## Building Interaction Matrix and Enriched Data
# 
# ### 1. **Initialize Interaction Matrix**:
#    - An interaction matrix is created with dimensions corresponding to the unique count of users and songs, initialized with zeros.
# 
# ### 2. **Map Users and Songs to Matrix Indices**:
#    - Dictionaries `user_indices` and `song_indices` are created to map user and song identifiers to matrix indices for easy access.
# 
# ### 3. **Populate Interaction Matrix**:
#    - Iterate through the DataFrame `df`, using mapped indices to fill the matrix with the logarithm of song popularity incremented by one, to factor in popularity dynamics in interactions.
# 
# ### 4. **Map Song Features**:
#    - Extract and map song-related features like release date, popularity, danceability, and others into dictionaries from `df`, indexed by song ID.
# 
# ### 5. **Prepare Data for Detailed Interaction DataFrame**:
#    - Arrays are prepared for user IDs, song IDs, song features, and interaction values by iterating over the interaction matrix and mapping features for each song-user pair.
# 
# ### 6. **Construct Feature-Rich DataFrame**:
#    - A new DataFrame `interaction_df` is created to encapsulate user IDs, song IDs, their interactions, and all the additional song features such as release, popularity, and more.
# 
# This process effectively creates a structured and detailed view of user-song interactions, which is essential for tasks like recommendation systems or user behavior analysis based on music preferences.
# 

# In[9]:


interaction_matrix = np.zeros((df['user'].nunique(), len(unique_names_song)))

# Map users and songs to matrix indices
user_indices = {user: idx for idx, user in enumerate(df['user'].unique())}
song_indices = {song: idx for idx, song in enumerate(unique_names_song)}

for index, row in df.iterrows():
    user_idx = user_indices[row['user']]
    song_idx = song_indices[row['song']]
    interaction_matrix[user_idx, song_idx] = np.log(row['song_popularity'] + 1)

# Prepare dictionaries to map song IDs to their features
song_features = {
    'release': df.set_index('song')['release'].to_dict(),
    'popularity': df.set_index('song')['popularity'].to_dict(),
    'danceability': df.set_index('song')['danceability'].to_dict(),
    'energy': df.set_index('song')['energy'].to_dict(),
    'key': df.set_index('song')['key'].to_dict(),
    'mode': df.set_index('song')['mode'].to_dict(),
    'valence': df.set_index('song')['valence'].to_dict(),
    'tempo': df.set_index('song')['tempo'].to_dict(),
    'time_of_day': df.set_index('song')['time_of_day'].to_dict(),
}

# Create lists for DataFrame including additional features
user_ids, song_ids, releases, popularities, danceabilities, energies, keys, modes, valences, tempos, interactions, time_of_days = [], [], [], [], [], [], [], [], [], [], [], []
for user in user_indices:
    for song in song_indices:
        user_ids.append(user_indices[user])
        song_ids.append(song_indices[song])
        interactions.append(interaction_matrix[user_indices[user], song_indices[song]])
        # Map each song to its additional features
        releases.append(song_features['release'][song])
        popularities.append(song_features['popularity'][song])
        danceabilities.append(song_features['danceability'][song])
        energies.append(song_features['energy'][song])
        keys.append(song_features['key'][song])
        modes.append(song_features['mode'][song])
        valences.append(song_features['valence'][song])
        tempos.append(song_features['tempo'][song])
        time_of_days.append(song_features['time_of_day'][song])

# Create the interaction DataFrame
interaction_df = pd.DataFrame({
    'user_id': user_ids,
    'song_id': song_ids,
    'release': releases,
    'popularity': popularities,
    'danceability': danceabilities,
    'energy': energies,
    'key': keys,
    'mode': modes,
    'valence': valences,
    'tempo': tempos,
    'time_of_day': time_of_days,
    'interaction': interactions
})

print(interaction_df)


# ## Data Encoding and Basic Statistics
# 
# ### 1. **Encode User and Song Identifiers**:
#    - `LabelEncoder` is used to transform non-numeric user and song identifiers into numeric representations. The transformed identifiers are stored in new columns `user_id` and `song_id` in the DataFrame `df`.
# 
# ### 2. **Calculate Unique Counts**:
#    - Calculate the number of unique users (`N`) and the number of unique songs (`M`) from the newly encoded `user_id` and `song_id` columns.
# 
# This process provides a foundational step in preparing the dataset for more complex analytical tasks, such as modeling user-song interactions in recommendation systems.
# 

# In[10]:


user_encoder = LabelEncoder()
song_encoder = LabelEncoder()
df['user_id'] = user_encoder.fit_transform(df['user'])
df['song_id'] = song_encoder.fit_transform(df['song'])

N = df.user_id.nunique()  # Number of users
M = df.song_id.nunique()  # Number of songs

print(N, M)
print(df.shape, interaction_df.shape)


# ## Data Splitting and Exploration
# 
# ### 1. **Split Data into Training and Testing Sets**:
#    - The `interaction_df` DataFrame is split into training (`df_train`) and testing sets (`df_test`) using a 20% test size allocation and a random seed for reproducibility.
# 
# ### 2. **Evaluate Unique Interaction Values**:
#    - Determine the number of unique interaction values within the `df_train` using `interaction.nunique()` to understand the diversity of user-song interactions.
# 
# ### 3. **Isolate Continuous Features**:
#    - Extract continuous features (columns from the third to the second-last) from both `df_train` and `df_test` into `continuous_data_train` and `continuous_data_test` respectively.
#    - Print shapes of the continuous datasets and `df_train` to understand their dimensions.
# 
# ### 4. **Check for Missing Values**:
#    - Calculate and display the total count of missing values per column in both `df_train` and `df_test` using `isnull().sum()` to assess data cleanliness and readiness for further processing.

# In[11]:


df_train, df_test = train_test_split(interaction_df, test_size=0.2, random_state=42)
print(df_train)


# In[12]:


df_train.interaction.nunique()


# In[13]:


continuous_data_train = df_train.iloc[:, 2:-1]
continuous_data_test = df_test.iloc[:, 2:-1]
print(continuous_data_train.shape, continuous_data_test.shape, df_train.shape)
print(continuous_data_train)


# In[14]:


df_test.isnull().sum()


# In[15]:


df_train.isnull().sum()


# ## Model Loading and Inspection
# 
# ### 1. **Load Pre-trained Model**:
#    - The Keras model is loaded from a specified path (`model_path`), where it was previously saved as `model.h5`.
# 
# ### 2. **Display Model Architecture**:
#    - Use `model.summary()` to print the structure of the model. This includes details of all layers, their types, outputs, and the number of parameters both trainable and non-trainable.

# In[16]:


model_path = '../helpers/model.h5'
model = keras.models.load_model(model_path)
model.summary()


# ## Predictive Model Input Preparation and Execution
# 
# ### 1. **Define Continuous Features**:
#    - Specify the list of continuous features such as `release`, `popularity`, `danceability`, `energy`, `key`, `mode`, `valence`, `tempo`, and `time_of_day`, which are critical for the model’s input.
# 
# ### 2. **Select Specific User**:
#    - A specific user (`specific_user_id`) is selected from the test data to focus the predictions on.
# 
# ### 3. **Prepare User Input Array**:
#    - Create an array `user_input` where the selected user’s ID is repeated for each song, ensuring each song is paired with the user for prediction purposes.
# 
# ### 4. **Prepare Song Input Array**:
#    - Generate `song_input` as an array of indices representing all unique songs (`M` is the total count of unique songs).
# 
# ### 5. **Map Song IDs to DataFrame Indices**:
#    - Construct a dictionary `song_id_to_index` that maps each song ID to its corresponding index in `interaction_df` for efficient data retrieval.
# 
# ### 6. **Prepare Continuous Data for Model Input**:
#    - Use the mapping from song IDs to fetch and structure the continuous data corresponding to all songs, ensuring the data aligns correctly with each song ID in the input array.
# 
# ### 7. **Execute Predictions**:
#    - Make predictions for all songs for the selected user using the prepared inputs (`user_input`, `song_input`, and `continuous_data_input`). 
# 
# ### 8. **Adjust Predictions**:
#    - Normalize predicted interactions by adding the mean interaction value (`mu`) from the training set to each prediction, compensating for any baseline shifts in interaction levels.

# In[17]:


user_ids_test = df_test.user_id.values
song_ids_test = df_test.song_id.values
continuous_features = ["release", "popularity", "danceability", "energy", "key", "mode", "valence", "tempo",
                       "time_of_day"]

# Select a specific user for the prediction
specific_user_id = user_ids_test[0]

# Prepare input data for the model
M = interaction_df['song_id'].nunique()  # Total number of unique songs
user_input = np.array([specific_user_id] * M)  # Repeat the user ID for each song
song_input = np.array(range(M))  # Array of all unique song IDs

# Map song IDs to indices in df
song_id_to_index = {id: idx for idx, id in enumerate(interaction_df['song_id'].unique())}

# Prepare continuous data for all songs
continuous_data_input = np.array(
    [interaction_df.loc[song_id_to_index[song_id], continuous_features] for song_id in song_input])

# Make predictions for this user with all songs, including the continuous data
predicted_interactions = model.predict([user_input, song_input, continuous_data_input])

mu = df_train.interaction.mean()  # Mean interaction value for normalization
predicted_interactions = predicted_interactions.flatten() + mu
print(predicted_interactions)


# ## Extracting Top Song Recommendations
# 
# ### 1. **Determine Number of Recommendations**:
#    - Define `N` as 10 to specify the number of top recommendations to be retrieved.
# 
# ### 2. **Identify Top Recommendations**:
#    - Use `np.argsort()` on the `predicted_interactions` array to get indices of songs sorted by predicted interaction strength.
#    - Reverse the order (`[::-1]`) to start with the highest values and select the top `N` indices (`top_n_indices`).
# 
# ### 3. **Map Indices to Original Song IDs**:
#    - Convert the top indices (`top_n_indices`) back to original song IDs using the `song_encoder.inverse_transform()` function, yielding `top_n_song_ids`.

# In[23]:


# Determine the number of top recommendations top 10
N = 10
top_n_indices = np.argsort(predicted_interactions)[::-1][:N]

# Convert the indices to original song IDs
top_n_song_ids = song_encoder.inverse_transform(top_n_indices)

print(f"Top {N} recommended song IDs for user {specific_user_id} are:", top_n_song_ids)
print(user_input.shape)
print(song_input.shape)
print(continuous_data_input.shape)
print(df[df.user_id == specific_user_id].song.unique())


# ## Displaying Detailed Song Information for Recommendations
# 
# ### 1. **Load Song Metadata**:
#    - Import song metadata from `id_information.csv` using `pd.read_csv()`, specifying a tab (`'\t'`) as the delimiter, into the DataFrame `id_information`.
# 
# ### 2. **Filter Relevant Song Details**:
#    - Filter `id_information` to include only the entries corresponding to the `top_n_song_ids`, which are the IDs of the top recommended songs. This is accomplished using the `isin()` method, ensuring that only relevant song information is considered.
# 
# ### 3. **Display Top Recommended Songs**:
#    - Print details of the top recommended songs specifically tailored for the user (`specific_user_id`). The details displayed include the artist, song title, and album name from the `recommended_songs` DataFrame.
#    - This step highlights the song information, providing a more meaningful context to the recommendations, such as knowing the artist and album for each recommended song.

# In[19]:


# Load song information
id_information = pd.read_csv('../data/id_information.csv', sep='\t')

# Filter id_information to only include the top N recommended song IDs
recommended_songs = id_information[id_information['id'].isin(top_n_song_ids)]

# Print the details of the top N recommended songs
print(f"Top {N} recommended songs for user {specific_user_id} are:")
print(recommended_songs[['artist', 'song', 'album_name']])


# ## Analyzing and Displaying User's Recent Song History
# 
# ### 1. **Filter Songs for Specific User**:
#    - Extract rows from the DataFrame `df` where the `user_id` matches the specific user (`specific_user_id`). This subset contains all the songs interacted with by this particular user.
# 
# ### 2. **Sort Songs by Recent Play**:
#    - Sort the filtered data (`user_songs`) by the `timestamp` column in descending order to prioritize the most recent interactions. This sorted DataFrame is stored as `user_songs_sorted`.
# 
# ### 3. **Identify Last 5 Played Songs**:
#    - Retrieve the IDs of the last five songs played by this user from the top of the sorted DataFrame, ensuring these are the most recent songs interacted with.
# 
# ### 4. **Filter Song Metadata**:
#    - Use the song IDs (`last_5_song_ids`) to filter `id_information` to include only metadata for these last five songs. This step ensures that the information displayed pertains only to the most recent song interactions.
# 
# ### 5. **Display Song Information**:
#    - Print details about these last five songs, including the artist, song title, and album name, providing a comprehensive view of the user’s most recent music preferences.

# In[24]:


# Filter the DataFrame for the specific user
user_songs = df[df.user_id == specific_user_id]

# Sort the data by the timestamp column in descending order
user_songs_sorted = user_songs.sort_values(by='timestamp', ascending=False)

last_5_song_ids = user_songs_sorted['song'].head(5).values
print(last_5_song_ids)

last_5_songs_info = id_information[id_information['id'].isin(last_5_song_ids)]
print("Information about the last 5 songs played:")
print(last_5_songs_info[['artist', 'song', 'album_name']])


# ## Similarity Calculation for Recommended Songs
# 
# ### 1. **Function for Similarity Calculation**:
#    - The `calculate_similarity` function computes the cosine similarity between a target song's feature vector and a set of song features. It returns the similarity scores from a 2D array to a 1D array for easier manipulation.
# 
# ### 2. **Initialize Variables**:
#    - Set `maxi` to -1 to track the highest similarity score found.
#    - Set `best_idx` to 0 to store the index of the song with the highest similarity.
# 
# ### 3. **Iterate Over Top Recommended Songs**:
#    - Loop through the first 10 recommended song IDs (`top_n_song_ids`).
#    - For each recommended song:
#      - Extract the feature vector from `interaction_df` using the song's index derived from `song_indices`.
# 
# ### 4. **Fetch Features of Last 5 Played Songs**:
#    - Construct an array of feature vectors for the last five played songs, again using indices from `interaction_df`.
# 
# ### 5. **Calculate Similarity for Each Recommended Song**:
#    - For each recommended song, compute its similarity to each of the last five played songs using the previously defined `calculate_similarity` function.
#    - Sum the similarity scores to get an overall similarity measure for the recommended song against all last played songs.
# 
# ### 6. **Identify Song with Highest Similarity**:
#    - Track and update the maximum similarity score (`maxi`) and the corresponding index (`best_idx`) if the current song's summed similarity score exceeds the previous maximum.
# 
# This method effectively determines which of the top recommended songs are most similar to the user's recent listening habits.
# 

# In[25]:


def calculate_similarity(target_song_features, songs_features):
    similarity = cosine_similarity([target_song_features], songs_features)
    return similarity[0]  # similarity[0] because the result is in a 2D array

maxi = -1
best_idx = 0


# In[26]:


for i in range(10):
    recommended_song_id = top_n_song_ids[i]
    # Get the feature vector for the recommended song
    recommended_song_features = interaction_df.loc[interaction_df.song_id[song_indices[recommended_song_id]]]

    # Get the feature vectors for the last 5 played songs
    last_5_songs_features = np.array(
        [interaction_df.loc[interaction_df.song_id[song_indices[song_id]]] for song_id in last_5_song_ids])

    # Calculate similarity
    similarities = calculate_similarity(np.array(recommended_song_features).reshape(1, -1)[0], last_5_songs_features)
    print(np.sum(similarities), end="\n\n\n")

    if np.sum(similarities) > maxi:
        maxi = np.sum(similarities)
        best_idx = i


# ## Generating and Comparing Predicted Recommendations with Actual Preferences
# 
# ### 1. **Define Function to Retrieve Actual Relevant Songs**:
#    - The `get_relevant_songs` function is designed to fetch the unique song IDs that a specific user has interacted with from the DataFrame `df`. This establishes a baseline of songs that are known to be relevant to the user.
# 
# ### 2. **Initialize Prediction and Actual Dictionaries**:
#    - Two dictionaries, `predictions` and `actual`, are initialized to store the predicted top song IDs and actual relevant song IDs for each user, respectively.
# 
# ### 3. **Iterate Over a Subset of Users**:
#    - Loop through each user ID in the test set (limited to the first 10 users for demonstration). This looping facilitates the prediction and validation process for multiple users in a manageable subset.
# 
# ### 4. **Generate Predictions for Each User**:
#    - For each user:
#      - Create an input array (`user_input`) that repeats the user ID for each song, corresponding to the total number of unique songs (`M`).
#      - Prepare `continuous_data_input` by collecting the continuous feature data for all songs relevant to the current user input.
#      - Predict interaction scores using the model for all songs with the prepared inputs. Flatten the result to simplify handling.
#      - Sort the predicted scores in descending order and extract the indices of the top `N` scores.
#      - Use `song_encoder.inverse_transform` to convert these indices back into original song IDs (`top_n_song_ids`).
# 
# ### 5. **Store Predictions and Actual Song IDs**:
#    - Store the top `N` predicted song IDs for each user in the `predictions` dictionary.
#    - Fetch and store the actual relevant songs for the user using `get_relevant_songs` and store them in the `actual` dictionary.
# 
# ### 6. **Output Progress**:
#    - Print the total number of users being processed and the current progress after each user's data is processed to monitor the computation and ensure it is proceeding correctly.

# In[29]:


def get_relevant_songs(user_id, df):
    return df[df.user_id == user_id]['song'].unique()

predictions = {}
actual = {}

i = 0
# Get predictions and actual relevant songs for each user
for user_id in user_ids_test[:]:  # Limit to first 10 users for the example
    # Predict top N songs
    user_input = np.array([user_id] * M)
    continuous_data_input = np.array(
        [interaction_df.loc[song_id_to_index[song_id], continuous_features] for song_id in song_input])
    predicted_interactions = model.predict([user_input, song_input, continuous_data_input]).flatten()
    top_n_indices = np.argsort(predicted_interactions)[::-1][:N]
    top_n_song_ids = song_encoder.inverse_transform(top_n_indices)

    # Store the top N song IDs for each user
    predictions[user_id] = top_n_song_ids.tolist()

    # Get actual relevant songs for the user
    actual_relevant_songs = get_relevant_songs(user_id, df)
    actual[user_id] = actual_relevant_songs.tolist()
    i += 1
    print(len(user_ids_test), i)


# ## Recommendation System Evaluation Metrics
# 
# ### Metric Calculations
# 
# #### 1. **Precision at K**:
#    - Measures the proportion of recommended items in the top-K set that are relevant.
# 
# #### 2. **Recall at K**:
#    - Assesses how many relevant items are found in the top-K recommendations.
# 
# #### 3. **Mean Average Precision at K (MAP@K)**:
#    - Computes the mean of the average precision scores for each user, considering only the top-K recommendations.
# 
# #### 4. **Mean Reciprocal Rank (MRR)**:
#    - Calculates the average of the reciprocal of the rank of the first relevant item among the recommendations.
# 
# #### 5. **Normalized Discounted Cumulative Gain at K (NDCG@K)**:
#    - Evaluates the gain of a recommendation based on its position in the result list, giving higher importance to hits at top ranks.
# 
# ### Functions Defined
# 
# - **`precision_at_k`**: Compares the top-K predicted items to the actual relevant items for each user to calculate precision.
# - **`recall_at_k`**: Identifies how many of the relevant items appear in the top-K predictions for each user.
# - **`mean_avg_precision_at_k`** and **`mean_average_precision_at_k`**: Both calculate the average precision at K for predictions against actual data.
# - **`mean_reciprocal_rank`**: Computes the average reciprocal rank where the rank is the position of the first relevant recommendation.
# - **`dcg_at_k`**: Computes the Discounted Cumulative Gain at K, a measure of ranking quality.
# - **`ndcg_at_k`**: Normalizes the DCG at K by the ideal or perfect DCG at K, providing a measure of the model's performance relative to the best possible scenario.
# 

# In[ ]:


def precision_at_k(actual, predicted, k):
    precision_scores = []
    for user_id in actual:
        # Initialize true positives count
        true_positives = 0
        # Check if user exists in predictions
        if user_id in predicted and len(predicted[user_id]) >= k:
            # Count the number of relevant items in the top k predictions
            true_positives = len(set(predicted[user_id][:k]) & set(actual[user_id]))
        # Calculate precision for this user
        precision = true_positives / float(k)
        precision_scores.append(precision)
    # Return the average precision at k for all users
    return sum(precision_scores) / len(precision_scores)


def recall_at_k(actual, predicted, k):
    recall_scores = []
    for user_id in actual:
        # Initialize true positives count
        true_positives = 0
        # Check if user exists in predictions
        if user_id in predicted:
            # Count the number of relevant items in the top k predictions
            true_positives = len(set(predicted[user_id][:k]) & set(actual[user_id]))
            recall = true_positives / float(len(actual[user_id]))
        else:
            # If no predictions for the user, recall is 0
            recall = 0.0
        recall_scores.append(recall)
    # Return the average recall at k for all users
    return sum(recall_scores) / len(recall_scores)


def avg_precision_at_k(actual, predicted, k=10):
    ap_sum = 0
    for user, true_items in actual.items():
        pred_items = predicted[user][:k]
        hits = 0
        sum_precs = 0
        for i, p in enumerate(pred_items):
            if p in true_items:
                hits += 1
                sum_precs += hits / (i + 1.0)
        ap_sum += sum_precs / min(len(true_items), k)
    return ap_sum / len(actual)


def mean_avg_precision_at_k(actual, predicted, k=10):
    return avg_precision_at_k(actual, predicted, k)


def mean_average_precision_at_k(actual, predicted, k=10):
    AP_sum = 0.0
    for user_id in actual:
        if user_id in predicted:
            pred_items = predicted[user_id][:k]
            hits = 0
            sum_precisions = 0
            for i, p in enumerate(pred_items):
                if p in actual[user_id] and p not in pred_items[:i]:
                    hits += 1
                    sum_precisions += hits / (i + 1.0)
            AP_sum += sum_precisions / min(len(actual[user_id]), k)
    return AP_sum / len(actual)


def mean_reciprocal_rank(actual, predicted):
    MRR_sum = 0.0
    for user_id in actual:
        if user_id in predicted:
            pred_items = predicted[user_id]
            for rank, p in enumerate(pred_items, start=1):
                if p in actual[user_id]:
                    MRR_sum += 1.0 / rank
                    break
    return MRR_sum / len(actual)


def dcg_at_k(relevances, k):
    relevances = np.asfarray(relevances)[:k]
    if relevances.size:
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0


def ndcg_at_k(actual, predicted, k=10):
    NDCG_sum = 0.0
    for user_id in actual:
        if user_id in predicted:
            pred_items = predicted[user_id][:k]
            true_relevances = [1 if item in actual[user_id] else 0 for item in pred_items]
            ideal_relevances = [1] * len(actual[user_id])
            NDCG_sum += dcg_at_k(true_relevances, k) / dcg_at_k(ideal_relevances, k)
    return NDCG_sum / len(actual)

k = 10
precision = precision_at_k(actual, predictions, k)
recall = recall_at_k(actual, predictions, k)
map_k = mean_average_precision_at_k(actual, predictions, k)
mrr = mean_reciprocal_rank(actual, predictions)
ndcg_k = ndcg_at_k(actual, predictions, k)

print(f"Precision@{k}: {precision}")
print(f"Recall@{k}: {recall}")
print(f"MAP@{k}: {map_k}")
print(f"MRR: {mrr}")
print(f"NDCG@{k}: {ndcg_k}")

