import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dot, Add, Flatten, Dense, Activation, Dropout, Concatenate
from keras.optimizers import SGD
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)        # Set the display width to 1000 characters
pd.set_option('display.max_rows', 10)       # Display up to 10 rows


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

metadata_path = '../data/id_metadata.csv'
listening_history_path = '../data/listening_history.csv'
metadata_df = pd.read_csv(metadata_path, delimiter='\t')
df = pd.read_csv(listening_history_path, delimiter='\t')

df['timestamp'] = pd.to_datetime(df['timestamp'])
metadata_df.rename(columns={'id': 'song'}, inplace=True)
df['time_of_day'] = df['timestamp'].apply(classify_time_of_day)
print("Loaded listening history:")
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

latest_date = df['timestamp'].max()
one_week_ago = latest_date - pd.Timedelta(days=7)
df = df[df['timestamp'] >= one_week_ago]
print('\n\n')
print("Filtered last week:")
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

df = pd.merge(df, metadata_df[
    ['song', 'release', 'popularity', 'danceability', 'energy', 'key', 'mode', 'valence', 'tempo']], on='song')

numeric_cols_df = df.select_dtypes(include=np.number).columns
sscaler = StandardScaler()
df[numeric_cols_df] = sscaler.fit_transform(df[numeric_cols_df])
print('\n\n')
print("Merged with features and scaled:")
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
print('\n\n')
print("Null values in df:")
print(df.isnull().sum())

unique_names_song = df.song.unique()
unique_names_user = df.user.unique()
print('\n\n')
print("unique_names_song.shape, unique_names_user.shape: ")
print(unique_names_song.shape, unique_names_user.shape)
song_popularity = df['song'].value_counts() / len(unique_names_song)
df['song_popularity'] = df['song'].map(song_popularity)
print('\n\n')
print("df with popularity:")
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
print('\n\n')
print("Interaction matrix:")
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
user_encoder = LabelEncoder()
song_encoder = LabelEncoder()
df['user_id'] = user_encoder.fit_transform(df['user'])
df['song_id'] = song_encoder.fit_transform(df['song'])

N = df.user_id.nunique()  # Number of users
M = df.song_id.nunique()  # Number of songs
print('\n\n')
print('Number of users, Number of songs:')
print(N, M)

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

df_train, df_test = train_test_split(interaction_df, test_size=0.2, random_state=42)
print('\n\n')
print("df_train:")
print(df_train)
print('\n\n')
print("df_train.interaction.nunique():")
print(df_train.interaction.nunique())
continuous_data_train = df_train.iloc[:, 2:-1]
continuous_data_test = df_test.iloc[:, 2:-1]
print('\n\n')
print('continuous_data_train.shape, continuous_data_test.shape, df_train.shape:')
print(continuous_data_train.shape, continuous_data_test.shape, df_train.shape)
print('\n\n')
print('continuous_data_train:')
print(continuous_data_train)
print('\n\n')
print('df_test.isnull().sum():')
print(df_test.isnull().sum())
print('\n\n')
print('df_train.isnull().sum():')
print(df_train.isnull().sum())

# ## Deep Learning Model Construction and Training for Recommendation System
#
# ### Model Design:
#
# #### 1. **Embedding Layers**:
#    - **User and Song Embeddings**: Separate embedding layers for users (`u_embedding`) and songs (`s_embedding`), each with a dimensionality of `K=15`, are created to capture the latent factors from user and song IDs.
#    - **Bias Embeddings**: Additional embeddings (`u_bias` and `s_bias`) for users and songs respectively to model individual propensities beyond the interaction effects captured by the dot product.
#
# #### 2. **Main Interaction Branch**:
#    - **Dot Product**: Computes the dot product between user and song embeddings to capture the interaction strength.
#    - **Addition of Bias**: Adds user and song biases to the interaction terms.
#    - **Flatten**: Flattens the output for further processing or merging with other model branches.
#
# #### 3. **Continuous Data Branch**:
#    - An additional input layer (`continuous_input`) takes in continuous feature data related to interactions (e.g., time of day, song features).
#    - **Concatenation with Embeddings**: Concatenates flattened user and song embeddings with the continuous data.
#    - **Fully Connected Layers**: Multiple dense layers with dropout for regularization and ELU activation to model nonlinearities.
#
# #### 4. **Combining Branches**:
#    - The outputs of the main interaction branch and the continuous data branch are combined using addition to integrate the learned embeddings with the contextual data.
#
# ### Model Compilation and Training:
#
# - **Loss Function**: Mean Squared Error (MSE), suitable for regression tasks like predicting interaction strengths.
# - **Optimizer**: Stochastic Gradient Descent (SGD) with learning rate adjustments and momentum to converge efficiently.
# - **Metrics**: Tracks MSE during training for both training and validation data.
#
# ### Training Process:
#
# - The model is trained using the user IDs, song IDs, and continuous feature data from `df_train`.
# - Targets are the interaction values adjusted by subtracting the mean interaction (`mu`) to normalize the data.
# - Validation is performed using similar data from `df_test`.
# - Training and validation losses, as well as MSE, are plotted for each epoch to monitor overfitting and convergence.
#
# ### Saving and Loading the Model:
#
# - If the model file (`model.h5`) does not exist, it is saved after training. If it exists, it is loaded directly, saving time and resources if retraining is not necessary.
#
# ### Usage:
#
# - This model can be used in a recommendation system to predict the strength of interaction between users and songs based on both their intrinsic characteristics captured through embeddings and contextual/continuous data.
#
# This structured approach ensures that the model not only learns from the intrinsic properties of users and songs but also incorporates additional contextual information, potentially enhancing the accuracy and personalization of the recommendations.

K = 15 # define the size of embeddings, capture the relations in data (10-50)

mu = df_train.interaction.mean()  # Mean interaction for normalization
epochs = 50
model_path = '../helpers/model.h5'
if not os.path.exists(model_path):
    u = Input(shape=(1,))
    s = Input(shape=(1,))
    u_embedding = Embedding(N, K)(u) # (N, 1, K)
    s_embedding = Embedding(M, K)(s) # (N, 1, K)


    # main branch
    u_bias = Embedding(N, 1)(u) # (N, 1, 1)
    s_bias = Embedding(M, 1)(s) # (N, 1, 1)
    x = Dot(axes=2)([u_embedding, s_embedding]) # (N, 1, 1)
    x = Add()([x, u_bias, s_bias])
    x = Flatten()(x) # (N, 1)

    # CONTINUOUS BRANCH
    continuous_input = Input(shape=(continuous_data_train.shape[1],))


    # side branch
    u_embedding = Flatten()(u_embedding) # (N, K)
    s_embedding = Flatten()(s_embedding) # (N, K)
    y = Concatenate()([u_embedding, s_embedding, continuous_input]) # (N, 2K)
    y = Dense(512)(y)
    y = Activation('elu')(y)
    y = Dropout(0.3)(y)
    y = Dense(512)(y)
    y = Activation('elu')(y)
    y = Dropout(0.3)(y)
    y = Dense(512)(y)
    y = Activation('elu')(y)
    y = Dropout(0.3)(y)
    y = Dense(1)(y)


    # merge
    x = Add()([x, y])

    model = Model(inputs=[u, s, continuous_input], outputs=x)
    model.compile(
      loss='mse',
      optimizer=SGD(learning_rate=0.08, momentum=0.9),
      metrics=['mse'],
    )

    r = model.fit(
      x=[df_train.user_id.values, df_train.song_id.values, continuous_data_train.values],
      y=df_train.interaction.values - mu,
      epochs=epochs,
      batch_size=128,
      validation_data=(
        [df_test.user_id.values, df_test.song_id.values, continuous_data_test.values],
        df_test.interaction.values - mu
      )
    )

    plt.plot(r.history['loss'], label="train loss")
    plt.plot(r.history['val_loss'], label="test loss")
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.clf()

    plt.plot(r.history['mse'], label="train mse")
    plt.plot(r.history['val_mse'], label="test mse")
    plt.legend()
    plt.savefig('mse_plot.png')
    plt.clf()
    model.save(model_path)
    print('\n\n')
    print("Model trained and saved as 'model.h5'.")
else:
    print('\n\n')
    print("Model 'model.h5' already exists. Loading model...")
    model = load_model(model_path)


# In[13]:


model.summary()
# Plot training and validation loss and MSE
if 'r' in locals():  # Check if training results are available
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(r.history['loss'], label="train loss")
    plt.plot(r.history['val_loss'], label="test loss")
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(r.history['mse'], label="train mse")
    plt.plot(r.history['val_mse'], label="test mse")
    plt.title('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print('\n\n')
    print("No training session to plot, model was loaded from file.")

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
print('\n\n')
print("predicted_interactions:")
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

# Determine the number of top recommendations top 10
N = 10
top_n_indices = np.argsort(predicted_interactions)[::-1][:N]

# Convert the indices to original song IDs
top_n_song_ids = song_encoder.inverse_transform(top_n_indices)

print('\n\n')
print(f"Top {N} recommended song IDs for user {specific_user_id} are:", top_n_song_ids)
print('\n\n')
print('user_input.shape, song_input.shape, continuous_data_input.shape:')
print(user_input.shape, song_input.shape, continuous_data_input.shape)
print('\n\n')
print('Specified users unique songs:')
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

# Load song information
id_information = pd.read_csv('../data/id_information.csv', sep='\t')

# Filter id_information to only include the top N recommended song IDs
recommended_songs = id_information[id_information['id'].isin(top_n_song_ids)]

# Print the details of the top N recommended songs
print('\n\n')
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

# Filter the DataFrame for the specific user
user_songs = df[df.user_id == specific_user_id]

# Sort the data by the timestamp column in descending order
user_songs_sorted = user_songs.sort_values(by='timestamp', ascending=False)

last_5_song_ids = user_songs_sorted['song'].head(5).values
print('\n\n')
print(last_5_song_ids)

last_5_songs_info = id_information[id_information['id'].isin(last_5_song_ids)]
print('\n\n')
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

def calculate_similarity(target_song_features, songs_features):
    similarity = cosine_similarity([target_song_features], songs_features)
    return similarity[0]  # similarity[0] because the result is in a 2D array

maxi = -1
best_idx = 0

for i in range(10):
    recommended_song_id = top_n_song_ids[i]
    # Get the feature vector for the recommended song
    recommended_song_features = interaction_df.loc[interaction_df.song_id[song_indices[recommended_song_id]]]

    # Get the feature vectors for the last 5 played songs
    last_5_songs_features = np.array(
        [interaction_df.loc[interaction_df.song_id[song_indices[song_id]]] for song_id in last_5_song_ids])

    # Calculate similarity
    similarities = calculate_similarity(np.array(recommended_song_features).reshape(1, -1)[0], last_5_songs_features)

    if np.sum(similarities) > maxi:
        maxi = np.sum(similarities)
        best_idx = i

print('\n\n')
print('Maximum similarity, most similar song id:')
print(maxi, best_idx)


# In[19]:


recommended_song_id = top_n_song_ids[best_idx]
recommended_song = id_information[id_information.id == recommended_song_id]
print('\n\n')
print("The next played song would be: ")
print(recommended_song[['artist', 'song', 'album_name']])



