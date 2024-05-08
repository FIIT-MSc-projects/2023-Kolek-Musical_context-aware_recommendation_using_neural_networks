import numpy as np
import pandas as pd
import sys
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Open your file for writing
# file = open('output.txt', 'w')
#
# # Save the original stdout so we can revert sys.stdout after we're done with the file
# original_stdout = sys.stdout
#
# # Redirect print statements to the file
# sys.stdout = file
#
# sys.stderr = open('training_errors_warnings.txt', 'w')  # To capture errors and warnings


# Classify time of day
def classify_time_of_day(timestamp):
    hour = timestamp.hour
    if 4 <= hour < 12:
        return 1
    elif 12 <= hour < 20:
        return 2
    else:
        return 3


# Load data
metadata_path = '../data/id_metadata.csv'
listening_history_path = '../data/listening_history.csv'
metadata_df = pd.read_csv(metadata_path, delimiter='\t')
df = pd.read_csv(listening_history_path, delimiter='\t')

df['timestamp'] = pd.to_datetime(df['timestamp'])
metadata_df.rename(columns={'id': 'song'}, inplace=True)
df['time_of_day'] = df['timestamp'].apply(classify_time_of_day)
print(df)

# Find the latest date in the data
latest_date = df['timestamp'].max()

# Calculate the date 14 days (2 weeks) prior to the latest date
two_weeks_ago = latest_date - pd.Timedelta(days=7)

# Filter the DataFrame to include only the last two weeks of data
df = df[df['timestamp'] >= two_weeks_ago]

print(df)

df = pd.merge(df, metadata_df[
    ['song', 'release', 'popularity', 'danceability', 'energy', 'key', 'mode', 'valence', 'tempo']], on='song')

numeric_cols_df = df.select_dtypes(include=np.number).columns
sscaler = StandardScaler()
df[numeric_cols_df] = sscaler.fit_transform(df[numeric_cols_df])

df.head()

df.isnull().sum()

unique_names_song = df.song.unique()
unique_names_user = df.user.unique()
print(unique_names_song.shape, unique_names_user.shape)

df.isnull().values.any()  # Calculate song popularity
song_popularity = df['song'].value_counts() / len(unique_names_song)
df['song_popularity'] = df['song'].map(song_popularity)

# Create an empty interaction matrix
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

interaction_df.head()

user_encoder = LabelEncoder()
song_encoder = LabelEncoder()
df['user_id'] = user_encoder.fit_transform(df['user'])
df['song_id'] = song_encoder.fit_transform(df['song'])

N = df.user_id.nunique()  # Number of users
M = df.song_id.nunique()  # Number of songs

print(N, M)
print(df.shape, interaction_df.shape)

df_train, df_test = train_test_split(interaction_df, test_size=0.2, random_state=42)
df_train.head(50)

df_train.tail()

df_train.interaction.nunique()

continuous_data_train = df_train.iloc[:, 2:-1]
continuous_data_test = df_test.iloc[:, 2:-1]
print(continuous_data_train.shape, continuous_data_test.shape, df_train.shape)
continuous_data_train.head()

df_train.isnull().sum()

df_test.isnull().sum()

model_path = '../helpers/model.h5'
model = keras.models.load_model(model_path)

# Show the model architecture
model.summary()


user_ids_test = df_test.user_id.values
song_ids_test = df_test.song_id.values
continuous_features = ["release", "popularity", "danceability", "energy", "key", "mode", "valence", "tempo",
                       "time_of_day"]

# Select a specific user for the prediction
specific_user_id = user_ids_test[0]  # Example: taking the first user in the test set

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

mu = df_train.interaction.mean()  # Mean interaction value for normalization (if used during training)
predicted_interactions = predicted_interactions.flatten() + mu

# Determine the number of top recommendations, e.g., top 10
N = 10
top_n_indices = np.argsort(predicted_interactions)[::-1][:N]

# Convert the indices to original song IDs
top_n_song_ids = song_encoder.inverse_transform(top_n_indices)

# Output the recommended songs
print(f"Top {N} recommended song IDs for user {specific_user_id} are:", top_n_song_ids)
print(user_input.shape)
print(song_input.shape)
print(continuous_data_input.shape)

id_information = pd.read_csv('../data/id_information.csv', sep='\t')
id_information.head()

print(df[df.user_id == specific_user_id].song.unique())

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Calculate similarity
def calculate_similarity(target_song_features, songs_features):
    similarity = cosine_similarity([target_song_features], songs_features)
    return similarity[0]  # similarity[0] because the result is in a 2D array


maxi = -1
best_idx = 0

last_5_song_ids = ['OeQ2tiben0HEo3LV', '0HNuxBP54vQhSmyJ', 'RA9JZ0JD6OQJTDoL', '0TanEa2QCnKAedsL', 'QrV9CUo46zazVaWV']


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
print()
print()
print(maxi, best_idx)

recommended_song_id = top_n_song_ids[best_idx]
print(id_information[id_information.id == recommended_song_id])

last_5_songs_info = id_information[id_information['id'].isin(last_5_song_ids)]

# Display the information of the last 5 songs
print(last_5_songs_info)


from sklearn.metrics import precision_score, recall_score, f1_score


def get_relevant_songs(user_id, df):
    """
    Get a list of relevant songs for a given user.
    This function needs to be adapted based on how relevance is defined in your dataset.
    """
    # Example: get songs that the user has interacted with
    return df[df.user_id == user_id]['song'].unique()


# Select a user and predict top N songs
specific_user_id = user_ids_test[0]
predicted_interactions = model.predict([user_input, song_input, continuous_data_input])
flattened_arr = predicted_interactions.flatten()

# Sort the array in descending order
sorted_indices = np.argsort(flattened_arr)[::-1]
sorted_arr = flattened_arr[sorted_indices]

# Get actual relevant songs for the user
actual_relevant_songs = get_relevant_songs(specific_user_id, df)

# Convert actual relevant songs and top recommended songs to a binary format
actual_binary = [1 if song_encoder.inverse_transform([song_id])[0] in actual_relevant_songs else 0 for song_id in
                 song_ids_test]
predicted_binary = [1 if song_id in sorted_indices[:len(actual_relevant_songs)].tolist() else 0 for song_id in
                    song_ids_test]

# Calculate precision, recall, and F1 score
precision = precision_score(actual_binary, predicted_binary)
recall = recall_score(actual_binary, predicted_binary)
f1 = f1_score(actual_binary, predicted_binary)

print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


# Placeholder for storing predictions for all users
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


# Calculate metrics for all users
# Calculate precision and recall at k
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


