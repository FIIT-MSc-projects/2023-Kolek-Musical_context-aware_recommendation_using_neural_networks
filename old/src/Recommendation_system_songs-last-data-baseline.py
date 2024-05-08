#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dot, Add, Flatten, Dense, Activation, Dropout, Concatenate
from keras.optimizers import SGD
import numpy as np


# In[2]:


# Load data
metadata_path = '../data/id_metadata.csv'
listening_history_path = '../data/listening_history.csv'
metadata_df = pd.read_csv(metadata_path, delimiter='\t')
df = pd.read_csv(listening_history_path, delimiter='\t')


print(df)

# In[3]:


# Ensure the 'timestamp' column is datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Find the latest date in the data
latest_date = df['timestamp'].max()

# Calculate the date 14 days (2 weeks) prior to the latest date
two_weeks_ago = latest_date - pd.Timedelta(days=7)

# Filter the DataFrame to include only the last two weeks of data
df = df[df['timestamp'] >= two_weeks_ago]

print(df)

# In[4]:



numeric_cols_df = df.select_dtypes(include=np.number).columns
sscaler = StandardScaler()
# df[numeric_cols_df] = sscaler.fit_transform(df[numeric_cols_df])
print(df)

# In[5]:


print(df.isnull().sum())

# In[6]:


unique_names_song = df.song.unique()
unique_names_user = df.user.unique()

print("unique_names_song:", unique_names_song.shape)
print("unique_names_user:", unique_names_user.shape)

# In[7]:


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


# Create lists for DataFrame including additional features
user_ids, song_ids, releases, popularities, danceabilities, energies, keys, modes, valences, tempos, interactions, time_of_days = [], [], [], [], [], [], [], [], [], [], [], []
for user in user_indices:
    for song in song_indices:
        user_ids.append(user_indices[user])
        song_ids.append(song_indices[song])
        interactions.append(interaction_matrix[user_indices[user], song_indices[song]])


# Create the interaction DataFrame
interaction_df = pd.DataFrame({
    'user_id': user_ids,
    'song_id': song_ids,
    'interaction': interactions
})

print(interaction_df)

# In[8]:


user_encoder = LabelEncoder()
song_encoder = LabelEncoder()
df['user_id'] = user_encoder.fit_transform(df['user'])
df['song_id'] = song_encoder.fit_transform(df['song'])

N = df.user_id.nunique()  # Number of users
M = df.song_id.nunique()  # Number of songs

print("Number of users: ", N)
print("Number of songs: ", M)
df.shape, interaction_df.shape

# In[9]:


df_train, df_test = train_test_split(interaction_df, test_size=0.2, random_state=42)
print(df_train, df_test)

# In[10]:


print("unique interactions: ", df_train.interaction.nunique())

K = 15  # define the size of embeddings, capture the relations in data (10-50)

mu = df_train.interaction.mean()  # Mean interaction for normalization
epochs = 1
model_path = '../helpers/model.h5'
if not os.path.exists(model_path):
    u = Input(shape=(1,))
    s = Input(shape=(1,))
    u_embedding = Embedding(N, K)(u)  # (N, 1, K)
    s_embedding = Embedding(M, K)(s)  # (N, 1, K)

    ##### main branch
    u_bias = Embedding(N, 1)(u)  # (N, 1, 1)
    s_bias = Embedding(M, 1)(s)  # (N, 1, 1)
    x = Dot(axes=2)([u_embedding, s_embedding])  # (N, 1, 1)
    x = Add()([x, u_bias, s_bias])
    x = Flatten()(x)  # (N, 1)

    ##### model definition
    model = Model(inputs=[u, s], outputs=x)
    model.compile(
        loss='mse',
        optimizer=SGD(learning_rate=0.08, momentum=0.9),
        metrics=['mse'],
    )

    # Now, train the model
    r = model.fit(
        x=[df_train.user_id.values, df_train.song_id.values],
        y=df_train.interaction.values - mu,
        epochs=epochs,
        batch_size=128,
        validation_data=(
            [df_test.user_id.values, df_test.song_id.values],
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
    print("Model trained and saved as 'model.h5'.")
else:
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
    print("No training session to plot, model was loaded from file.")

# In[14]:


import numpy as np

user_ids_test = df_test.user_id.values
song_ids_test = df_test.song_id.values

# Select a specific user for the prediction
specific_user_id = user_ids_test[0]  # Example: taking the first user in the test set

# Prepare input data for the model
M = interaction_df['song_id'].nunique()  # Total number of unique songs
user_input = np.array([specific_user_id] * M)  # Repeat the user ID for each song
song_input = np.array(range(M))  # Array of all unique song IDs

# Map song IDs to indices in df
song_id_to_index = {id: idx for idx, id in enumerate(interaction_df['song_id'].unique())}

# Make predictions for this user with all songs
predicted_interactions = model.predict([user_input, song_input])

# Convert predictions back to the original scale, if needed
mu = df_train.interaction.mean()  # Mean interaction value for normalization (if used during training)
predicted_interactions = predicted_interactions.flatten() + mu

# Determine the number of top recommendations, e.g., top 10
N = 10
top_n_indices = np.argsort(predicted_interactions)[::-1][:N]

# Convert the indices to original song IDs
top_n_song_ids = song_encoder.inverse_transform(top_n_indices)

# Output the recommended songs
print(f"Top {N} recommended song IDs for user {specific_user_id} are:", top_n_song_ids)

# In[15]:


print("user_input.shape: ", user_input.shape)

# In[16]:


id_information = pd.read_csv('../data/id_information.csv', sep='\t')
print(id_information)

# In[17]:


unique_songs = df[df.user_id == specific_user_id].song.unique()

# Print a user-friendly message
print(f"The user with ID {specific_user_id} has interacted with {len(unique_songs)} unique songs.")

# Print a subset of the songs for brevity
print(unique_songs)

# In[18]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Example song features DataFrame
# Assume df_features is a DataFrame with songs as rows and features as columns

# Calculate similarity
def calculate_similarity(target_song_features, songs_features):
    similarity = cosine_similarity([target_song_features], songs_features)
    return similarity[0]  # similarity[0] because the result is in a 2D array


maxi = -1
best_idx = 0
last_5_song_ids = list()

# REWORKa
last_5_song_ids = ['aPPVq97XeQv8mqsU', 'SSA3WorrB4G8ww60', 'lJKIbZNzpS6IsNgh', 'd8QDyWffh9zwJ4Gs', 'oLHuLrmZyV1oEfdu']
# print(last_5_song_ids)

# In[19]:


recommended_song_id = top_n_song_ids[best_idx]
id_information[id_information.id == recommended_song_id]

# In[20]:


last_5_songs_info = id_information[id_information['id'].isin(last_5_song_ids)]

# Display the information of the last 5 songs
print(last_5_songs_info)

# In[29]:


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
predicted_interactions = model.predict([user_input, song_input])
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


def precision_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result


def recall_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result


def mean_average_precision(actual, predicted, k):
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)


def mean_reciprocal_rank(relevant_results):
    for index, score in enumerate(relevant_results):
        if score:
            return 1.0 / (index + 1)
    return 0


def ndcg_at_k(actual, predicted, k):
    def dcg_at_k(scores, k):
        return sum([score / np.log2(idx + 2) for idx, score in enumerate(scores[:k])])

    actual_scores = [1 if song in actual else 0 for song in predicted[:k]]
    best_scores = sorted(actual_scores, reverse=True)
    ideal_dcg = dcg_at_k(best_scores, k)
    actual_dcg = dcg_at_k(actual_scores, k)
    if not ideal_dcg:
        return 0.0
    return actual_dcg / ideal_dcg

