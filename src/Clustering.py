#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import np_utils
import matplotlib.pyplot as plt
import seaborn as sns
import keras
# import plotly.express as px
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


# In[2]:


listening_history = pd.read_csv('../data/listening_history.csv', sep='\t')
id_metadata = pd.read_csv('../data/id_metadata.csv', sep='\t')
id_information = pd.read_csv('../data/id_information.csv', sep='\t')


# In[3]:


listening_history.head()


# In[4]:


id_metadata.head()


# In[5]:


selected_features = id_metadata.copy()
selected_features = selected_features.drop(columns=['spotify_id'])
selected_features.set_index("id", inplace=True)
selected_features


# In[6]:


plt.figure(figsize=(10, 8)) 
ax = sns.heatmap(selected_features.corr(), annot=True)
plt.show()


# In[7]:


selected_features.hist(bins=50, figsize=(20,10))
plt.show()


# In[8]:


columns_to_cluster = ['popularity', 'release', 'danceability', 'energy',
                      'key', 'mode', 'valence', 'tempo', 'duration_ms']


# In[9]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
songs_scaled = scaler.fit_transform(selected_features)


# In[10]:


columns_to_cluster_scaled = ['scaled_popularity', 'scaled_release', 'scaled_danceability', 'scaled_energy',
                      'scaled_key', 'scaled_mode', 'scaled_valence', 'scaled_tempo', 'scaled_duration_ms']

df_songs_scaled = pd.DataFrame(songs_scaled, columns=columns_to_cluster_scaled)
df_songs_scaled.head()


# In[11]:


plt.figure(figsize=(10, 8)) 
ax = sns.heatmap(df_songs_scaled.corr(), annot=True)
plt.show()


# In[12]:


# from sklearn.cluster import KMeans

# n_clusters = range(1, 20)

# wcss = []  # Within-cluster sum of squares
# for i in n_clusters:  # Test different numbers of clusters
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(songs_scaled)
#     wcss.append(kmeans.inertia_)


# In[13]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

n_clusters = range(2,30)
ssd = []
sc = []


for n in n_clusters:
    km = KMeans(n_clusters=n, max_iter=300, n_init=10, init='k-means++', random_state=42)
    km.fit(songs_scaled)
    preds = km.predict(songs_scaled) 
    centers = km.cluster_centers_ 
    ssd.append(km.inertia_) 
    score = silhouette_score(songs_scaled, preds, metric='euclidean')
    sc.append(score)
    print("Number of Clusters = {}, Silhouette Score = {}".format(n, score))


# In[14]:


plt.plot(n_clusters, sc, marker='.', markersize=12, color='red')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score behavior over the number of clusters')
plt.show()

plt.plot(n_clusters, ssd, marker='.', markersize=12)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for optimal K')
plt.show()


# In[15]:


selected_features['cluster'] = preds


# In[16]:


# I will provide the Python code that assigns the clusters to the dataset and counts the number of instances in each cluster.
# This code assumes that 'songs_scaled' is your dataset and 'km' is your trained KMeans model.
# Note that I will use the number of clusters with the highest silhouette score from your provided outputs, which is 6.

# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import pandas as pd

# # Assume songs_scaled is a pre-processed and scaled dataset ready for clustering
# # For demonstration, I'll create a dummy pandas DataFrame to simulate this.
# # songs_scaled = pd.DataFrame(...)  # This would be your actual scaled data

# # Number of clusters
# optimal_n_clusters = range(2,8)


# for n in optimal_n_clusters:
#     km_optimal = KMeans(n_clusters=n, max_iter=300, n_init=10, init='k-means++', random_state=42)
#     km_optimal.fit(songs_scaled)
#     cluster_labels = km_optimal.predict(selected_features)
#     centers = km_optimal.cluster_centers_ 
#     ssd.append(km_optimal.inertia_) 
#     score = silhouette_score(songs_scaled, preds, metric='euclidean')
#     sc.append(score)
#     print("Number of Clusters = {}, Silhouette Score = {}".format(n, score))

# # Assign clusters back to the original data
# selected_features['cluster'] = cluster_labels

# # Count the number of instances in each cluster
# cluster_counts = selected_features['cluster'].value_counts()

# # Output the counts for each cluster
# print(cluster_counts)


# In[17]:


selected_features.to_csv('../helpers/selected_features_clustered.csv')


# In[18]:


selected_features_clustered = pd.read_csv('../helpers/selected_features_clustered.csv', sep=',')
selected_features_clustered.set_index("id", inplace=True)
selected_features_clustered


# In[19]:


selected_features_clustered.groupby('cluster').count()


# In[20]:


preds


# In[21]:


X = selected_features_clustered.drop(columns=['cluster'])
y = selected_features_clustered['cluster']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# In[23]:


# One-hot encode the target labels
y_train_encoded = to_categorical(y_train)
y_val_encoded = to_categorical(y_val)


# In[24]:


# Build the neural network
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_val_encoded.shape[1], activation='softmax'))  # Use 'softmax' for multi-class problems


# In[25]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


# In[26]:


# Train the model using the one-hot encoded labels
history = model.fit(X_train, y_train_encoded, validation_data=(X_val, y_val_encoded), epochs=25, batch_size=64, verbose=0)


# In[27]:


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:





# In[28]:


from sklearn.metrics import confusion_matrix, accuracy_score
from numpy import argmax
# Evaluate the model
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)

# Convert predictions from one-hot encoded to class labels
y_pred_train = argmax(y_pred_train, axis=1)
y_pred_val = argmax(y_pred_val, axis=1)

train_acc = accuracy_score(y_train, y_pred_train)
val_acc = accuracy_score(y_val, y_pred_val)

print(f"Training Accuracy: {train_acc}")
print(f"Validation Accuracy: {val_acc}")

# Confusion Matrix
cm_train = confusion_matrix(y_train, y_pred_train)
print("Confusion Matrix:")
print(cm_train)


# In[ ]:





# In[29]:


sample = selected_features_clustered.sample(n=1)
inputs = scaler.transform(sample.drop(columns=['cluster']))
predicted = selected_features_clustered[selected_features_clustered['cluster'] == argmax(model.predict(inputs))].sort_values(by=['popularity'], ascending= False).head(1)
print("Input: \n",
      "id: ", sample.index.values[0],
      "\n",
      "cluster: ", sample['cluster'].values[0],
      "\n",
      "artist: ", id_information.loc[id_information['id'] == sample.index.values[0]].artist.values[0],
      "\n",
      "song: ", id_information.loc[id_information['id'] == sample.index.values[0]].song.values[0] 
     )
print("Predicted: \n",
      "id: ", predicted.index.values[0],
      "\n",
      "cluster: ", predicted.cluster.values[0],
      "\n",
      "artist: ", id_information.loc[id_information['id'] == predicted.index.values[0]].artist.values[0],
      "\n",
      "song: ", id_information.loc[id_information['id'] == predicted.index.values[0]].song.values[0] 
     )


# In[30]:


sample


# In[31]:


predicted


# In[ ]:




