# Musical Context-Aware Recommendation Using Neural Networks

This project is part of the diploma thesis by Bc. Peter Oliver Kolek under the supervision of doc. Ing. Giang Nguyen Thu, Ph.D. The aim is to design and implement a prototype for a music recommendation system using neural networks to address challenges like cold starts, data sparsity, and system scalability in streaming services like Spotify and Apple Music.

## Project Overview

With the rising popularity of streaming services, personalizing music recommendations has become critical for enhancing user experience and revenue. The recommendation system prototype in this project leverages neural networks to uncover hidden patterns in data and model non-linear interactions, requiring substantial training data to perform effectively.

## Current State

Traditional recommendation methods are increasingly being supplemented by deep learning techniques, which can deliver more relevant results. This project evaluates current (deep) neural network-based recommendation systems oriented towards user preferences and song characteristics.

## Data Analysis

Prior to training the neural network model, a thorough data analysis was conducted to understand user behavior, song popularity, and other patterns within the data. This analysis included:
- Exploratory Data Analysis (EDA) to summarize the main characteristics of the data with visualizations and statistics.

The insights from this analysis informed the feature selection and model architecture decisions in the recommendation system.

## Implementation

The prototype includes:
- Data preprocessing scripts to prepare the user interaction and song metadata for model training.
- A neural network model built using Keras, designed to predict user-song interactions.
- Evaluation scripts employing metrics such as Precision@k, Recall@k, MAP@k, MRR, and NDCG@k to assess the quality of the recommendations.

## Evaluation

The system's performance is evaluated using the following metrics:
- Precision@k: Measures the proportion of recommended items in the top-k set that are relevant.
- Recall@k: Measures the proportion of relevant items that are captured in the top-k recommendations.
- MAP@k (Mean Average Precision): Calculates the mean of the Average Precision scores for each user.
- MRR (Mean Reciprocal Rank): Focuses on the rank of the first relevant recommendation.
- NDCG@k (Normalized Discounted Cumulative Gain): Accounts for the quality of the ranking of recommendations.

## Acknowledgments

This project has been developed and supervised in collaboration with doc. Ing. Giang Nguyen Thu, Ph.D. The proposal has been approved and will be carried out as part of the diploma thesis work.

## Contact

- Student: Peter Oliver Kolek, Bc. (p.o.kolek@gmail.com)
- Researcher: Giang Nguyen Thu, doc. Ing. PhD.
