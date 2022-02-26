# HMproduct
## H&amp;M personalized fashion recommendations

<strong>Project goal:</strong> Provide product recommendations based on previous purchase

<strong>Kaggle project link:</strong> https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations  

---
## Recommender Systems
```
+-- Collaborative filtering
|    +-- Memory-based
|    |   +-- User-user similarity (User-based)
|    |   +-- Item-item similarity (Item-based)
|    |       (use consine similarity or Euclidian distance)
|    |
|    +-- Model-based
|         (use Matrix Factorization, SVD)
|
+-- Content-based
|     (use item-property similarity, user past preferences)
|
+-- Hybrid
```
### General questions

- Given that some customer_id don't have transaction history, how do we recommend products to these customers?
- Intuitively, purchasing a fashion product could be influenced by gender (women vs. men's cloths), age, season (e.g., winter vs. summer clothes), price, and possibly area (customer data include zip code). Can we tell whether these factors actually influence customer purchase from the datasets?
- The customer data include FN (customer gets Fashion News or not), Active (customer is active in communication or not), club_member_status, and fashion_news_frequency. How would this information influence customers' purchasing behavior?
- How should we split data into training and validation sets?

### Outcome evaluation
- Top-selling model: Only recommend top selling products to every customer
- Random model: Randomly recommend 12 products to each customer
- Compare outcomes between our model and Fixed or Random model

### Data input and output
- Input: 1371980 customer_id (in sample_submission.csv)
- Output: 12 article_id for each customer_id

### Models to use
1. Collaborative filtering
  - user-user collaborative filtering / user-based recommender
    * Create binary vector of purchased items for each customer,
      |customer| article_1 | article_2 | article_3 |
      |--|--|--|--|
      | Ben | 1 | 1 | 1 |
      | John | 1 | 0 | 1 |
      | David| 1 | 0 | 0 |
    * Pairwise calculate cosine similarity between customers
      ```
      from sklearn.metrics import pairwise
      vector_Ben = [[1,1,1]]
      pairwise.cosine_similarity([[1,1,0]],[[1,0,1]]) #For Ben and John
      ```
    * For a target customer, identify other customers with highest cosine similarity
    * Recommend products purchased by other customers but not yet purchased by target customer

  - item-item collaborative filtering (e.g., Amazon)
    * Transpose the above matrix and run similar procedure
2. Content-based filtering (based on the features of items themselves)

3. Hybrid recommendations


### References
#### Concept overview
- [How to Build a Product Recommendation System using Machine Learning](https://www.netguru.com/blog/product-recommendation-machine-learning)
- [What are Product Recommendation Engines?](https://towardsdatascience.com/what-are-product-recommendation-engines-and-the-various-versions-of-them-9dcab4ee26d5)
- [Machine Learning for Recommender systems — Part 1 (algorithms, evaluation and cold start)](https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed)
- [Recommender Systems in Python 101](https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101)
-[Introduction to Recommender Systems](https://tryolabs.com/blog/introduction-to-recommender-systems)

#### Collaborative Filtering (CF)
- [Various Implementations of Collaborative Filtering](https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0)
- [Building a movie recommender system with Python](https://medium.com/@bkexcel2014/building-movie-recommender-systems-using-cosine-similarity-in-python-eff2d4e60d24)
- [Building a Song Recommendation System using Cosine Similarity and Euclidian Distance](https://medium.com/@mark.rethana/building-a-song-recommendation-system-using-cosine-similarity-and-euclidian-distance-748fdfc832fd)
- [Build a Recommendation Engine With Collaborative Filtering](https://realpython.com/build-recommendation-engine-collaborative-filtering/)

#### CF user-based
#### CF item-based
- [Item-based collaborative filtering in Python](https://towardsdatascience.com/item-based-collaborative-filtering-in-python-91f747200fab)

#### CF model-based

#### Content-based Recommender
- [Beginners Guide to learn about Content Based Recommender Engines](https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-learn-content-based-recommender-systems/)

#### Evaluation metrics
