# HMproduct
## H&amp;M personalized fashion recommendations

<strong>Project goal:</strong> Provide product recommendations based on previous purchase

<strong>Kaggle project link:</strong> https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations  

---

### General questions

- Given that some customer_id don't have transaction history, how do we recommend products to these customers?
- Intuitively, purchasing a fashion product could be influenced by gender (women vs. men's cloths), age, season (e.g., winter vs. summer clothes), price, and possibly area (customer data include zip code). Can we tell whether these factors actually influence customer purchase from the datasets?
- The customer data include FN (customer gets Fashion News or not), Active (customer is active in communication or not), club_member_status, and fashion_news_frequency. How would this information influence customers' purchasing behavior?
- How should we split data into training and validation sets?

### Outcome evaluation
- Fixed model: Only recommend top selling products to every customer
- Random model: Randomly recommend 12 products to each customer
- Compare outcomes between our model and Fixed or Random model

### Data input and output
- Input: 1371980 customer_id (in sample_submission.csv)
- Output: 12 article_id for each customer_id




References

- https://www.netguru.com/blog/product-recommendation-machine-learning
- https://towardsdatascience.com/what-are-product-recommendation-engines-and-the-various-versions-of-them-9dcab4ee26d5
- https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed
