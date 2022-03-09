import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def all_user_collaborative_filter(X, similar_user_number = 1, max_recommend_items = None):

    """
    user-based collaborative filtering for every user in X

    X: a dataframe containing two columns, first column is users, second is purchased item
    similar_user_number: select the number of top similar users
    max_recommend_items: an integer to specify the max. number of recommended items

    """

    #add a value=1 column to data
    X = pd.DataFrame({"user":X.iloc[:,0],
                    "item":X.iloc[:,1],
                    "value":[1]*len(X)})

    #drop duplicate rows, if one purchased more than one items
    X.drop_duplicates(inplace=True)

    #convert dataframe to user-by-item matrix and fill nan with 0
    user_item_df = X.pivot(index=X.columns[0], columns=X.columns[1])['value'].fillna(0).astype('int8')

    #convert user_item_df to numpy matrix
    matrix = np.array(user_item_df)

    #print("start cosine")
    #calculate cosine similarity
    similarity = cosine_similarity(matrix)
    #print("cosine end")

    #sort user index by similarity (left-high/right-low) for each row(user)
    sorted_sim_index = np.fliplr(np.argsort(similarity))

    #number of total users
    num_total_users = matrix.shape[0]
    #item id/name list
    item_list = list(user_item_df.columns)

    #empty dict to store final results
    recommended_dict = {}

    #for each target user
    for i in range(num_total_users):
        user_id = user_item_df.index[i]

        recommended_items = []

        #Look at similar users
        for j in range(similar_user_number):
            not_purchased = matrix[sorted_sim_index[i,j+1]] - matrix[i]

            #check out item names
            for k in np.where(not_purchased == 1)[0]:
                if item_list[k] not in recommended_items:
                    recommended_items.append(item_list[k])
                    if max_recommend_items == None:
                        continue
                    elif len(recommended_items) == max_recommend_items:
                        break
            if max_recommend_items == None:
                continue
            elif len(recommended_items) == max_recommend_items:
                break

        #add target user_id and recommended items to dict
        recommended_dict[user_id] = recommended_items

        #print message if recommended_items less than max_recommend_items
        if max_recommend_items != None and len(recommended_items) < max_recommend_items:
            print(user_id,"has items less than",max_recommend_items)

    return recommended_dict

def user_collaborative_filter(X, target_user, similar_user_number = 1,
                              max_recommend_items = None, similar_user_thresh = None):
    """
    user-based collaborative filter for a specific user in X

    X: a dataframe containing two columns, first column is users, second is purchased item
    target_user: a string specified the user we plan to give recommendations
    similar_user_number: select the number of top similar users
    max_recommend_items: an integer to specify the max. number of recommended items
    similar_user_thresh (WORK IN PROGRESS): a floating number between 0 and 1 to set
                                            a minimum threshold for consine similarity
    """

    #add a value=1 column to data
    X = pd.DataFrame({"user":X.iloc[:,0],
                    "item":X.iloc[:,1],
                    "value":[1]*len(X)})

    #drop duplicate rows, if one purchased more than one items
    X.drop_duplicates(inplace=True)

    #convert dataframe to user-by-item matrix and fill nan with 0
    user_item_df = X.pivot(index=X.columns[0], columns=X.columns[1] )['value'].fillna(0)

    #convert user_item_df to numpy matrix
    matrix = np.array(user_item_df)

    #print("start cosine")
    #calculate cosine similarity
    similarity = cosine_similarity(matrix)
    #print("cosine end")

    #get target user index
    target_user_index = int(np.where(user_item_df.index == target_user)[0])

    #get the row of target user similarity
    target_user_similarity = similarity[target_user_index,:]

    #sort the index of user similarity in descending order
    #use [::-1] to reverse the order
    #[1:] filter out the target user [0] itself
    similar_user_index = np.argsort(target_user_similarity)[::-1][1:]

    recommended_items = []
    item_list = list(user_item_df.columns)
    #get unpurchased item index by substrating target user from most similar user
    for i in range(similar_user_number):
        not_purchased = matrix[similar_user_index[i]] - matrix[target_user_index]

        #check out item names
        for j in np.where(not_purchased == 1)[0]:
            if item_list[j] not in recommended_items:
                recommended_items.append(item_list[j])
                if max_recommend_items is not None:
                    if len(recommended_items) == max_recommend_items:
                        return recommended_items

    return recommended_items

def average_precision(x):
    """
    Calculate average precision for each user
    x: a binary vector
    """
    x = np.array(x)

    #number of items
    k = len(x)

    precision = 0
    numerator = 0

    for i in range(k):

        numerator += x[i]
        precision += numerator / (i+1)

    return precision/k

def MAP(X, y):
    '''
    Calculate Mean Average Precision for the recommendation
    X: A dictionary containing user id as key and recommended items as values
    y: two column data frame of validation data (1st column is user id; 2nd is item id)

    return: MAP, number_of_users
    '''
    number_users = 0

    unique_user_in_test = y.iloc[:,0].unique()

    #store average precision for each target user
    AP_res = []

    #for each target user
    for user, recommend_items in tqdm(X.items()):

        #when found in test data
        if user in unique_user_in_test:
            number_users += 1

            #get actual purchased items from y
            subset = y.loc[y.iloc[:,0] == user]
            purchased_items = set(subset.iloc[:,1])

            #empty array to store binary data
            purchased_vector = []

            #for each recommended item
            for item in recommend_items:

                #when it is actually purchased
                if item in purchased_items:
                    purchased_vector.append(1)
                #if not purchased
                else:
                    purchased_vector.append(0)

            #calculate Average Precision
            AP = average_precision(purchased_vector)
            #append to AP_res
            AP_res.append(AP)


    print("Number of users:", number_users)
    return sum(AP_res)/len(AP_res), number_users

def AOP(X, y):
    """
    Calculate Average of Precision:
    number of recommended items actually purchased / number of recommended items
    X: A dictionary containing user id as key and recommended items as values
    y: two column data frame of validation data (1st column is user id; 2nd is item id)

    return: AOP, number_of_users
    """
    number_users = 0

    unique_user_in_test = y.iloc[:,0].unique()

    #store average precision for each target user
    AP_res = []

    #for each target user
    for user, recommend_items in tqdm(X.items()):

        #when found in test data
        if user in unique_user_in_test:
            number_users += 1

            #get actual purchased items from y
            subset = y.loc[y.iloc[:,0] == user]
            purchased_items = set(subset.iloc[:,1])

            #empty array to store binary data
            purchased_vector = []

            #for each recommended item
            for item in recommend_items:

                #when it is actually purchased
                if item in purchased_items:
                    purchased_vector.append(1)
                #if not purchased
                else:
                    purchased_vector.append(0)

            #calculate average of precision
            AP = sum(purchased_vector) / len(purchased_vector)
            #append to AP_res
            AP_res.append(AP)

    print("Number of users:", number_users)
    return sum(AP_res)/len(AP_res), number_users
