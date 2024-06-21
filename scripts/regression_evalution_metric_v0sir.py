import numpy as np
import math

import numpy as np
import math
def order_indices(ratings, imdb_list):
    '''generate the sequece of indices as rank based on the rating and imdb rating values. First rank at first position'''
    rating_array = np.array(ratings)
    unique_ratings = sorted(list(set(ratings)))[::-1]

    order_indics = []

    for ur in unique_ratings:
        rtng_indices = np.where(rating_array == ur)[0]
       

        imdb = [imdb_list[r_indx] for r_indx in rtng_indices]
        imdb = list(set(imdb))
        imdb = sorted(imdb, reverse=True)
      

        for i in imdb:
            for r_indx in rtng_indices:
                if imdb_list[r_indx] == i:
                    order_indics.append(r_indx)       

    return order_indics

def dcg_at_K(actual_ratings, indeces_order, K):
    '''generates the dcg on a cut of value of K'''
    k, dcg = 0, 0
    K = K if K<=len(actual_ratings) else len(actual_ratings)

    for indx, rating_indx in enumerate(indeces_order):
        relavance = actual_ratings[rating_indx]
        numerator = pow(2,relavance) - 1
        denomirator = math.log2(indx+2)
        dcg += numerator/denomirator

        k += 1
        if k == K:
            return dcg
        

def mrr_at_k(ac_order_indc, pr_order_indc, actual_ratings, predicted_ratings, K):
    '''generates mrr@k'''
    k, rr = 0, 0
    K = K if K<=len(predicted_ratings) else len(predicted_ratings)
    for i, indx in enumerate(pr_order_indc):
        if k<K:
            if predicted_ratings[indx] >= 3 and actual_ratings[ac_order_indc[i]] >= 3:
                rr += 1/(i+1)
            elif predicted_ratings[indx] < 3 and actual_ratings[ac_order_indc[i]] < 3:
                rr += 1/(i+1)
        k += 1
    MRR_at_k = rr/K
    return(MRR_at_k)

def fcp_at_K(actual_ratings, pr_order_indc, K): #Fraction of Concordant Pairs at K
    K = K if K<=len(actual_ratings) else len(actual_ratings)
    concordant = 0

    top_k_indices = pr_order_indc[:K]

    for i, indx in enumerate(top_k_indices):
        if i < (K-1):
            nxt_top_k_indices = pr_order_indc[i+1:K]
            for nxt_indics in nxt_top_k_indices:
                concordant += 1 if actual_ratings[indx] >= actual_ratings[nxt_indics] else 0

    total_pair = math.comb(K, 2)
    fcp_at_k = concordant/total_pair
    return fcp_at_k

def ndcg_mrr_fpc_k(actual_ratings, predicted_ratings, imdb_list, K):
    '''takes the actual and predicted ratings of movies with imdb list and cutoff k. Returns the ndgc@K, mrr@k and fcp@k'''
    ac_order_indc = order_indices(actual_ratings, imdb_list)
    pr_order_indc = order_indices(predicted_ratings, imdb_list)

    # calculating ndcg
    ndcg_k = dcg_at_K(actual_ratings, pr_order_indc, K)/dcg_at_K(actual_ratings, ac_order_indc, K)

    # calculating mrr
    MRR_at_k = mrr_at_k(ac_order_indc, pr_order_indc, actual_ratings, predicted_ratings, K)

    # calculating fcp
    fcp_at_k = fcp_at_K(actual_ratings, pr_order_indc, K)


    return (ndcg_k, MRR_at_k, fcp_at_k)

def MAE_MSE_RMSE_R2_at_K(actual_ratings, predicted_ratings, K):
    '''it returns MAE@K, MSE@K, RMSE@K and R2'''

    K = K if K<=len(actual_ratings) else len(actual_ratings)

    k, AE, SE, SS_res, SS_tot = 0, 0, 0, 0, 0
    ac_avg = np.average(np.array(actual_ratings))

    for indx, _ in enumerate(actual_ratings):
        if k < K:
            #  calculate AE
            AE += abs(actual_ratings[indx] - predicted_ratings[indx])

            # calculate SE
            error = (actual_ratings[indx] - predicted_ratings[indx])
            SE += pow(error, 2)


        # calculate R2
        error = (actual_ratings[indx] - predicted_ratings[indx])
        SS_res += pow(error, 2)
        error = actual_ratings[indx] - ac_avg
        SS_tot += pow(error, 2)

        k += 1
       

    MAE_at_K = AE/K
    MSE_at_K = SE/K
    RMSE_at_K = pow(MSE_at_K, 0.5)
    R2 = 1 - (SS_res/SS_tot)

    return(MAE_at_K, MSE_at_K, RMSE_at_K, R2)


if __name__ == "__main__":

    '''
    1. Code for evaluating seven evalution metrices: ndcg@k, mrr@k, fcp@k, MAE@k, MSE@k, RMSE@k and R2.

    2. To evaluate these just send the actual rating list and predicted rating list of movies along with the 
    movies imdb value(in integer list) and a value of k (= 3, 5, 10, 15, 20) for a single user.

    3. This code will give you the result for a single user. Finally you have to calculate the average value
    over the dataset.

    4. Ashish, don't run chatGPT of for generating response for every value of k. Store the response of chatGPT
    in your storage and then calculate these score on different k values just feteching the response from the stored
    file.

    5. I am sharing a file where you'll find the movie id wise imdb value.

    6. Bhaskar, generate these scores for your baseline model.

    7. use our training/ testing split.

    8. report me in the shared file table.'''

    actual_ratings = [5, 4, 3, 4, 1]
    predicted_ratings = [4, 5, 4, 2, 5]
    imdb_list = [9.0, 8.5, 9.0, 4.5, 3.0]
    k = 3

    result1 = ndcg_mrr_fpc_k(actual_ratings, predicted_ratings, imdb_list, k)
    print(f'ndcg@{k} = {result1[0]}, mrr_k@{k} = {result1[1]}, fcp@{k} = {result1[2]}')

    result2 = MAE_MSE_RMSE_R2_at_K(actual_ratings, predicted_ratings, k)
    print(f'MAE@{k} = {result2[0]}, MSE@{k} = {result2[1]}, RMSE@{k} = {result2[2]}, R2 = {result2[3]}')

