# for rating not in 1-5

# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np


class RatingMetric:
    """Rating Metric.

    Attributes
    ----------
    name: string,
        Name of the measure.

    type: string, value: 'rating'
        Type of the metric, e.g., "ratingsing", "rating".

    """

    def __init__(self, name=None, k=-1, higher_better=False):
        assert hasattr(k, "__len__") or k == -1 or k > 0
        self.type = 'rating'
        self.name = name
        self.k = k
        self.higher_better = higher_better

    def compute(self, **kwargs):
        raise NotImplementedError()


class MAE(RatingMetric):
    """Mean Absolute Error.

    Attributes
    ----------
    name: string, value: 'MAE'
        Name of the measure.

    """

    def __init__(self,k=-1):
        RatingMetric.__init__(self, name='MAE@{}'.format(k),k=k)
    # def __init__(self):
    #     RatingMetric.__init__(self, name='MAE')

    def compute(self, gt_ratings, pd_ratings, weights=None, **kwargs):
        """Compute Mean Absolute Error.

        Parameters
        ----------
        gt_ratings: Numpy array
            Ground-truth rating values.

        pd_ratings: Numpy array
            Predicted rating values.

        weights: Numpy array, optional, default: None
            Weights for rating values.

        **kwargs: For compatibility

        Returns
        -------
        mae: A scalar.
            Mean Absolute Error.

        """
        if self.k > 0:
            truncated_pd_ratings = pd_ratings[:self.k]
            truncated_gt_ratings=gt_ratings[:self.k]
        else:
            truncated_pd_ratings = pd_ratings
            truncated_gt_ratings=gt_ratings
        
        array_min = np.min(truncated_pd_ratings)
        array_max = np.max(truncated_pd_ratings)

        # Normalize the array to the range [0, 1]
        if array_max == array_min:
            truncated_pd_ratings=truncated_pd_ratings
        else:
            normalized_array = (truncated_pd_ratings - array_min) / (array_max - array_min)
            # Scale the normalized array to the desired range
            truncated_pd_ratings = 1 + (5 - 1) * normalized_array
                
        
        
        mae = np.average(np.abs(truncated_gt_ratings - truncated_pd_ratings), axis=0, weights=weights)
        return mae


class MSE(RatingMetric):
    """Mean Squared Error.

    Attributes
    ----------
    name: string, value: 'MSE'
        Name of the measure.

    """
    
    def __init__(self,k=-1):
        RatingMetric.__init__(self, name='MSE@{}'.format(k),k=k)

    def compute(self, gt_ratings, pd_ratings, weights=None, **kwargs):
        """Compute Mean Squared Error.

        Parameters
        ----------
        gt_ratings: Numpy array
            Ground-truth rating values.

        pd_ratings: Numpy array
            Predicted rating values.

        weights: Numpy array, optional, default: None
            Weights for rating values.

        **kwargs: For compatibility

        Returns
        -------
        mse: A scalar.
            Mean Squared Error.

        """
        if self.k > 0:
            truncated_pd_ratings = pd_ratings[:self.k]
            truncated_gt_ratings=gt_ratings[:self.k]
        else:
            truncated_pd_ratings = pd_ratings
            truncated_gt_ratings=gt_ratings
        
        array_min = np.min(truncated_pd_ratings)
        array_max = np.max(truncated_pd_ratings)

        # Normalize the array to the range [0, 1]
        if array_max == array_min:
            truncated_pd_ratings=truncated_pd_ratings
        else:
            normalized_array = (truncated_pd_ratings - array_min) / (array_max - array_min)
            # Scale the normalized array to the desired range
            truncated_pd_ratings = 1 + (5 - 1) * normalized_array
                    
        mse = np.average((truncated_gt_ratings - truncated_pd_ratings) ** 2, axis=0, weights=weights)
        return mse


class RMSE(RatingMetric):
    """Root Mean Squared Error.

    Attributes
    ----------
    name: string, value: 'RMSE'
        Name of the measure.

    """
    def __init__(self,k=-1):
        RatingMetric.__init__(self, name='RMSE@{}'.format(k),k=k)

    def compute(self, gt_ratings, pd_ratings, weights=None, **kwargs):
        """Compute Root Mean Squared Error.

        Parameters
        ----------
        gt_ratings: Numpy array
            Ground-truth rating values.

        pd_ratings: Numpy array
            Predicted rating values.

        weights: Numpy array, optional, default: None
            Weights for rating values.

        **kwargs: For compatibility

        Returns
        -------
        rmse: A scalar.
            Root Mean Squared Error.

        """
        if self.k > 0:
            truncated_pd_ratings = pd_ratings[:self.k]
            truncated_gt_ratings=gt_ratings[:self.k]
        else:
            truncated_pd_ratings = pd_ratings
            truncated_gt_ratings=gt_ratings
        
        array_min = np.min(truncated_pd_ratings)
        array_max = np.max(truncated_pd_ratings)

        # Normalize the array to the range [0, 1]
        if array_max == array_min:
            truncated_pd_ratings=truncated_pd_ratings
        else:
            normalized_array = (truncated_pd_ratings - array_min) / (array_max - array_min)
            # Scale the normalized array to the desired range
            truncated_pd_ratings = 1 + (5 - 1) * normalized_array
                
        
        mse = np.average((truncated_gt_ratings - truncated_pd_ratings) ** 2, axis=0, weights=weights)
        rmse=np.sqrt(mse)
        # print(mse, rmse)
        return rmse

class FCP(RatingMetric):
    def __init__(self,k=-1):
        RatingMetric.__init__(self, name='FCP@{}'.format(k),k=k)

    def compute(self, gt_ratings, pd_ratings, weights=None, **kwargs):
        
        if self.k > 0:
            truncated_pd_ratings = pd_ratings[:self.k]
            truncated_gt_ratings=gt_ratings[:self.k]
        else:
            truncated_pd_ratings = pd_ratings
            truncated_gt_ratings=gt_ratings
        
        array_min = np.min(truncated_pd_ratings)
        array_max = np.max(truncated_pd_ratings)

        # Normalize the array to the range [0, 1]
        if array_max == array_min:
            truncated_pd_ratings=truncated_pd_ratings
        else:
            normalized_array = (truncated_pd_ratings - array_min) / (array_max - array_min)
            # Scale the normalized array to the desired range
            truncated_pd_ratings = 1 + (5 - 1) * normalized_array
                
        print(truncated_pd_ratings, truncated_gt_ratings)
        
        concordant_pairs=0
        disconcordant_pairs=0
        n=len(truncated_gt_ratings)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if (truncated_gt_ratings[i]>truncated_gt_ratings[j] and truncated_pd_ratings[i]>truncated_pd_ratings[j]):
                        concordant_pairs+=1
                    else:
                        disconcordant_pairs+=1

        total_pairs=concordant_pairs+disconcordant_pairs
        if(total_pairs == 0):
            total_pairs=n
        else:
            total_pairs=total_pairs

        c_index = np.float64(concordant_pairs /total_pairs)
        return c_index

class R2(RatingMetric):
    def __init__(self,k=-1):
        RatingMetric.__init__(self, name='R2@{}'.format(k),k=k)

    def compute(self, gt_ratings, pd_ratings, weights=None, **kwargs):
    
        if self.k > 0:
            truncated_pd_ratings = pd_ratings[:self.k]
            truncated_gt_ratings=gt_ratings[:self.k]
        else:
            truncated_pd_ratings = pd_ratings
            truncated_gt_ratings=gt_ratings
        
        array_min = np.min(truncated_pd_ratings)
        array_max = np.max(truncated_pd_ratings)

        # Normalize the array to the range [0, 1]
        if array_max == array_min:
            truncated_pd_ratings=truncated_pd_ratings
        else:
            normalized_array = (truncated_pd_ratings - array_min) / (array_max - array_min)
            # Scale the normalized array to the desired range
            truncated_pd_ratings = 1 + (5 - 1) * normalized_array
                
        
        # mse = np.average((truncated_gt_ratings - truncated_pd_ratings) ** 2, axis=0, weights=weights)
        # return np.sqrt(mse)

        # Assuming truncated_gt_ratings and truncated_pd_ratings are already defined
        # Calculate the mean of ground truth ratings
        mean_gt_ratings = np.mean(truncated_gt_ratings)
        
        # print(mean_gt_ratings, gt_ratings, pd_ratings)
        # Calculate TSS
        tss = np.sum((truncated_gt_ratings - mean_gt_ratings) ** 2)

        # Calculate RSS
        rss = np.sum((truncated_gt_ratings - truncated_pd_ratings) ** 2)

        # Calculate R-squared
        if(tss == 0):
            tss=rss
        else:
            return 1 - (rss / tss)
        
        # print (truncated_pd_ratings, truncated_gt_ratings, (1-(rss/tss)))
        return 1 - (rss / tss)