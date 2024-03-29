
    def sample_top_n_recs(self, ml, test_subject=85, k=10):
        
        for algo in self.algorithms:
            print("\nRecommending for: ",ml.get_movie_name(test_subject))
            print("\nUsing recommender ", algo.get_name())
            
            print("\nBuilding recommendation model...")
            train_set = self.dataset.get_full_train_set()
            algo.get_algorithm().fit(train_set)
            
            print("Computing recommendations...")
            test_set = self.dataset.get_anti_test_set_for_user(test_subject)
        
            predictions = algo.get_algorithm().test(test_set)
            # print(predictions)
            
            recommendations = []
            
            print ("\nWe recommend:")
            for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
                int_movie_id = int(movie_id)
                recommendations.append((int_movie_id, actual_rating, estimated_rating))
            
            recommendations.sort(key=lambda x: x[0], reverse=False)
            for ratings in recommendations[:k]:
                print(ml.get_movie_name(ratings[0]), ratings[1])
            print(recommendations)            
            print("-----------------------------------------")
            return recommendations