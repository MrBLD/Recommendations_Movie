# Recommendation System Evaluation Repository

## Overview

This repository is designed to evaluate and benchmark various recommendation algorithms using popular evaluation metrics. I employ two primary libraries: the Surprise library for traditional collaborative filtering techniques and a custom Cornac library for more advanced models. The algorithms evaluated include:

### From Surprise Library:
1. **ItemKNN** - Item-based k-nearest neighbors
2. **UserKNN** - User-based k-nearest neighbors
3. **SVD** - Singular value decomposition
4. **SVD++** - An extension of SVD that considers implicit ratings
5. **Random** - Random recommendations for baseline comparison

### From Custom Cornac Library:
1. **BPR** - Bayesian Personalized Ranking
2. **VBPR** - Visual Bayesian Personalized Ranking
3. **PMF** - Probabilistic Matrix Factorization
4. **MCF** - Multi-criteria Collaborative Filtering
5. **GridSearch_VAECF** - Variational AutoEncoder for Collaborative Filtering with Grid Search
6. **RandomSearch_VAECF** - Variational AutoEncoder for Collaborative Filtering with Random Search
7. **LightGCN** - Light Graph Convolutional Networks
8. **NGCF** - Neural Graph Collaborative Filtering

### Evaluation Metrics:
- **ndcg@k** - Normalized Discounted Cumulative Gain at k
- **Relative_mrr@k** - Relative Mean Reciprocal Rank at k
- **fcp@k** - Fraction of Concordant Pairs at k
- **MAE@k** - Mean Absolute Error at k
- **MSE@k** - Mean Squared Error at k
- **RMSE@k** - Root Mean Squared Error at k
- **R²@k** - Coefficient of Determination at k

## Repository Structure

The repository is organized as follows:

- **notebooks/**: Jupyter notebooks for recommedation model's training/evaluation.
- **/data/ml-100k**: Contains the datasets used for evaluation.
- **/scripts/**: Source code for data preprocessing, and evaluation scripts.
- **/Results/**: Directory to store the results of the evaluations and its exloratory analysis.
<!-- - **requirements.txt**: List of dependencies and libraries needed to run the code. -->
- **README.md**: This documentation file.

## Setup and Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-username/recommendation-system-evaluation.git
cd recommendation-system-evaluation
# pip install -r requirements.txt
```

### Required Libraries

- **Surprise**: A Python scikit for building and analyzing recommender systems.
- **Cornac**: A recommender system library for both collaborative filtering and content-based methods.
- **numpy**: Fundamental package for array computing.
- **pandas**: Data manipulation and analysis library.
- **scikit-learn**: Machine learning library for Python.
- **torch**: PyTorch, a deep learning framework for advanced models.
- **evaluate**: A library for evaluating machine learning models.

Install the libraries using:
```bash
pip install surprise cornac numpy pandas scikit-learn torch evaluate
```

## Data Preparation

Place your datasets in the `/data/ml-100k` directory. The repository expects the data in a specific format:

1. **User-item interaction data**: A standard movielens file formats with columns `userId`, `itemId`, `rating`.
2. **Additional data**: Similar to Movielens-100k dataset

## Running the Evaluations

### Evaluating Models from the Surprise Library

To evaluate the models from the Surprise library, use the provided Notebook `Rec_Baselines5_v2_Final.ipynb`. This notbook will train and test the specified models on the dataset provided and export the evaluations in csv formats inside the Results directory.

```bash
notebook Rec_Baselines5_v2_Final.ipynb
```

### Evaluating Models from the Cornac Library

To evaluate the models from the Cornac library, use the provided Notebooks. These notbook will train and test the specified models on the dataset provided. It will generate the evaluations based on the metrics declared in the notebooks(modify as per the needs).

```bash
notebook LightGCN_NDCF.ipynb
notebook PMF_MCF_text_to_graph.ipynb
notebook VAECF_param_search.ipynb
notebook VBPR_BPR_text.ipynb
```

Both scripts will output the evaluation metrics to the console and save detailed results in the `/results/` directory.

### Evaluation Metrics

The following metrics are computed:

- **ndcg@k**: Measures the quality of the recommendations by considering the position of the relevant items.
- **Relative_mrr@k**: Evaluates the rank of the first relevant item in the recommendation list.
- **fcp@k**: Fraction of Concordant Pairs, measuring the proportion of correctly ordered pairs.
- **MAE@k**: Mean Absolute Error of the top-k recommendations.
- **MSE@k**: Mean Squared Error of the top-k recommendations.
- **RMSE@k**: Root Mean Squared Error of the top-k recommendations.
- **R²@k**: Coefficient of Determination for the top-k recommendations.

The metric calculations are handled within the evaluation scripts.

## Customizing and Extending

You can customize and extend the evaluation scripts to include additional models or metrics:

1. **Add new models**: Implement the model training and prediction in `cornac/models` in the repository `https://github.com/MrBLD/cornac.git`.
2. **New evaluation metrics**: Add the metric computation in `cornac/eva_methods`.

## Contributing

Contributions are welcome! If you want to contribute, please fork the repository and submit a pull request. Make sure to follow the established code style and include tests for any new features.

## References

- [Surprise Documentation](https://surprise.readthedocs.io/en/stable/)
- [Cornac Documentation](https://cornac.readthedocs.io/en/latest/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Feel free to reach out if you have any questions or need further assistance!

Happy Recommending!