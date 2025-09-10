# Data volume
- Linear models need fewer samples (e.g., 50–100 per feature, 5 features -> 5 * 50 = 250 samples minimum).
In general the more complicated the model, the more data is needed for training it.
- use regularization when the amount of data is limited. Regularization reduces variance between training and testing sets (prevents overfitting).

### Regularization in Various Model Types
| Model Type | Specific Technique | How Regularization is Applied |
| :--- | :--- | :--- |
| **Linear Models** | **Ridge (L2), Lasso (L1), Elastic Net** | Adds penalty to the linear coefficients (weights). |
| **Neural Networks** | **L2/L1 Weight Decay, Dropout** | • **L2/L1:** Penalizes the weights in the network layers.<br>• **Dropout:** Randomly drops neurons during training, acting as a powerful regularization technique. |
| **Support Vector Machines (SVMs)** | **The `C` Parameter** | The `C` parameter is essentially an **L2 regularization** parameter. A small `C` means strong regularization (smoother decision boundary). |
| **Tree-Based Models** | **Pruning, Max Depth, Min Samples per Leaf** | While not an additive penalty in the loss function, these techniques **directly restrict model complexity** to reduce overfitting, which is the functional goal of regularization. |
| **Gradient Boosting Machines (e.g., XGBoost, LightGBM)** | **L2/L1 regularization, Max Depth, Learning Rate** | Frameworks like XGBoost explicitly include L1 (`alpha`) and L2 (`lambda`) regularization terms in their objective functions to penalize the weights of the individual trees. |

### Key Differences Between L1 and L2 Regularization
| Aspect | L1 Regularization (Lasso) | L2 Regularization (Ridge) |
| :--- | :--- | :--- |
| **Penalty Term** | Penalizes the absolute value of weights | Penalizes the squared value of weights |
| **Effect on Weights** | Creates sparse models (many weights become exactly zero) | Shrinks weights evenly (all weights become small but non-zero) |
| **Solution Type** | Feature selection | Feature weighting |
| **Robustness** | More robust to outliers in data | Less robust; sensitive to outliers |
| **Computational** | Non-differentiable at zero (requires special solvers) | Everywhere differentiable (easier to optimize) |
| **Geometric Shape** | Diamond-shaped constraint region | Circular-shaped constraint region |


# Data quality
- Noise. Noisy data - high variance with low signal. A model is overly sensitive to noise or irrelevant patterns in the data, while failing to capture the true underlying signal (the meaningful patterns). This often leads to poor generalization on unseen data.
High Variance -> the model is overfitting.
Low Signal -> the data contains weak or obscured patterns.
Scatterplots: Random dispersion suggests noise.
PCA/t-SNE: If projected data points show no clusters, noise may dominate.
- Imbalance. Imbalance occurs when one class dominates others (e.g., 99% negative, 1% positive).


# Model
| Model | Linear or Non-Linear? | Explanation |
| :--- | :--- | :--- |
| **Linear Regression** | **Linear** | Models the output as a weighted sum of the input features. |
| **Logistic Regression** | **Linear** | Despite its name, it's a linear classifier. It makes decisions based on a linear combination of inputs passed through a non-linear *activation* function (sigmoid). The decision boundary is linear. |
| **Linear SVM** | **Linear** | Finds a linear hyperplane that best separates the classes. |
| **Decision Tree** | **Non-Linear** | Splits the feature space into axis-aligned rectangles. This is a highly non-linear and discontinuous process. |
| **Random Forest** | **Non-Linear** | An ensemble of non-linear Decision Trees. |
| **Gradient Boosting (XGBoost, etc.)** | **Non-Linear** | An ensemble of non-linear weak learners (usually trees). |
| **Neural Networks (with non-linear activations)** | **Non-Linear** | The combination of layers with non-linear activation functions (ReLU, sigmoid, tanh) allows them to approximate any complex non-linear function. |
| **RBF SVM** | **Non-Linear** | Uses the Kernel Trick with the Radial Basis Function (RBF) kernel to learn highly non-linear decision boundaries. |

Many machine learning algorithms (like SVM) are fundamentally linear. They work great for data that can be separated by a line or plane. But if the data cannot be separated by a line the kernel trick is applied. The kernel trick solves this with a 3-step process:
1. Transform: Map the original data points from their original input space to a much higher-dimensional feature space.
2. Solve: In this new high-dimensional space, the data becomes linearly separable. Now a linear algorithm (like a linear SVM) can find a separating hyperplane.
3. The Trick: Do all this without ever actually performing the expensive transformation!


# Transformation
- for categorical data - codification, one-hot encoding 
- standardization/normalization


# Cleaning
- consider dropping attributes the values of which are missing for the majority of observations (> 50%)
- data imputation is the process of filling in or replacing missing values

| Category | Method | Description | Best for... |
| :--- | :--- | :--- | :--- |
| **Simple** | Mean / Median Imputation | Replaces missing values with the mean or median of the column. | Numerical data with a small percentage of missing values. |
| | Mode Imputation | Replaces missing values with the most frequent value. | Categorical or discrete data. |
| | Arbitrary Value Imputation | Replaces missing values with a pre-defined number (e.g., -1, 999). | Flagging missingness as a distinct category. |
| **Predictive** | KNN Imputation | Uses the K-nearest neighbors to find similar data points and impute based on their values. | Datasets with a small number of features and complex relationships. |
| | Regression Imputation | Uses a regression model to predict and fill in the missing values based on other features. | Data with a strong linear or non-linear relationship between variables. |
| **Advanced** | MICE (Multiple Imputation by Chained Equations) | Creates multiple complete datasets by iteratively imputing missing values with a predictive model. | Complex datasets with a moderate to high percentage of missing values. |
| | PMM (Predictive Mean Matching) | A variation of MICE that uses a regression model to find the closest observed value to the predicted one for imputation. | When imputed values need to be realistic and within the existing data range. |
- remove observations with missing values for the majority of attributes
- consider removing outliers, but make sure not to remove minority class observations (anomaly detection)
- Note: data should be brought as close as possible to normal distribution


# Reduction
- remove duplicate observations
- remove duplicate attributes (correlation analysis)
- attribute selection (forward, backward)
- dimensionality reduction. Purpose:
    - Fighting the Curse of Dimensionality: As the number of features (dimensions) in a dataset grows, the data becomes increasingly sparse. This makes it difficult for many machine learning models to find meaningful patterns, leading to poor performance and long training times.
    - Reducing Overfitting: With fewer features to model, there's less risk of the model learning noise in the data. PCA can effectively act as a form of regularization. By compressing the data into a few principal components, you're forcing the model to focus on the most important underlying patterns, not the random fluctuations of individual features.


# Clustering
- can identify distinct clusters and then analyze the characteristics of the data points within each group.
- can be used for dimensionality reduction (represent each cluster with a single centroid)
- can be used for anomaly detection (identify outliers, common technique in fraud detection or network intrusion detection)
- can add a label that indicates which cluster each data point belongs to (unsupervised)
- can use clustering on unlabeled data to find groups, and then a small number of labeled examples can be used to label entire clusters, a process known as semi-supervised learning

# Data linearity
Use residual plots to figure out if features have non-linear relationships. If non-linear you'll see a clear pattern (e.g., a U-shape or curve) in the residuals. This is a dead giveaway that a linear model is missing a key pabesttern.

Train a simple linear model and a simple non-linear model (like a Random Forest or Gradient Boosting machine) on the same data. If the non-linear model significantly outperforms the linear model, your data has important non-linear relationships.


### Techinques for dimensionality reduction in non-linear data
| Technique | Best For | Key Consideration |
| :--- | :--- | :--- |
| **UMAP** | **General-purpose non-linear reduction & visualization.** | Fast, preserves global & local structure. Modern default choice. |
| **t-SNE** | **Detailed visualization of local clusters & patterns.** | Slow, for visualization only. Excellent for fine-grained analysis. |
| **Isomap** | Data on a clear, continuous manifold (e.g., Swiss roll). | Computationally heavy. Geodesic distance-based. |
| **Kernel PCA** | A non-linear extension of the PCA framework. | Requires kernel selection. Good for certain non-linearities. |
| **Autoencoders** | **Extremely complex, large-scale data** (e.g., images). | Most flexible but requires expertise and large datasets. |


# Feature extraction
- TF-IDF (Term Frequency-Inverse Document Frequency).
Term Frequency (TF):k How often a word appears in a document.
Inverse Document Frequency (IDF): Measures how unique or rare a term is across all documents. Common words (e.g., "the", "is") get penalized.
TF-IDF Score combines TF and IDF to weigh terms.
TF-IDF filters out common words, highlights important words
Used in: Search engines, document classification, keyword extraction.
TF-IDF does not capture semantics or context (ignores word order, struggles with synonyms).
- Word embeddings (Word2Vec, GloVe).
These are dense vector embedding techniques that capture semantic meaning of words by analyzing their co-occurrence patterns in large text corpora. Unlike traditional methods (e.g., TF-IDF), they represent words as continuous vectors in a lower-dimensional space, where geometric relationships (distance, direction) reflect linguistic relationships (capture meanings beyond words, map words to vectors where distance = semantic similarity).
Synonyms: "happy" ≈ "joyful" (close in vector space).
Analogies: Paris - France ≈ Berlin - Germany
Hierarchies: "animal" → "dog" → "poodle".
Unlike TF-IDF, word embeddings capture semantics (but don't capture context).
- Transformers (BERT).
Use pretrained models (BERT) to project text data into the latent space.
Unlike TF-IDF and word embeddings, transformers capture both semantics and context.
- Word2Vec vs BERT:
The core difference between Word2Vec and BERT is that Word2Vec produces a static embedding for each word, while BERT generates a dynamic, contextualized embedding.
    - Word2Vec Embeddings: Word2Vec learns a single, fixed vector for each word in a vocabulary. For example, the word "bank" will have the same embedding regardless of whether it's used in the sentence "I went to the bank to deposit money" or "I sat on the bank of the river." It learns its representation by looking at the words that appear around it during training, but once trained, the embedding is static.
    - BERT Embeddings: BERT takes the entire sentence as input and uses its Transformer architecture to generate an embedding for each word based on all the other words in that specific sentence.

# Data synthesis (oversampling)
- SMOTE (for numeric features), SMOTE-NC (for categorical features, assigns the most frequent category among the k-nearest neighbors)
- ADASYN - generates synthetic samples with a focus on adaptively creating more samples near difficult-to-learn decision boundaries.
ADASYN generates more samples near boundaries, while SMOTE does balanced distribution.
- GAN (generative adversarial network)
- data augmentation (for text - synonym replacement)


# Performance metrics
- confusion matrix

    | | Predicted Positive | Predicted Negative |
    | :--- | :--- | :--- |
    | **Actual Positive** | True Positive (TP) | False Negative (FN) |
    | **Actual Negative** | False Positive (FP) | True Negative (TN) |
- precision-recall metrics.

    Precision - measures how many predicted positives are correct.
    $$
    \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
    $$
    Recall  (also called Sensitivity or True Positive Rate) - measure the ability of a model to correctly identify all relevant instances of the positive class. "Out of all actual positive cases, how many did the model correctly predict?"
    $$
    \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
    $$
- F1 score - harmonic mean of precision and recall. Balances both metrics (best for severe imbalance).
- ROC-AUC.
    ROC Curve (Receiver Operating Characteristic): Plots True Positive Rate (recall) vs. False Positive Rate (FPR) at various thresholds.

    AUC (Area Under Curve). 1.0 = Perfect classifier. 0.5 = Random guessing.
    
    Useful for binary classification with moderate imbalance.
- PR-AUC (Precision-Recall AUC) - focuses on precision-recall trade-off.

    Better than ROC-AUC for severe imbalance (ignores true negatives).
- Matthews Correlation Coefficient (MCC).
1 = Perfect prediction, 0 = Random, -1 = Inverse prediction.
Robust to imbalance.
- cross-validation - evaluates how well a model generalizes to unseen data. Instead of splitting data into just one train-test set, CV divides the data multiple times to get a more reliable estimate of model performance.


## todo
- how to process time-series data
