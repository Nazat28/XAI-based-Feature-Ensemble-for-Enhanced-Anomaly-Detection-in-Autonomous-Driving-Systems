# XAI-based-Feature-Ensemble-for-Enhanced-Anomaly-Detection-in-Autonomous-Driving-Systems

The rapid advancement of autonomous vehicle (AV) technology has introduced significant
challenges in ensuring transportation security and reliability. Traditional AI models for anomaly detection
in AVs are often opaque, posing difficulties in understanding and trusting their decision-making processes.
This paper proposes a novel feature ensemble framework that integrates multiple Explainable AI (XAI)
methods—SHAP, LIME, and DALEX—with various AI models to enhance both anomaly detection and
interpretability. By fusing top features identified by these XAI methods across six diverse AI models
(Decision Trees, Random Forests, Deep Neural Networks, K-Nearest Neighbors, Support Vector Machines,
and AdaBoost), the framework creates a robust and comprehensive set of features critical for detecting
anomalies. These feature sets, produced by our feature ensemble framework, are evaluated using independent
classifiers (CatBoost, Logistic Regression, and LightGBM) to ensure unbiased performance. We evaluated
our feature ensemble approach on two popular autonomous driving datasets (VeReMi and Sensor) datasets.
Our feature ensemble technique demonstrates improved accuracy, robustness, and transparency of AI
models, contributing to safer and more trustworthy autonomous driving systems


**Overview of the Framework at a Glance**

![Fusion_pls](https://github.com/user-attachments/assets/a17ff5c0-bb04-4040-a836-f6c70dd3f5d7)

**Datasets**

VeReMi dataset: Download link -> https://github.com/josephkamel/VeReMi-Dataset

Sensor dataset: Sensor dataset is provided in this repository named "Labeled_dataset_CPSS.csv"


**To run experiments**

First, run all the six models (DT, RF, DNN, KNN, SVM, AdaBoost) uploaded in the XAI_Frequency_Analysis repository with all the features and then also run DALEX_six_models. Then, reduce the features used for the AI models one by one and run the models. Store the results on the go. In this way, you can know how many times do the main features come first (the feature with most contribution) and do the frequency analysis using the following algortihm:

Calculate the final importance of each feature using a
weighted scoring system:
-- A feature receives 3 points for each first-place
ranking.
-- It receives 2 points for each second-place ranking.
-- It receives 1 point for each third-place ranking.
-- The formula used for final feature importance is:
Final Feature Importance Score = A × 3 + B × 2 +
C × 1, where:

∗ A = number of times the feature ranked first.

∗ B = number of times the feature ranked second.

∗ C = number of times the feature ranked third.


Use the final ranked list of features as input for independent classifiers (here, CatBoost, LR, and LGBM) to evaluate the effectiveness of the feature ensemble method in improving anomaly detection.



