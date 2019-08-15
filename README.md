# Comparative Study of Naive Bayes, KNN and Random Forest for Breast Cancer Prediction.

# Data Description
The Wisconsin Breast Cancer Diagnostic Datasets has a dimension of 569 columns and 32 rows (569 × 32), it has one numeric attribute such as id field and 30 real attributes and a class label, which is one categorical attribute. The dataset has two class
values for detection and diagnosis they are Benign (B) and Malignant (M) because it is a two class classification problem also known as binary classification. The class attribute is 212 Malignant and 357 Benign as the dataset does not contain any
missing values. All features values are computed and recorded in four significant digits.

# METHODOLOGY
There are three main steps involved in this
research work, they include:
A. Data collection.
B. Data cleaning and preparation.
C. Developing the model

A . Data Collection.
Data collection involves measuring and gathering of information on interested variables in a systematic fashion in other to establish and archive the research objectives and evaluate the outcomes. The objective of data collection is to capture quality
evidence that translates to rich data analysis and allows the implementation of a credible and convincing system. The Wisconsin Breast Cancer Diagnostic Dataset was obtained from UCI Machine Learning Data Repository.

B. Data Cleaning and Preparation.
This is a very important phase in processing and cleaning inconsistent and unstructured data. This phase is essential for the model development because good model is less important than good data. The dataset contains 33 features with 569 samples. The
last column is an empty column and it is of no use, hence we removed it. In the next phase, we separated label data and feature data. The feature data will be used for training the model and the label data use for diagnosis which says whether the tumor is Benign or Malignant.  We divided the data into training and testing where 80% of the total dataset points are used for training
sample, while 20% of the total sample dataset points are used for the testing sample.

C. Developing the model
The following models were used: Naive Bayes, KNN and Random Forest
