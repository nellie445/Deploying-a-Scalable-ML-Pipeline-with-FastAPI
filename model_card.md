# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained on U.S. Census Income data to predict whether an individual's salary is greater than $50K annually.  
The project uses scikit-learn for preprocessing and model training, including categorical encoding and inference deployment through FastAPI.

## Intended Use

This model is intended for educational purposes as part of an ML pipeline deployment project.  
Its primary function is binary classification of income category (`>50K` or `<=50K`) based on census demographic and employment-related features.

It should not be used for real-world hiring, lending, or socioeconomic decision-making.

## Training Data

The training data comes from the UCI Census Income dataset (`census.csv`), which includes demographic and employment features such as:

- Age  
- Workclass  
- Education  
- Marital Status  
- Occupation  
- Race  
- Sex  
- Hours per week  
- Native Country  

Categorical variables were one-hot encoded before training.

## Evaluation Data

The dataset was split into training and test sets using an 80/20 train-test split with `random_state=42`.

The test set was used to evaluate:
- Precision  
- Recall  
- F1 Score  

Additionally, slice-based evaluation was performed across multiple categorical groups.

## Metrics

Model Performance:

- Precision: **0.7419**
- Recall: **0.6384**
- F1 Score: **0.6863**

These metrics indicate moderate predictive capability, with stronger precision than recall.

## Ethical Considerations

This model uses demographic and socioeconomic variables, including race, sex, and marital status, which may encode societal biases present in historical census data.

Potential ethical concerns include:

- Reinforcement of historical bias  
- Unequal predictive performance across demographic groups  
- Inappropriate use in high-stakes social or financial decisions  

Careful fairness analysis would be required before any practical deployment.

## Caveats and Recommendations

- This model was built for instructional purposes, not production use.  
- Performance may vary significantly across population subgroups.  
- The dataset may not represent current economic or social realities.  
- Additional fairness, bias, and calibration testing is recommended.  
- Hyperparameter tuning and cross-validation could improve performance.