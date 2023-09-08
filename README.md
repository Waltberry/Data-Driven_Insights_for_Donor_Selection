# Data-Driven_Insights_for_Donor_Selection
Data-Driven Insights for Donor Selection: A Machine Learning Approach

![larm-rmah-AEaTUnvneik-unsplash](https://github.com/Waltberry/Data-Driven_Insights_for_Donor_Selection/assets/63509339/b2a4ea3c-ca4b-41e7-b559-0ccc6981160e) credit to Photo by <a href="https://unsplash.com/@larm?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Larm Rmah</a> on <a href="https://unsplash.com/photos/AEaTUnvneik?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  


## Data-Driven Insights for Donor Selection: A Machine Learning Approach

You have been hired to deliver actionable insights to support your client, who is a national charitable organization. The client seeks to use the results of a previous postcard mail solicitation for donations to improve outcomes in the next campaign. You want to determine which individuals in their mailing database have characteristics similar to those of your most profitable donors. By soliciting only these people, your client can spend less money on the solicitation effort and more money on charitable concerns.

You have been provided two datasets:

- **Donor Raw Data**: This is historical data containing previous donor details. The "Target_B" column provides information about whether they have donated in the past campaign or not.

- **Prospective Donors**: This is a list of new contacts that your client is interested in reaching out to in the next campaign.

You are required to deliver a ten-page PowerPoint presentation with actionable recommendations to help the client identify which prospects they should focus their next campaign on.

### Data Preprocessing and Exploration

#### Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import scipy.stats as stats
from sklearn.feature_selection import chi2
from scipy.stats import chi2_contingency
```

#### Handling Missing Values
- Missing fields: Target_D, Donor_Age, Income_Group, Wealth_Rating, Months_Since_Last_Prom_Response.
- Possible techniques to handle missing values: Mean, median, or mode imputation, regression imputation, K-nearest neighbors (KNN) imputation, multiple imputation.

#### TARGET_D
```python
donor_raw_data['TARGET_D'].fillna(0, inplace=True)
```

#### DONOR_AGE
- Checked the distribution of Donor_age and chose median imputation.

#### INCOME_GROUP
- Chose mode imputation for handling missing values.

#### WEALTH_RATING
- Explored the distribution and performed custom imputation based on INCOME_GROUP.

#### MONTHS_SINCE_LAST_PROM_RESP
- Explored the distribution and removed rows with missing values as they were negligible.

#### Handling Irregular Values
- Dealt with irregular values in URBAN_CITY, SES, and CLUSTER_CODE.

#### Feature Engineering
- Created new features such as DONOR_AGE_GROUP, GIFT_RANGE, INCOME_TO_AGE_RATIO, and RESPONSE_RATE.

#### Feature Selection / Dimensionality Reduction
- Conducted correlation analysis and selected features with high correlation with the target variable.
- Performed univariate feature selection using chi-squared tests for categorical variables.

### Exploratory Data Analysis (EDA)

- Conducted EDA including pair plots and box plots to understand relationships and distributions.
- Visualized categorical features against the target variable to identify patterns.

### Building ML Model

- Prepared the data, including encoding categorical variables and splitting into train and test sets.
- Trained a Random Forest Classifier to predict donor response.
- Evaluated the model's accuracy and generated a classification report.
- Explored feature importance and visualized the results.

### Prospect Selection

- Used the trained model to predict responses for prospective donors.
- Ranked prospective donors by predicted response probability.
- Selected the top prospects for the next campaign and saved the results to a CSV file.

For a more detailed analysis and visualizations, please refer to the Python notebook. The PowerPoint presentation will provide actionable recommendations based on this analysis to help improve your client's campaign outcomes.

If you have any questions or need further information, please feel free to reach out to me at [onyero.ofuzim@eng.uniben.edu](mailto:onyero.ofuzim@eng.uniben.edu).


