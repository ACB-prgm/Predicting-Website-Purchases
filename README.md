# Phase 3 Project
## Maximize Website Revenue by Predicting Users' Purchases


### Introduction
The purpose of this notebook is to simulate solving a real world machine learning classification problem.  I use several ensemble tree models (XGBoost and Light GBM) to predict whether a user's activity on a company website will translate to a sale with ~94% accuracy and good precision.  This would allow for the comapny to optimize their website, better target their marketing, and incorporate dynamic pricing into their site in order to boost revenue.

### Resources
• [Jupyter Notebook / Code](https://github.com/ACB-prgm/Phase3Project.nosync/blob/master/student.ipynb)
• [Presentation](https://github.com/ACB-prgm/Phase3Project.nosync/blob/master/Phase_3_Presentation.pdf)
• [Data](https://github.com/ACB-prgm/Phase3Project.nosync/tree/master/Data)
• [Kaggle Page](https://www.kaggle.com/competitions/online-purchase-prediction/data?select=shop_train.csv)
• [Final Kaggle Submission](https://github.com/ACB-prgm/Phase3Project.nosync/commit/3bc555b94ccaea46d81b57e55b0b8e43603d21ef)

### Data
Each row represents a user's session on an online store, with 6165 user sessions in total (~15% of which resulted in purchases).  
The following are the column/feature descriptions:

`admin_pages`, `info_pages`, `product_pages` - number of pages in different categories visited by the user  
`admin_seconds`, `info_seconds`, `product_seconds` - time spent by the user on different page categories  
`page_value`, `bounce_rate`, `quit_rate` - numbers from Google Analytics  
`is_holiday` - the proximity of important days for retail (such as the New Year)  
`month` - month (categorical variable)  
`operating_system_id`, `browser_id`, `region_id`, `traffic_type_id` are also categorical variables, although they are written as numbers  
`is_new_visitor`, `is_weekend` - binary signs  
`has_purchase` - binary attribute, target variable. It is he who needs to learn to predict.

### Methods
#### Previewing the Data
I first inspected the data for abnormalities including Nan values, outliers, odd distributions, etc, and found that this was a fairly clean dataset.  The exception was that the there was a large class imbalance as most users will not purchase anything.  This is part of the reason why I chose to use ensemble methods as they are robust to class imbalances.  
#### Data Preparation and Pipeline Construction
Then I split the data into training and validation data, and began creating my preprocessing pipeline. I created a custom function that would convert the month variable to a numerical and ordinal variable, and added the rows `total_time` and `avg_time`, which represent the total amount of time the user spent on the website and the average amount of time they spent in the three categories respectively. Next, I used sklearn's standard scaler to scale the entire dataset to ensure the model would interpret the features on the same scale.  Finally I added a classifier, first XGBoost, then Light GBM.
#### Model Training and Hyperparameter Tuning
I fit the pipeline on the training data and scored it on the validation data, looking at the accuracy, f1-score, ROC-AUC, and precision as metrics.  I noticed that the model was overfitting by observing a near perfect performance on the training data and considerably worse performance on the validation data.  I also noticed that the precision and f1-scores were very low.  Thus, I next performed a grid search and selected hyperparameters to affect these issues.  `n_estimators` and `max_depth` to reduce overfitting, and `scale_pos_weight` to balance the weight of the positive class (minority class) and increase precision. I then scored the model by submitting the model's predictions to kaggle.  I repeated the process with the Light GBM model and achieved slightly better results.

### Conclusion
LGBM was considerably faster to train and slightly more performant than XGBoost in this scenario.  This approach proved effective and can be used for real world prediction of purchases as well as other website and application end points.
