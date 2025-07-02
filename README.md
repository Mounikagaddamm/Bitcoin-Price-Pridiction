# 📈 Bitcoin Price Prediction using Machine Learning

## 🧩 Problem Statement

The aim of this project is to predict whether the **closing price of Bitcoin** will rise or fall the next day based on historical market data. This binary classification task leverages statistical features derived from price movements to train machine learning models.


## 📊 Dataset

* **Source**: bitcoin.csv (assumed to be daily historical Bitcoin data)
* **Features Used**:

  * Open, High, Low, Close prices
  * Engineered features:

    * open-close (difference between open and close prices)
    * low-high (range of prices in a day)
    * is_quarter_end (whether the day is at a quarter end)


## 🧠 Model Details

Three models were trained and evaluated:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)** with polynomial kernel
3. **XGBoost Classifier**

The target is binary:

* 1: Closing price increases the next day
* 0: Closing price decreases or stays the same

All features were standardized using StandardScaler, and data was split 70% train / 30% test.



## ✅ Results & Accuracy

 Model               --  Train AUC Score--  Validation AUC Score 
 
 Logistic Regression -- \~0.61          --  \~0.58               
 SVM (Poly Kernel)   -- \~0.61          --  \~0.58               
 XGBoost Classifier  -- \~0.62          --  \~0.58               

* Confusion matrix and ROC-AUC score were used for evaluation. The model shows a slight predictive edge over random guessing, suggesting room for improvement through deeper feature engineering or advanced time-series modeling.


## 🛠️ How to Run the Code

1. Clone the repository or download the .ipynb / .py file.

2. Install required libraries (preferably in a virtual environment):


   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   

3. Ensure the bitcoin.csv dataset is in the working directory or update the path in the code:

   df = pd.read_csv('bitcoin.csv')
   

4. Run the script:

   python bitcoin_price_prediction_using_machine_learning_in_python.py
   

5. View the model outputs, accuracy scores, and confusion matrix visualizations.




