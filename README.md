# Customer Happiness Prediction  

---

## Background  
We are one of the fastest-growing startups in the logistics and delivery domain. With a global expansion strategy, making customers happy is a top priority. Predicting customer happiness allows us to take targeted actions to improve satisfaction and operations.  

Gathering actionable customer feedback is challenging, but essential. This project utilizes survey data to predict customer happiness and identify factors driving satisfaction.  

---

## Data Description  
The dataset includes responses to six survey questions, with scores from 1 (lowest) to 5 (highest).  

**Target Variable:**  
- `Y`: Customer happiness (0 = unhappy, 1 = happy).  

**Features:**  
1. `X1`: My order was delivered on time.  
2. `X2`: Contents of my order were as I expected.  
3. `X3`: I ordered everything I wanted to order.  
4. `X4`: I paid a good price for my order.  
5. `X5`: I am satisfied with my courier.  
6. `X6`: The app makes ordering easy for me.  

---

## Goals and Success Metrics  

### **Goal:**  
Predict whether a customer is happy or unhappy based on survey responses.  

### **Success Metrics:**  
- Achieve an accuracy score of **73% or higher**.  
- Identify the **minimal set of features** preserving prediction performance.  

---

## Project Structure  

```plaintext
customer_happiness/  
│  
├── data/  
│   ├── raw/                   # Raw datasets  
│   ├── processed/             # Processed datasets  
│  
├── notebooks/  
│   ├── 1 data_exploration.ipynb    # Data exploration  
│   ├── 2 model_training.ipynb      # Model training  
│  
├── src/  
│   ├── __init__.py                 # Package initialization  
│   ├── data_exploration.py         # Data cleaning and transformation  
│   ├── feature_analysis.py         # Feature engineering and selection   
│   ├── utils.py                    # utils 
│
├── models/  
│   └── best_extratrees_model.pkl   # Trained models  
│  
├── requirements.txt                # Dependencies  
├── README.md                       # Project overview  
└── .gitignore                      # Files to ignore  
```

## Getting Started 

### Installation 
1. Clone the repository: ```bash git clone https://github.com/<your-username>/customer-happiness.git cd customer-happiness

### Install dependencies 

bash pip install -r requirements.txt
