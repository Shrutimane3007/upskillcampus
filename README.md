# 🚦 Smart City Traffic Forecasting
An end-to-end Data Science and Machine Learning project that forecasts traffic volume using historical traffic data. The project applies data preprocessing, feature engineering, exploratory data analysis (EDA), and predictive modeling to identify traffic patterns and forecast future vehicle flow for smarter urban traffic management.

## 📌 Project Overview
Traffic congestion is one of the major challenges in smart cities. Accurate traffic forecasting helps transportation authorities optimize traffic signals, improve road planning, reduce congestion, and enhance commuter experience.
This project analyzes historical traffic data collected from multiple junctions and builds machine learning models capable of predicting future traffic volume.

## 🎯 Objectives
- Understand and analyze historical traffic data.
- Perform data cleaning and preprocessing.
- Extract meaningful features from date and time.
- Perform Exploratory Data Analysis (EDA).
- Visualize traffic trends and patterns.
- Train Machine Learning models for traffic forecasting.
- Evaluate model performance using appropriate metrics.
- Predict future traffic volume.

## 📂 Project Structure
Smart-City-Traffic-Forecasting/
│
├── data/
│   ├── raw/
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
├── requirements.txt
│
└── README.md
The trained model file (model.pkl) is not included because it exceeds GitHub's file size limit. Run the training notebook to regenerate the model.

## 📊 Dataset
The dataset contains hourly traffic information from multiple city junctions.

### Features
| Feature  | Description                  |
|----------|------------------------------|
| DateTime | Date and time of observation |
| Junction | Junction ID                  |
| Vehicles | Number of vehicles           |
| ID       | Unique record identifier     |

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- VS Code
- Git & GitHub

## 📈 Exploratory Data Analysis (Completed)
✔ Data Inspection
✔ Data Cleaning
✔ Missing Value Analysis
✔ Feature Engineering
✔ Traffic Distribution Analysis
✔ Hourly Traffic Analysis
✔ Monthly Traffic Analysis
✔ Daily Traffic Trend Analysis

## ⚙️ Feature Engineering
The following features were extracted from the `DateTime` column:
- Year
- Month
- Day
- Hour
- Day of Week
- Weekend Indicator
These engineered features help capture temporal traffic patterns.

## 📉 Current Progress
- [x] Project Setup
- [x] Environment Configuration
- [x] Data Understanding
- [x] Data Cleaning
- [x] Feature Engineering
- [x] Exploratory Data Analysis
- [x] Data Preprocessing
- [x] Model Building
- [x] Model Evaluation
- [x] Hyperparameter Tuning
- [x] Traffic Forecasting
- [x] Model Deployment

## 🚀 Future Improvements
- Compare multiple forecasting models.
- Hyperparameter optimization.
- Build an interactive dashboard.
- Deploy the model using Streamlit or Flask.
- Real-time traffic prediction using live APIs.


## 👨‍💻 Author
"Shruti Mane"
Bachelor of Engineering (Computer Engineering)
Aspiring Data Scientist & Machine Learning Engineer

⭐ If you found this project useful, feel free to star the repository.
