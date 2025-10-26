# 🚀 CS210 Software Engineering Project

Welcome to our **CS210 Software Engineering Project**, conducted under the guidance of **Dr. Priodyuti Pradhan**.  
This project is a part of our coursework at IIIT Raichur, aiming to apply the principles of software engineering in a real-world team-based environment.

---

## 🌱 Our Project: DeepAgro

**DeepAgro** is a smart agriculture-based **Machine Learning project** designed to empower farmers with AI-driven tools.  
The project will include:

- 🌾 **Crop Recommendation System** – Suggesting the best crops based on soil and climate data.  
- 📈 **Yield Prediction** – Estimating expected yield for better planning.  
- 🍃 **Leaf Disease Prediction** – Detecting plant diseases using image recognition.  
- 🤖 **Farmer Chatbot** – Assisting farmers with queries in a user-friendly manner.  

---

## 📌 What Have We Done Till Now?

- ✅ Completed the **Feasibility Report** for the project - https://drive.google.com/drive/folders/1d_iCbjAw_72OWou1FsPdBFdg2EcPSrTW.  
- ✅ Defined the initial **requirements and scope**.  

---

## ⚙️ Software Engineering Model

For the development of **DeepAgro**, we have chosen the **Agile Model** to ensure iterative progress, adaptability, and continuous feedback.  

---

## 👥 Team Members

| Name                    | Roll Number   |
|-------------------------|---------------|
| Aditya Upendra Gupta    | AD24B1003     |
| Aaditya Awasthi         | CS24B1001     |
| Aditya Raj              | CS24B1004     |
| Sudhavalli Murali       | CS24B1057     |
| Kinshu Keshri           | AD24B1034     |
| Rishita                 | CS24B1021     |

### 📊 Overview
The Crop Recommendation System uses machine learning to suggest the most suitable crops based on soil parameters and environmental conditions. The system provides **top 3 crop recommendations** along with an input summary and intelligent warnings for outlier values.

### 📁 Dataset
We are using the **Crop Recommendation Dataset** from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset), which contains essential soil and environmental parameters.

### 🔍 Features Description

| Feature | Description | Unit | Typical Range |
|---------|-------------|------|---------------|
| **N** | Nitrogen content in soil | kg/ha | 0 - 140 |
| **P** | Phosphorus content in soil | kg/ha | 5 - 145 |
| **K** | Potassium content in soil | kg/ha | 5 - 205 |
| **Temperature** | Average temperature | °C | 8 - 44 |
| **Humidity** | Relative humidity | % | 14 - 100 |
| **pH** | Soil pH value | - | 3.5 - 9.9 |
| **Rainfall** | Annual rainfall | mm | 20 - 300 |

### 🤖 Machine Learning Models

We experimented with multiple classification algorithms to find the best performing model:

| Model | Accuracy | Status |
|-------|----------|--------|
| **XGBoost** | **99.3%** | ✅ **Selected** |
| Support Vector Machine (SVM) | 97.8% | - |
| Logistic Regression | 95.2% | - |
| K-Nearest Neighbors (KNN) | 97.3% | - |
| Decision Tree | 98.1% | - |

> **XGBoost** demonstrated superior performance and was chosen as the final model for deployment.

### ✨ Key Features

#### 🏆 Top 3 Crop Recommendations
The system provides the **top 3 most suitable crops** ranked by suitability score, giving farmers multiple options to choose from.

#### 📋 Input Summary
After analysis, users receive a comprehensive summary of their input parameters, helping them understand the conditions of their farmland.

#### ⚠️ Intelligent Warnings
The system includes smart validation to detect outlier values and provide warnings:

- ⚠️ **pH < 3.5 or pH > 10**: "Warning! pH value is extremely unusual. Please verify your soil test."
- ⚠️ **Negative values**: "Invalid input detected. Please enter positive values for nutrients."
- ⚠️ **Temperature extremes**: "Temperature value seems unusual for agricultural land."
- ⚠️ **Humidity extremes**: "Humidity percentage should be between 0-100%."

### 🎯 Example Output
```
🌾 Top 3 Recommended Crops:
1. Rice (Confidence: 94%)
2. Cotton (Confidence: 88%)
3. Maize (Confidence: 82%)

📊 Your Input Summary:
- Nitrogen: 85 kg/ha
- Phosphorus: 58 kg/ha
- Potassium: 41 kg/ha
- Temperature: 28°C
- Humidity: 75%
- pH: 6.5
- Rainfall: 180 mm

✅ All parameters are within normal range!
```

---

## 👥 Team Members

| Name | Roll Number |
|-------------------------|---------------|
| Aditya Upendra Gupta | AD24B1003 |
| Aaditya Awasthi | CS24B1001 |
| Aditya Raj | CS24B1004 |
| Sudhavalli Murali | CS24B1057 |
| Kinshu Keshri | AD24B1034 |
| Rishita | CS24B1021 |

---

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## 📈 Project Timeline

- **Phase 1**: Requirements & Feasibility ✅
- **Phase 2**: Module 1 - Crop Recommendation ✅
- **Phase 3**: Module 2 - Yield Prediction 🔄
- **Phase 4**: Module 3 - Leaf Disease Detection ⏳
- **Phase 5**: Module 4 - Farmer Chatbot ⏳
- **Phase 6**: Integration & Deployment ⏳

---

## 📞 Contact

For queries or suggestions, feel free to reach out to any team member or raise an issue in this repository.

---

<div align="center">
  
### 🌾 Empowering Farmers with AI 🌾

**Made with ❤️ by Team DeepAgro**

</div>

