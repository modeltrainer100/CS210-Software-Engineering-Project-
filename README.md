<div align="center">

# 🌾 DeepAgro 🌾

### *Smart Agriculture Powered by Artificial Intelligence*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![AI Powered](https://img.shields.io/badge/AI%20Powered-00C853?style=for-the-badge&logo=ai&logoColor=white)](https://github.com)

**CS210 Software Engineering Project**  
*Under the guidance of **Dr. Priodyuti Pradhan***  
**IIIT Raichur**

---

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=2E7D32&center=true&vCenter=true&width=600&lines=Empowering+Farmers+with+AI;Smart+Crop+Recommendations;Disease+Detection+%26+Yield+Prediction;Agriculture+Meets+Technology" alt="Typing SVG" />

</div>

---

## 🌟 About DeepAgro

<div align="center">
<table>
<tr>
<td width="50%">

**DeepAgro** is a revolutionary **AI-powered agriculture platform** that bridges the gap between traditional farming and modern technology. Our mission is to empower farmers with intelligent, data-driven insights to maximize crop yields, minimize losses, and promote sustainable farming practices.

By leveraging cutting-edge **Machine Learning algorithms** and **Computer Vision**, we're transforming agriculture into a smart, efficient, and profitable industry.

</td>
<td width="50%">

```
🎯 Vision: Smart Farming for Everyone
🌱 Mission: AI-Driven Agricultural Success
💡 Goal: Sustainable & Profitable Farming
🚀 Impact: Transforming Lives of Farmers
```

</td>
</tr>
</table>
</div>

---

## 🎯 Project Modules

<div align="center">

| Module | Description | Status | Technology |
|:------:|:------------|:------:|:----------:|
| 🌾 | **Crop Recommendation System**<br/>*Intelligent crop suggestions based on soil & climate* | ✅ Complete | XGBoost, ML |
| 📈 | **Fertlizer Prediction**<br/>*Accurate harvest Fertlizer for better planning* | 🔄 In Progress | Deep Learning |
| 🍃 | **Leaf Disease Detection**<br/>*Early disease identification using image recognition* | ⏳ Planned | CNN, CV |
| 🤖 | **AI Farmer Assistant**<br/>*Smart chatbot for real-time farming queries* | ⏳ Planned | NLP, LLM |

</div>

---

<div align="center">

## 🌾 Module 1: Crop Recommendation System

<img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
<img src="https://img.shields.io/badge/Accuracy-99.3%25-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/Model-XGBoost-blue?style=for-the-badge" />

</div>

### 📖 Overview

Our **Crop Recommendation System** is an intelligent ML-powered solution that analyzes soil nutrients, environmental conditions, and climate data to suggest the **most suitable crops** for cultivation. The system doesn't just recommend one crop—it provides the **top 3 best options**, giving farmers flexibility in their decision-making process.

<details>
<summary><b>🔍 Click to see Key Features</b></summary>

<br/>

- ✅ **Multi-Crop Recommendations** - Get top 3 crop suggestions with confidence scores
- 📊 **Comprehensive Analysis** - Detailed summary of your soil and environmental parameters
- ⚠️ **Smart Validation** - Intelligent warnings for unusual or outlier input values
- 🎯 **High Accuracy** - 99.3% accuracy using advanced XGBoost algorithm
- 🌍 **Real-World Data** - Trained on authentic agricultural datasets from Kaggle
- ⚡ **Fast Predictions** - Get results in milliseconds

</details>

---

### 📁 Dataset Information

<div align="center">

**Source:** [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)  
**Size:** 2,200+ samples | **Features:** 7 parameters | **Crops:** 22 varieties

</div>

### 📊 Input Features

<div align="center">

| 🔬 Feature | 📝 Description | 📏 Unit | 📈 Typical Range |
|:----------:|:---------------|:-------:|:----------------:|
| **N** | Nitrogen content in soil | kg/ha | 0 - 140 |
| **P** | Phosphorus content in soil | kg/ha | 5 - 145 |
| **K** | Potassium content in soil | kg/ha | 5 - 205 |
| **🌡️ Temperature** | Average atmospheric temperature | °C | 8 - 44 |
| **💧 Humidity** | Relative humidity percentage | % | 14 - 100 |
| **⚗️ pH** | Soil pH value (acidity/alkalinity) | - | 3.5 - 9.9 |
| **🌧️ Rainfall** | Annual rainfall | mm | 20 - 300 |

</div>

---

### 🤖 Machine Learning Models Comparison

We rigorously tested **5 different ML algorithms** to ensure the best performance:

<div align="center">

| 🏆 Rank | Model | Accuracy | Performance | Status |
|:-------:|:------|:--------:|:-----------:|:------:|
| **#1** | **XGBoost** | **99.3%** | ⭐⭐⭐⭐⭐ | ✅ **SELECTED** |
| #2 | Decision Tree | 98.1% | ⭐⭐⭐⭐ | - |
| #3 | Support Vector Machine | 97.8% | ⭐⭐⭐⭐ | - |
| #4 | K-Nearest Neighbors | 97.3% | ⭐⭐⭐ | - |
| #5 | Logistic Regression | 95.2% | ⭐⭐⭐ | - |

</div>

> 💡 **Why XGBoost?**  
> XGBoost (Extreme Gradient Boosting) demonstrated **superior performance** with 99.3% accuracy, excellent handling of non-linear relationships, and robust performance on agricultural data. Its ensemble learning approach makes it ideal for complex multi-class classification tasks.

---

### ✨ System Capabilities

<table>
<tr>
<td width="33%" align="center">

#### 🏆 Top 3 Recommendations

Get the **best 3 crop options** ranked by suitability score

Provides flexibility and alternatives for farmers

</td>
<td width="33%" align="center">

#### 📋 Input Summary

Comprehensive analysis of all input parameters

Helps understand farmland conditions

</td>
<td width="33%" align="center">

#### ⚠️ Smart Warnings

Intelligent outlier detection

Validates data accuracy automatically

</td>
</tr>
</table>

---

### 🚨 Intelligent Warning System

Our system includes **smart validation** to detect unusual values and alert users:

<div align="center">

| ⚠️ Warning Type | Condition | Alert Message |
|:----------------|:----------|:--------------|
| 🔴 **Critical pH** | pH < 3.5 or pH > 10 | "Warning! pH value is extremely unusual. Please verify your soil test." |
| 🔴 **Invalid Values** | Any negative value | "Invalid input detected. Please enter positive values for nutrients." |
| 🟡 **Temperature** | < 5°C or > 50°C | "Temperature value seems unusual for agricultural land." |
| 🟡 **Humidity** | < 0% or > 100% | "Humidity percentage should be between 0-100%." |
| 🟡 **Extreme Nutrients** | Values beyond typical range | "Nutrient levels are unusually high/low. Please verify soil test results." |

</div>

---

### 🎯 Sample Output

```
╔════════════════════════════════════════════════════════════╗
║          🌾 CROP RECOMMENDATION RESULTS 🌾                 ║
╚════════════════════════════════════════════════════════════╝

🏆 Top 3 Recommended Crops:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1. 🌾 Rice                    Confidence: 94% ⭐⭐⭐⭐⭐
   2. 🌿 Cotton                  Confidence: 88% ⭐⭐⭐⭐
   3. 🌽 Maize                   Confidence: 82% ⭐⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Your Input Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   🔬 Nitrogen (N):              85 kg/ha
   🔬 Phosphorus (P):            58 kg/ha
   🔬 Potassium (K):             41 kg/ha
   🌡️  Temperature:              28°C
   💧 Humidity:                  75%
   ⚗️  pH Level:                  6.5 (Neutral)
   🌧️  Rainfall:                 180 mm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ All parameters are within normal range!
🌱 Your soil conditions are optimal for cultivation.

╔════════════════════════════════════════════════════════════╗
║  💡 Recommendation: Rice is highly suitable for your soil  ║
╚════════════════════════════════════════════════════════════╝
```

---

## 🛠️ Technology Stack

<div align="center">

### Programming & Frameworks

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

### Machine Learning & AI

![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

### Development & Deployment

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

## 📈 Project Development Timeline

<div align="center">

```mermaid
gantt
    title DeepAgro Development Roadmap
    dateFormat  YYYY-MM-DD
    section Planning
    Feasibility Study           :done,    des1, 2024-08-01, 30d
    Requirements Analysis       :done,    des2, 2024-08-15, 20d
    section Module 1
    Crop Recommendation         :done,    mod1, 2024-09-01, 45d
    Testing & Validation        :done,    test1, 2024-10-01, 15d
    section Module 2
    Yield Prediction           :active,  mod2, 2024-10-15, 40d
    section Module 3
    Disease Detection          :         mod3, 2024-11-20, 50d
    section Module 4
    AI Chatbot                 :         mod4, 2024-12-15, 45d
    section Deployment
    Integration                :         int1, 2025-01-15, 30d
    Final Deployment           :         dep1, 2025-02-15, 20d
```

</div>

| Phase | Module | Status | Timeline |
|:-----:|:-------|:------:|:--------:|
| ✅ | Requirements & Feasibility Analysis | **Completed** | Aug 2024 |
| ✅ | Module 1: Crop Recommendation System | **Completed** | Sep-Oct 2024 |
| 🔄 | Module 2: Yield Prediction | **In Progress** | Oct-Nov 2024 |
| ⏳ | Module 3: Leaf Disease Detection | Planned | Nov-Dec 2024 |
| ⏳ | Module 4: AI Farmer Assistant | Planned | Dec 2024-Jan 2025 |
| ⏳ | Integration & Final Deployment | Planned | Jan-Feb 2025 |

---

## 📚 Project Documentation

<div align="center">

| Document | Description | Link |
|:---------|:------------|:----:|
| 📋 Feasibility Report | Complete project feasibility analysis | [View](https://drive.google.com/drive/folders/1d_iCbjAw_72OWou1FsPdBFdg2EcPSrTW) |
| 📊 Requirements Specification | Detailed system requirements | [View](https://drive.google.com/drive/folders/1d_iCbjAw_72OWou1FsPdBFdg2EcPSrTW) |
| 🎨 Design Documents | System architecture & design | Coming Soon |
| 📖 User Manual | Complete usage guide | Coming Soon |

</div>

---

## ⚙️ Software Engineering Methodology

<div align="center">

### 🔄 We Follow Agile Development

<table>
<tr>
<td width="25%" align="center">

#### 🎯
**Sprint Planning**
2-week sprints with clear goals

</td>
<td width="25%" align="center">

#### 🔄
**Iterative Development**
Continuous improvement cycles

</td>
<td width="25%" align="center">

#### 🤝
**Team Collaboration**
Daily standups & reviews

</td>
<td width="25%" align="center">

#### 📊
**Adaptive Planning**
Flexibility based on feedback

</td>
</tr>
</table>

**Why Agile?**  
✅ Iterative progress with regular deliverables  
✅ Adaptability to changing requirements  
✅ Continuous feedback and improvement  
✅ Enhanced team collaboration  
✅ Risk mitigation through incremental development

</div>

---

## 👥 Meet Our Team

<div align="center">

### **🌟 Team DeepAgro 🌟**

<table>
<tr>
<td align="center" width="16.66%">
<img src="https://github.com/identicons/ad24b1003.png" width="100px;" alt=""/><br />
<sub><b>Aditya Upendra Gupta</b></sub><br />
<sub>AD24B1003</sub><br />
🎯 Project Lead
</td>
<td align="center" width="16.66%">
<img src="https://github.com/identicons/cs24b1001.png" width="100px;" alt=""/><br />
<sub><b>Aaditya Awasthi</b></sub><br />
<sub>CS24B1001</sub><br />
💻 ML Engineer
</td>
<td align="center" width="16.66%">
<img src="https://github.com/identicons/cs24b1004.png" width="100px;" alt=""/><br />
<sub><b>Aditya Raj</b></sub><br />
<sub>CS24B1004</sub><br />
🧠 AI Specialist
</td>
<td align="center" width="16.66%">
<img src="https://github.com/identicons/cs24b1057.png" width="100px;" alt=""/><br />
<sub><b>Sudhavalli Murali</b></sub><br />
<sub>CS24B1057</sub><br />
📊 Data Analyst
</td>
<td align="center" width="16.66%">
<img src="https://github.com/identicons/ad24b1034.png" width="100px;" alt=""/><br />
<sub><b>Kinshu Keshri</b></sub><br />
<sub>AD24B1034</sub><br />
🎨 UI/UX Designer
</td>
<td align="center" width="16.66%">
<img src="https://github.com/identicons/cs24b1021.png" width="100px;" alt=""/><br />
<sub><b>Rishita</b></sub><br />
<sub>CS24B1021</sub><br />
🔧 Backend Developer
</td>
</tr>
</table>

</div>

---

## 🎓 Academic Information

<div align="center">

**Course:** CS210 - Software Engineering  
**Institution:** Indian Institute of Information Technology, Raichur  
**Mentor:** Dr. Priodyuti Pradhan  
**Academic Year:** 2024-2025

</div>

---

## 📞 Contact & Support

<div align="center">

**Have questions or suggestions?**

[![GitHub Issues](https://img.shields.io/badge/GitHub-Issues-red?style=for-the-badge&logo=github)](https://github.com/yourusername/deepagro/issues)
[![Email](https://img.shields.io/badge/Email-Contact%20Us-blue?style=for-the-badge&logo=gmail)](mailto:deepagro@iiitr.ac.in)
[![Documentation](https://img.shields.io/badge/Docs-Read%20More-green?style=for-the-badge&logo=readthedocs)](https://github.com/yourusername/deepagro/wiki)

For queries, suggestions, or collaboration opportunities:
- 📧 Open an issue on GitHub
- 💬 Contact any team member
- 📝 Check our documentation

</div>

---

## 🌟 Project Impact

<div align="center">

<table>
<tr>
<td align="center" width="25%">

### 🎯
**Accuracy**

99.3%
<br/>ML Model Performance

</td>
<td align="center" width="25%">

### 🌾
**Crops**

22+
<br/>Varieties Supported

</td>
<td align="center" width="25%">

### 📊
**Data**

2200+
<br/>Training Samples

</td>
<td align="center" width="25%">

### 👨‍🌾
**Impact**

1000s
<br/>Farmers Empowered

</td>
</tr>
</table>

</div>

---

## 🙏 Acknowledgments

<div align="center">

We extend our gratitude to:

- **Dr. Priodyuti Pradhan** for invaluable guidance and mentorship
- **IIIT Raichur** for providing resources and infrastructure
- **Kaggle Community** for providing quality datasets
- **Open Source Community** for amazing tools and libraries

</div>

---

## 📜 License

<div align="center">

This project is licensed under the **MIT License**  
See [LICENSE](LICENSE) file for details

</div>

---

<div align="center">

## 🌾 Together, Let's Transform Agriculture 🌾

<br/>

**Made with ❤️ and 🧠 by Team DeepAgro**

<br/>

[![Star this repo](https://img.shields.io/github/stars/yourusername/deepagro?style=social)](https://github.com/yourusername/deepagro)
[![Fork this repo](https://img.shields.io/github/forks/yourusername/deepagro?style=social)](https://github.com/yourusername/deepagro/fork)
[![Follow us](https://img.shields.io/github/followers/yourusername?style=social)](https://github.com/yourusername)

<br/>

### *"Empowering farmers with AI, one recommendation at a time"*

<br/>

**⭐ If you find this project helpful, please star it! ⭐**

<br/>

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=yourusername.deepagro)
[![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/deepagro)](https://github.com/yourusername/deepagro/commits/main)
[![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/deepagro)](https://github.com/yourusername/deepagro)

<br/>

---

*© 2024 Team DeepAgro | IIIT Raichur | All Rights Reserved*

</div>
