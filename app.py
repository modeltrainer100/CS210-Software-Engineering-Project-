import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import time 
import io
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import time
import pickle 
import joblib
import requests
import json
from sklearn.compose import ColumnTransformer
translations = {
  "en": {
    "page_title": "DeepAgro - Smart Agriculture",
    "sidebar_title": "🌾 Navigation",
    "nav_home": "🏠 Home",
    "nav_crop": "🌾 Crop Prediction",
    "nav_fertilizer": "🧪 Fertilizer Recommendation",
    "nav_disease": "🔬 Disease Detection",
    "nav_chat": "🤖 DeepAgro AI Assistant",
    "nav_about": "👥 About Us",
    "home": {
      "header_logo": "🌱 DeepAgro",
      "header_tagline": "Smart Agriculture Solutions with AI & ML",
      "welcome_header": "🌟 Welcome to the Future of Agriculture!",
      "welcome_text": "DeepAgro leverages cutting-edge **Machine Learning** and **Artificial Intelligence** to revolutionize farming practices. Our platform provides intelligent insights for:",
      "card_crop_title": "🌾 Smart Crop Recommendation",
      "card_crop_desc": "Get personalized crop suggestions based on soil conditions, climate, and nutrients using advanced ML algorithms.",
      "card_fert_title": "🧪 Fertilizer Optimization",
      "card_fert_desc": "Receive precise fertilizer recommendations to maximize yield while minimizing environmental impact.",
      "card_disease_title": "🔬 AI-Powered Disease Detection",
      "card_disease_desc": "Upload leaf images for instant disease identification using state-of-the-art CNN deep learning models.",
      "metrics_header": "🚀 Key Features",
      "metric_crops": "Crop Types",
      "metric_fertilizers": "Fertilizer Types",
      "metric_accuracy": "Accuracy",
      "metric_power": "Powered",
      "why_choose_title": "🌟 Why Choose DeepAgro?",
      "why_choose_desc": "Experience the future of farming with our cutting-edge AI technology that transforms traditional agriculture into smart, data-driven decisions for maximum yield and sustainability.",
      "benefit_precision_title": "Precision Agriculture",
      "benefit_precision_desc": "Make data-driven decisions with pinpoint accuracy for optimal crop selection and resource management.",
      "benefit_sustain_title": "Sustainable Farming",
      "benefit_sustain_desc": "Reduce waste and environmental impact while maximizing productivity through intelligent recommendations.",
      "benefit_realtime_title": "Real-time Analysis",
      "benefit_realtime_desc": "Get instant insights and predictions powered by advanced machine learning algorithms and computer vision."
    },
    "crop_prediction": {
      "main_title": "🌾 Intelligent Crop Recommendation System",
      "subtitle": "Get AI-powered crop suggestions based on your soil and environmental conditions.",
      "expander_header": "ℹ️ Understanding Crop Prediction Parameters",
      "expander_info_text": "Our AI model analyzes multiple factors to recommend the best crops for your land. Each parameter plays a crucial role in determining crop suitability:",
      "how_it_works": "📊 **How it works:** Our machine learning algorithm processes your input data and compares it with thousands of successful crop combinations to provide personalized recommendations with confidence scores.",
      "env_factors_header": "🌡️ Environmental Factors",
      "temp_label": "🌡️ Temperature (°C)",
      "temp_info": "<strong>Temperature Impact:</strong> Ambient temperature in degrees Celsius. Different crops thrive in different temperature ranges - tropical crops prefer 25-35°C while temperate crops prefer 15-25°C.",
      "hum_label": "💧 Humidity (%)",
      "hum_info": "<strong>Humidity Impact:</strong> Relative humidity percentage in the air. High humidity (>70%) suits crops like rice, while low humidity (<50%) is better for crops like wheat and barley.",
      "rain_label": "🌧️ Rainfall (mm)",
      "rain_info": "<strong>Rainfall Impact:</strong> Average rainfall amount in millimeters. Rice needs 150-300mm, wheat needs 30-100mm, while drought-resistant crops can survive with <50mm.",
      "ph_label": "⚗️ Soil pH Level",
      "ph_info": "<strong>pH Impact:</strong> Soil pH value measuring acidity/alkalinity. Most crops prefer 6.0-7.5 (slightly acidic to neutral). Acidic soils (<6) suit blueberries, while alkaline soils (>7.5) suit asparagus.",
      "nutrients_header": "🧪 Soil Nutrients (NPK Values)",
      "n_label": "🔵 Nitrogen (N) Content",
      "n_info": "<strong>Nitrogen (N) Role:</strong> Essential for leaf growth and chlorophyll production. Leafy vegetables need high N (80-120), while root vegetables need moderate N (40-80).",
      "p_label": "🟡 Phosphorus (P) Content",
      "p_info": "<strong>Phosphorus (P) Role:</strong> Vital for root development and flowering. Fruit crops need high P (60-100), while grasses need lower P (20-40).",
      "k_label": "🔴 Potassium (K) Content",
      "k_info": "<strong>Potassium (K) Role:</strong> Important for disease resistance and water regulation. Root vegetables and fruits need high K (80-150), while cereals need moderate K (40-80).",
      "summary_header": "📊 Current Input Summary",
      "summary_temp": "🌡️ **Temperature:**",
      "summary_hum": "💧 **Humidity:**",
      "summary_rain": "🌧️ **Rainfall:**",
      "summary_ph": "⚗️ **pH Level:**",
      "summary_n": "🔵 **Nitrogen (N):**",
      "summary_p": "🟡 **Phosphorus (P):**",
      "summary_k": "🔴 **Potassium (K):**",
      "reference_header": "📋 Ideal Ranges Reference",
      "ref_text": "<strong>Optimal Growing Conditions:</strong><br>• **Temperature:** 20-30°C (most crops)<br>• **Humidity:** 40-70% (optimal range)<br>• **Rainfall:** 50-200mm (varies by crop)<br>• **pH:** 6.0-7.5 (neutral to slightly acidic)<br>• **NPK:** Balanced ratios for healthy growth",
      "warning_temp": "🌡️ Temperature is outside typical growing range (5-45°C)",
      "warning_hum": "💧 Humidity levels may be challenging for most crops",
      "warning_ph": "⚗️ pH level is quite extreme and may limit crop options",
      "warning_n": "🔵 Very high nitrogen levels may cause excessive vegetative growth",
      "warning_p": "🟡 High phosphorus levels may interfere with other nutrient uptake",
      "warning_k": "🔴 Very high potassium levels may affect soil structure",
      "warnings_header": "⚠️ Input Warnings:",
      "validation_header": "✅ Validation Status",
      "validation_text": "All input values are within acceptable ranges! Your conditions look great for crop cultivation.",
      "predict_button": "🔮 Predict Best Crop",
      "loading_1": "Analyzing soil conditions...",
      "loading_2": "Processing environmental data...",
      "loading_3": "Matching with crop database...",
      "loading_4": "Finalizing recommendations...",
      "result_header": "🎯 Recommended Crop:",
      "result_confidence": "📊 Confidence Score:",
      "result_quality": "🌟 Match Quality:",
      "quality_excellent": "Excellent",
      "quality_good": "Good",
      "quality_fair": "Fair",
      "top_3_header": "📈 Top 3 Crop Recommendations",
      "crop_season": "Season",
      "crop_water": "Water Needs",
      "crop_match": "Match",
      "crop_suitability": "Suitability",
      "personalized_tips_header": "💡 Personalized Farming Tips",
      "tips_climate_header": "🌡️ Climate Considerations",
      "tips_temp_high": "<strong>🌡️ High Temperature Alert:</strong> Consider heat-resistant varieties, shade nets, and frequent irrigation scheduling. Install drip irrigation for water efficiency.",
      "tips_temp_low": "<strong>❄️ Cool Temperature:</strong> Ideal for cool-season crops. Consider frost protection measures like row covers and greenhouse cultivation.",
      "tips_temp_ok": "<strong>🌡️ Optimal Temperature:</strong> Perfect conditions for most crop varieties. Maintain consistent watering and monitor for pests.",
      "tips_hum_high": "<strong>💧 High Humidity Warning:</strong> Ensure proper plant spacing and ventilation to prevent fungal diseases. Consider fungicide treatments.",
      "tips_hum_low": "<strong>🏜️ Low Humidity Alert:</strong> Consider mulching and frequent light watering to maintain soil moisture. Use humidity retention techniques.",
      "tips_hum_ok": "<strong>💧 Good Humidity Levels:</strong> Favorable conditions for healthy plant growth. Monitor for optimal plant development.",
      "tips_soil_header": "🧪 Soil Management",
      "tips_ph_acidic": "<strong>⚗️ Acidic Soil:</strong> Consider adding lime to raise pH. Test for aluminum toxicity and add organic matter to improve soil structure.",
      "tips_ph_alkaline": "<strong>⚗️ Alkaline Soil:</strong> Consider adding sulfur or organic matter to lower pH. Monitor for micronutrient deficiencies.",
      "tips_ph_ok": "<strong>⚗️ Optimal pH Range:</strong> Perfect conditions for nutrient availability. Maintain soil health with regular organic amendments.",
      "tips_n_low": "<strong>🔵 Low Nitrogen:</strong> Consider nitrogen-rich fertilizers like urea or organic compost. Apply in split doses for better uptake.",
      "tips_n_high": "<strong>🔵 High Nitrogen:</strong> May cause excessive vegetative growth. Monitor carefully and reduce nitrogen input if needed.",
      "tips_p_low": "<strong>🟡 Low Phosphorus:</strong> Consider DAP or rock phosphate application. Essential for root development and flowering.",
      "tips_k_low": "<strong>🔴 Low Potassium:</strong> Consider MOP (Muriate of Potash) application. Important for disease resistance and water regulation.",
      "summary_box_header": "🌟 Your Personalized Crop Recommendation Summary",
      "summary_box_text": "Based on our AI analysis of your soil and environmental conditions, <strong>{}</strong> is the most suitable crop for your land with a <strong>{:.1f}% confidence score</strong>.",
      "summary_match_quality": "🎯 Match Quality",
      "summary_growth_potential": "🌱 Growth Potential",
      "summary_econ_viability": "💰 Economic Viability",
      "growth_high": "High",
      "growth_medium": "Medium",
      "growth_moderate": "Moderate",
      "econ_prof": "Profitable",
      "econ_good": "Good"
    },
    "fertilizer_recommendation": {
      "main_title": "🧪 Fertilizer Recommendation System",
      "subtitle": "Get optimal fertilizer suggestions based on your crop and soil conditions.",
      "section_info": "🌱 Crop & Soil Information",
      "section_env": "🌡️ Environmental Conditions",
      "section_nutrients": "🧪 Current Soil Nutrients",
      "crop_type_label": "Crop Type",
      "soil_type_label": "Soil Type",
      "temp_label": "Temperature (°C)",
      "hum_label": "Humidity (%)",
      "moisture_label": "Soil Moisture (%)",
      "nitrogen_label": "Nitrogen Content",
      "phosphorus_label": "Phosphorus Content",
      "potassium_label": "Potassium Content",
      "nutrient_status_header": "📊 Nutrient Status",
      "low": "🔴 Low",
      "medium": "🟡 Medium",
      "high": "🟢 High",
      "predict_button": "💡 Get Fertilizer Recommendation",
      "result_header": "🎯 Recommended Fertilizer:",
      "result_confidence": "📊 Confidence:",
      "result_info_pre": "For ",
      "result_info_in": " soil with:",
      "result_info_apply": "- Apply **{}** fertilizer",
      "result_info_tips": "- Consider current nutrient levels when determining quantity\n- Apply during the appropriate growth stage\n- Monitor soil moisture and weather conditions",
      "error_message": "Error in prediction. Please check your inputs."
    },
    "disease_detection": {
      "main_title": "🔬 Disease Detection",
      "subtitle": "Instantly identify disease from a leaf image using deep learning CNN models.",
      "upload_header": "📷 Upload Plant Leaf Image",
      "upload_guidelines_title": "📸 Image Upload Guidelines:",
      "upload_guidelines_text": "✓ Clear, well-lit leaf images<br>✓ Focus on affected areas or symptoms<br>✓ Supported formats: JPG, PNG, JPEG<br>✓ Maximum size: 10MB",
      "file_uploader_label": "Choose a leaf image...",
      "file_uploader_help": "Upload a clear image of a plant leaf",
      "uploaded_image_caption": "📷 Uploaded Leaf Image",
      "analyze_button": "🔍 Analyze for Diseases",
      "loading_message": "🧠 AI is analyzing the image...",
      "analysis_complete": "Analysis complete!",
      "result_header": "🎯 Predicted Disease:",
      "result_confidence": "📊 Confidence:",
      "disease_warning": "❗ Your plant may be diseased. Please consult a professional for confirmation.",
      "healthy_message": "✅ Plant appears to be healthy!"
    },
    "chatbot_page": {
        "main_title": "🤖 DeepAgro Chatbot - AI Agriculture Assistant",
        "subtitle": "Ask any question related to farming, crops, soil, or fertilizers!",
        "initial_message": "Hello! I am DeepAgro Chatbot, your AI agriculture assistant. How can I help you with your farm today?",
        "user_placeholder": "Ask your agriculture question here...",
        "send_button": "Send 🚀",
        "loading": "DeepChat is thinking...",
        "error": "Sorry, I encountered an error. Please try again or rephrase your question."
    },
    "about_page": {
      "main_title": "👥 About Us",
      "subtitle": "Meet the innovative team behind the smart agriculture revolution!",
      "mission_title": "🌟 Our Mission",
      "mission_text": "DeepAgro is dedicated to transforming traditional agriculture through cutting-edge AI and Machine Learning technologies. Our goal is to empower farmers with intelligent insights for better crop selection, optimal fertilizer use, and early disease detection.",
      "team_header": "👨‍💻 Our Development Team",
      "team_desc": "A passionate group of students from IIIT Raichur working together to revolutionize agriculture with technology.",
      "tech_stack_header": "🛠️ Technology Stack",
      "ml_title": "🤖 Machine Learning",
      "ml_text": "• XGBOOSTr<br>• Scikit-learn<br>• CNN <br>• Feature Engineering",
      "web_title": "🌐 Web Framework",
      "web_text": "• Streamlit<br>• Python Backend<br>• Interactive UI/UX<br>• Real-time Processing",
      "data_title": "📊 Data & Visualization",
      "data_text": "• Plotly for charts<br>• PIL for image processing<br>• Custom CSS styling<br>• Responsive Design",
      "features_header": "✨ Key Features",
      "smart_pred_header": "🎯 Smart Predictions",
      "smart_pred_list": "- **Crop Recommendation:** AI-powered crop selection based on soil and climate conditions\n- **Fertilizer Optimization:** Intelligent fertilizer recommendations for maximum yield\n- **Disease Detection:** Computer vision for plant disease identification",
      "ux_header": "🔧 User Experience",
      "ux_list": "- **Interactive Interface:** Easy-to-use sliders and input fields\n- **Real-time Analysis:** Instant predictions and recommendations\n- **Educational Content:** Detailed explanations and farming tips",
      "institution_title": "🏫 Institution",
      "institution_text": "<strong>Indian Institute of Information Technology, Raichur</strong><br>Innovating in agricultural technology and sustainable farming solutions.",
      "acknowledgements_title": "🙏 Acknowledgements",
      "acknowledgements_text": "Special thanks to our faculty advisors Dr. Priodyuti Pradhan and the IIIT Raichur community for their support and guidance in developing this agricultural AI solution.",
      "footer_title": "🌱 **DeepAgro**",
      "footer_slogan": "Empowering Agriculture with AI & ML",
      "footer_credit": "❤️ Built by Team DeepAgro | IIIT Raichur | 2025"
    }
  },
  "hi": {
        "page_title": "दीपएग्रो - स्मार्ट कृषि",
        "sidebar_title": "🌾 नेविगेशन",
        "nav_home": "🏠 होम",
        "nav_crop": "🌾 फसल भविष्यवाणी",
        "nav_fertilizer": "🧪 उर्वरक अनुशंसा",
        "nav_disease": "🔬 रोग का पता लगाना",
        "nav_chat": "🤖 दीपएग्रो एआई सहायक",
        "nav_about": "👥 हमारे बारे में",
        "home": {
            "header_logo": "🌱 दीपएग्रो",
            "header_tagline": "एआई और एमएल के साथ स्मार्ट कृषि समाधान",
            "welcome_header": "🌟 कृषि के भविष्य में आपका स्वागत है!",
            "welcome_text": "दीपएग्रो खेती के तरीकों में क्रांति लाने के लिए अत्याधुनिक **मशीन लर्निंग** और **आर्टिफिशियल इंटेलिजेंस** का लाभ उठाता है। हमारा मंच इसके लिए बुद्धिमान अंतर्दृष्टि प्रदान करता है:",
            "card_crop_title": "🌾 स्मार्ट फसल अनुशंसा",
            "card_crop_desc": "उन्नत एमएल एल्गोरिदम का उपयोग करके मिट्टी की स्थिति, जलवायु और पोषक तत्वों के आधार पर वैयक्तिकृत फसल सुझाव प्राप्त करें।",
            "card_fert_title": "🧪 उर्वरक अनुकूलन",
            "card_fert_desc": "पर्यावरणीय प्रभाव को कम करते हुए उपज को अधिकतम करने के लिए सटीक उर्वरक अनुशंसाएं प्राप्त करें।",
            "card_disease_title": "🔬 एआई-संचालित रोग का पता लगाना",
            "card_disease_desc": "अत्याधुनिक सीएनएन डीप लर्निंग मॉडल का उपयोग करके तत्काल रोग पहचान के लिए पत्ती की छवियां अपलोड करें।",
            "metrics_header": "🚀 मुख्य विशेषताएं",
            "metric_crops": "फसल के प्रकार",
            "metric_fertilizers": "उर्वरक के प्रकार",
            "metric_accuracy": "सटीकता",
            "metric_power": "संचालित",
            "why_choose_title": "🌟 दीपएग्रो क्यों चुनें?",
            "why_choose_desc": "हमारी अत्याधुनिक एआई तकनीक के साथ खेती के भविष्य का अनुभव करें जो अधिकतम उपज और स्थिरता के लिए पारंपरिक कृषि को स्मार्ट, डेटा-संचालित निर्णयों में बदल देती है।",
            "benefit_precision_title": "सटीक कृषि",
            "benefit_precision_desc": "इष्टतम फसल चयन और संसाधन प्रबंधन के लिए सटीक सटीकता के साथ डेटा-संचालित निर्णय लें।",
            "benefit_sustain_title": "टिकाऊ खेती",
            "benefit_sustain_desc": "बुद्धिमान अनुशंसाओं के माध्यम से उत्पादकता को अधिकतम करते हुए कचरे और पर्यावरणीय प्रभाव को कम करें।",
            "benefit_realtime_title": "वास्तविक समय विश्लेषण",
            "benefit_realtime_desc": "उन्नत मशीन लर्निंग एल्गोरिदम और कंप्यूटर विज़न द्वारा संचालित त्वरित अंतर्दृष्टि और भविष्यवाणियां प्राप्त करें।"
        },
        "crop_prediction": {
            "main_title": "🌾 इंटेलिजेंट फसल अनुशंसा प्रणाली",
            "subtitle": "अपनी मिट्टी और पर्यावरणीय परिस्थितियों के आधार पर एआई-संचालित फसल सुझाव प्राप्त करें।",
            "expander_header": "ℹ️ फसल भविष्यवाणी मापदंडों को समझना",
            "expander_info_text": "हमारा एआई मॉडल आपकी भूमि के लिए सर्वोत्तम फसलों की सिफारिश करने के लिए कई कारकों का विश्लेषण करता है। प्रत्येक पैरामीटर फसल की उपयुक्तता निर्धारित करने में एक महत्वपूर्ण भूमिका निभाता है:",
            "how_it_works": "📊 **यह कैसे काम करता है:** हमारा मशीन लर्निंग एल्गोरिदम आपके इनपुट डेटा को संसाधित करता है और व्यक्तिगत अनुशंसाएं प्रदान करने के लिए इसे हजारों सफल फसल संयोजनों के साथ तुलना करता है।",
            "env_factors_header": "🌡️ पर्यावरणीय कारक",
            "temp_label": "🌡️ तापमान (°C)",
            "temp_info": "<strong>तापमान प्रभाव:</strong> डिग्री सेल्सियस में परिवेश का तापमान। विभिन्न फसलें विभिन्न तापमान श्रेणियों में बढ़ती हैं - उष्णकटिबंधीय फसलें 25-35°C पसंद करती हैं जबकि शीतोष्ण फसलें 15-25°C पसंद करती हैं।",
            "hum_label": "💧 आर्द्रता (%)",
            "hum_info": "<strong>आर्द्रता प्रभाव:</strong> हवा में सापेक्ष आर्द्रता प्रतिशत। उच्च आर्द्रता (>70%) चावल जैसी फसलों के लिए उपयुक्त है, जबकि कम आर्द्रता (<50%) गेहूं और जौ जैसी फसलों के लिए बेहतर है।",
            "rain_label": "🌧️ वर्षा (मिमी)",
            "rain_info": "<strong>वर्षा प्रभाव:</strong> मिलीमीटर में औसत वर्षा की मात्रा। चावल को 150-300 मिमी की आवश्यकता होती है, गेहूं को 30-100 मिमी की, जबकि सूखा-प्रतिरोधी फसलें <50 मिमी के साथ जीवित रह सकती हैं।",
            "ph_label": "⚗️ मिट्टी का पीएच स्तर",
            "ph_info": "<strong>पीएच प्रभाव:</strong> मिट्टी का पीएच मान अम्लता/क्षारीयता को मापता है। अधिकांश फसलें 6.0-7.5 (थोड़ा अम्लीय से तटस्थ) पसंद करती हैं। अम्लीय मिट्टी (<6) ब्लूबेरी के लिए उपयुक्त है, जबकि क्षारीय मिट्टी (>7.5) शतावरी के लिए उपयुक्त है।",
            "nutrients_header": "🧪 मिट्टी के पोषक तत्व (एनपीके मान)",
            "n_label": "🔵 नायट्रोजन (N) सामग्री",
            "n_info": "<strong>नायट्रोजन (N) की भूमिका:</strong> पत्ती के विकास और क्लोरोफिल उत्पादन के लिए आवश्यक। पत्तेदार सब्जियों को उच्च N (80-120) की आवश्यकता होती है, जबकि जड़ वाली सब्जियों को मध्यम N (40-80) की आवश्यकता होती है।",
            "p_label": "🟡 फॉस्फरस (P) सामग्री",
            "p_info": "<strong>फॉस्फरस (P) की भूमिका:</strong> जड़ विकास और फूल आने के लिए महत्वपूर्ण। फल फसलों को उच्च P (60-100) की आवश्यकता होती है, जबकि घासों को कम P (20-40) की आवश्यकता होती है।",
            "k_label": "🔴 पोटेशियम (K) सामग्री",
            "k_info": "<strong>पोटेशियम (K) की भूमिका:</strong> रोग प्रतिरोध और जल विनियमन के लिए महत्वपूर्ण। जड़ वाली सब्जियों और फलों को उच्च K (80-150) की आवश्यकता होती है, जबकि अनाजों को मध्यम K (40-80) की आवश्यकता होती है।",
            "summary_header": "📊 वर्तमान इनपुट सारांश",
            "summary_temp": "🌡️ **तापमान:**",
            "summary_hum": "💧 **आर्द्रता:**",
            "summary_rain": "🌧️ **वर्षा:**",
            "summary_ph": "⚗️ **पीएच स्तर:**",
            "summary_n": "🔵 **नायट्रोजन (N):**",
            "summary_p": "🟡 **फॉस्फरस (P):**",
            "summary_k": "🔴 **पोटेशियम (K):**",
            "reference_header": "📋 आदर्श सीमा संदर्भ",
            "ref_text": "<strong>इष्टतम बढ़ती हुई स्थितियां:</strong><br>• **तापमान:** 20-30°C (अधिकांश फसलें)<br>• **आर्द्रता:** 40-70% (इष्टतम सीमा)<br>• **वर्षा:** 50-200mm (फसल के अनुसार भिन्न होता है)<br>• **पीएच:** 6.0-7.5 (तटस्थ से थोड़ा अम्लीय)<br>• **एनपीके:** स्वस्थ विकास के लिए संतुलित अनुपात",
            "warning_temp": "🌡️ तापमान सामान्य वृद्धि सीमा (5-45°C) से बाहर है",
            "warning_hum": "💧 आर्द्रता का स्तर अधिकांश फसलों के लिए चुनौतीपूर्ण हो सकता है",
            "warning_ph": "⚗️ पीएच स्तर काफी चरम है और फसल के विकल्पों को सीमित कर सकता है",
            "warning_n": "🔵 बहुत उच्च नाइट्रोजन स्तर अत्यधिक वनस्पति विकास का कारण बन सकता है",
            "warning_p": "🟡 उच्च फॉस्फरस स्तर अन्य पोषक तत्वों के अवशोषण में हस्तक्षेप कर सकता है",
            "warning_k": "🔴 बहुत उच्च पोटेशियम स्तर मिट्टी की संरचना को प्रभावित कर सकता है",
            "warnings_header": "⚠️ इनपुट चेतावनियाँ:",
            "validation_header": "✅ सत्यापन स्थिति",
            "validation_text": "सभी इनपुट मान स्वीकार्य सीमाओं के भीतर हैं! आपकी परिस्थितियाँ फसल की खेती के लिए बहुत अच्छी हैं।",
            "predict_button": "🔮 सर्वोत्तम फसल का अनुमान लगाएं",
            "loading_1": "मिट्टी की स्थितियों का विश्लेषण...",
            "loading_2": "पर्यावरणीय डेटा को संसाधित करना...",
            "loading_3": "फसल डेटाबेस के साथ मिलान...",
            "loading_4": "सिफारिशों को अंतिम रूप देना...",
            "result_header": "🎯 अनुशंसित फसल:",
            "result_confidence": "📊 आत्मविश्वास स्कोर:",
            "result_quality": "🌟 मैच गुणवत्ता:",
            "quality_excellent": "उत्कृष्ट",
            "quality_good": "अच्छा",
            "quality_fair": "ठीक",
            "top_3_header": "📈 शीर्ष 3 फसल अनुशंसाएँ",
            "crop_season": "मौसम",
            "crop_water": "पानी की आवश्यकता",
            "crop_match": "मिलान",
            "crop_suitability": "उपयुक्तता",
            "personalized_tips_header": "💡 वैयक्तिकृत खेती युक्तियाँ",
            "tips_climate_header": "🌡️ जलवायु विचार",
            "tips_temp_high": "<strong>🌡️ उच्च तापमान चेतावनी:</strong> गर्मी प्रतिरोधी किस्मों, छाया जाल, और लगातार सिंचाई के समय पर विचार करें। जल दक्षता के लिए ड्रिप सिंचाई स्थापित करें।",
            "tips_temp_low": "<strong>❄️ ठंडा तापमान:</strong> ठंडी-मौसम की फसलों के लिए आदर्श। पंक्ति कवर और ग्रीनहाउस खेती जैसे पाला संरक्षण उपायों पर विचार करें।",
            "tips_temp_ok": "<strong>🌡️ इष्टतम तापमान:</strong> अधिकांश फसल किस्मों के लिए सही स्थितियाँ। लगातार पानी देना बनाए रखें और कीटों की निगरानी करें।",
            "tips_hum_high": "<strong>💧 उच्च आर्द्रता चेतावनी:</strong> फंगल रोगों को रोकने के लिए उचित पौधे की दूरी और वेंटिलेशन सुनिश्चित करें। फंगीसाइड उपचारों पर विचार करें।",
            "tips_hum_low": "<strong>🏜️ कम आर्द्रता चेतावनी:</strong> मिट्टी की नमी बनाए रखने के लिए मल्चिंग और लगातार हल्के पानी देने पर विचार करें। आर्द्रता बनाए रखने की तकनीकों का उपयोग करें।",
            "tips_hum_ok": "<strong>💧 अच्छी आर्द्रता का स्तर:</strong> स्वस्थ पौधे के विकास के लिए अनुकूल स्थितियाँ। इष्टतम पौधे के विकास के लिए निगरानी करें।",
            "tips_soil_header": "🧪 मृदा प्रबंधन",
            "tips_ph_acidic": "<strong>⚗️ अम्लीय मिट्टी:</strong> पीएच बढ़ाने के लिए चूना जोड़ने पर विचार करें। एल्यूमीनियम विषाक्तता के लिए परीक्षण करें और मिट्टी की संरचना में सुधार के लिए जैविक पदार्थ जोड़ें।",
            "tips_ph_alkaline": "<strong>⚗️ क्षारीय मिट्टी:</strong> पीएच कम करने के लिए सल्फर या जैविक पदार्थ जोड़ने पर विचार करें। सूक्ष्म पोषक तत्वों की कमी के लिए निगरानी करें।",
            "tips_ph_ok": "<strong>⚗️ इष्टतम पीएच सीमा:</strong> पोषक तत्वों की उपलब्धता के लिए सही स्थितियाँ। नियमित जैविक संशोधनों के साथ मिट्टी के स्वास्थ्य को बनाए रखें।",
            "tips_n_low": "<strong>🔵 कम नायट्रोजन:</strong> यूरिया या जैविक खाद जैसे नायट्रोजन-समृद्ध उर्वरकों पर विचार करें। बेहतर अवशोषण के लिए विभाजित खुराक में लगाएं।",
            "tips_n_high": "<strong>🔵 उच्च नायट्रोजन:</strong> अत्यधिक वनस्पति विकास का कारण बन सकता है। सावधानीपूर्वक निगरानी करें और यदि आवश्यक हो तो नायट्रोजन इनपुट कम करें।",
            "tips_p_low": "<strong>🟡 कम फॉस्फरस:</strong> डीएपी या रॉक फॉस्फेट लगाने पर विचार करें। जड़ के विकास और फूल आने के लिए आवश्यक।",
            "tips_k_low": "<strong>🔴 कम पोटेशियम:</strong> एमओपी (पोटाश का म्यूरेट) लगाने पर विचार करें। रोग प्रतिरोध और जल विनियमन के लिए महत्वपूर्ण।",
            "summary_box_header": "🌟 आपकी वैयक्तिकृत फसल अनुशंसा का सारांश",
            "summary_box_text": "आपकी मिट्टी और पर्यावरणीय स्थितियों के हमारे एआई विश्लेषण के आधार पर, **{}** आपकी भूमि के लिए **{:.1f}% आत्मविश्वास स्कोर** के साथ सबसे उपयुक्त फसल है।",
            "summary_match_quality": "🎯 मैच गुणवत्ता",
            "summary_growth_potential": "🌱 विकास क्षमता",
            "summary_econ_viability": "💰 आर्थिक व्यवहार्यता",
            "growth_high": "उच्च",
            "growth_medium": "मध्यम",
            "growth_moderate": "मध्यम",
            "econ_prof": "लाभदायक",
            "econ_good": "अच्छा"
        },
        "fertilizer_recommendation": {
            "main_title": "🧪 उर्वरक अनुशंसा प्रणाली",
            "subtitle": "अपनी फसल और मिट्टी की स्थिति के आधार पर इष्टतम उर्वरक सुझाव प्राप्त करें।",
            "section_info": "🌱 फसल और मिट्टी की जानकारी",
            "section_env": "🌡️ पर्यावरणीय स्थितियाँ",
            "section_nutrients": "🧪 वर्तमान मिट्टी के पोषक तत्व",
            "crop_type_label": "फसल का प्रकार",
            "soil_type_label": "मिट्टी का प्रकार",
            "temp_label": "तापमान (°C)",
            "hum_label": "आर्द्रता (%)",
            "moisture_label": "मिट्टी की नमी (%)",
            "nitrogen_label": "नायट्रोजन सामग्री",
            "phosphorus_label": "फॉस्फरस सामग्री",
            "potassium_label": "पोटेशियम सामग्री",
            "nutrient_status_header": "📊 पोषक तत्व स्थिति",
            "low": "🔴 निम्न",
            "medium": "🟡 मध्यम",
            "high": "🟢 उच्च",
            "predict_button": "💡 उर्वरक अनुशंसा प्राप्त करें",
            "result_header": "🎯 अनुशंसित उर्वरक:",
            "result_confidence": "📊 आत्मविश्वास:",
            "result_info_pre": "के लिए ",
            "result_info_in": "मिट्टी में:",
            "result_info_apply": "- **{}** उर्वरक लगाएं",
            "result_info_tips": "- मात्रा निर्धारित करते समय वर्तमान पोषक तत्व स्तरों पर विचार करें\n- उचित वृद्धि चरण के दौरान लगाएं\n- मिट्टी की नमी और मौसम की स्थिति की निगरानी करें",
            "error_message": "भविष्यवाणी में त्रुटि। कृपया अपने इनपुट की जाँच करें।"
        },
        "disease_detection": {
            "main_title": "🔬 रोग का पता लगाना",
            "subtitle": "डीप लर्निंग सीएनएन मॉडल का उपयोग करके पत्ती की छवि से तत्काल रोग की पहचान करें।",
            "upload_header": "📷 पौधे की पत्ती की छवि अपलोड करें",
            "upload_guidelines_title": "📸 छवि अपलोड दिशानिर्देश:",
            "upload_guidelines_text": "✓ स्पष्ट, अच्छी तरह से प्रकाशित पत्ती की छवियां<br>✓ प्रभावित क्षेत्रों या लक्षणों पर ध्यान केंद्रित करें<br>✓ समर्थित प्रारूप: JPG, PNG, JPEG<br>✓ अधिकतम आकार: 10MB",
            "file_uploader_label": "एक पत्ती की छवि चुनें...",
            "file_uploader_help": "पौधे की पत्ती की एक स्पष्ट छवि अपलोड करें",
            "uploaded_image_caption": "📷 अपलोड की गई पत्ती की छवि",
            "analyze_button": "🔍 रोगों के लिए विश्लेषण करें",
            "loading_message": "🧠 एआई छवि का विश्लेषण कर रहा है...",
            "analysis_complete": "विश्लेषण पूरा हुआ!",
            "result_header": "🎯 अनुमानित रोग:",
            "result_confidence": "📊 आत्मविश्वास:",
            "disease_warning": "❗ आपका पौधा रोगग्रस्त हो सकता है। पुष्टि के लिए कृपया किसी पेशेवर से सलाह लें।",
            "healthy_message": "✅ पौधा स्वस्थ प्रतीत होता है!",
            "disease_names": {
                "Healthy": "स्वस्थ",
                "Apple Scab": "एप्पल स्कैब",
                "Black Rot": "ब्लैक रॉट",
                "Cedar Apple Rust": "सेडर एप्पल रस्ट",
                "Bacterial Blight": "बैक्टीरियल ब्लाइट",
                "Early Blight": "अर्ली ब्लाइट",
                "Late Blight": "लेट ब्लाइट",
                "Leaf Mold": "लीफ मोल्ड",
                "Septoria Leaf Spot": "सेप्टोरिया लीफ स्पॉट",
                "Target Spot": "टारगेट स्पॉट",
                "Mosaic Virus": "मोज़ेक वायरस"
            }
        },
        "about_page": {
            "main_title": "👥 हमारे बारे में",
            "subtitle": "स्मार्ट कृषि क्रांति के पीछे अभिनव टीम से मिलें!",
            "mission_title": "🌟 हमारा मिशन",
            "mission_text": "दीपएग्रो अत्याधुनिक एआई और मशीन लर्निंग प्रौद्योगिकियों के माध्यम से पारंपरिक कृषि को बदलने के लिए समर्पित है। हमारा लक्ष्य बेहतर फसल चयन, इष्टतम उर्वरक उपयोग, और प्रारंभिक रोग का पता लगाने के लिए बुद्धिमान अंतर्दृष्टि के साथ किसानों को सशक्त बनाना है।",
            "team_header": "👨‍💻 हमारी विकास टीम",
            "team_desc": "आईआईआईटी रायचूर के छात्रों का एक भावुक समूह जो प्रौद्योगिकी के साथ कृषि में क्रांति लाने के लिए मिलकर काम कर रहा है।",
            "tech_stack_header": "🛠️ प्रौद्योगिकी स्टैक",
            "ml_title": "🤖 मशीन लर्निंग",
            "ml_text": "• रैंडम फ़ॉरेस्ट क्लासिफायर<br>• सिकाईट-लर्न<br>• नम्पाई और पांडास<br>• फ़ीचर इंजीनियरिंग",
            "web_title": "🌐 वेब फ्रेमवर्क",
            "web_text": "• स्ट्रीमलिट<br>• पायथन बैकएंड<br>• इंटरैक्टिव यूआई/यूएक्स<br>• वास्तविक समय प्रसंस्करण",
            "data_title": "📊 डेटा और विज़ुअलाइज़ेशन",
            "data_text": "• चार्ट के लिए प्लॉटली<br>• इमेज प्रोसेसिंग के लिए पीआईएल<br>• कस्टम सीएसएस स्टाइलिंग<br>• रिस्पॉन्सिव डिज़ाइन",
            "features_header": "✨ मुख्य विशेषताएं",
            "smart_pred_header": "🎯 स्मार्ट भविष्यवाणियाँ",
            "smart_pred_list": "- **फसल अनुशंसा:** मिट्टी और जलवायु परिस्थितियों के आधार पर एआई-संचालित फसल चयन\n- **उर्वरक अनुकूलन:** अधिकतम उपज के लिए बुद्धिमान उर्वरक अनुशंसाएं\n- **रोग का पता लगाना:** पौधे के रोग की पहचान के लिए कंप्यूटर विजन",
            "ux_header": "🔧 उपयोगकर्ता अनुभव",
            "ux_list": "- **इंटरैक्टिव इंटरफ़ेस:** उपयोग में आसान स्लाइडर और इनपुट फ़ील्ड\n- **वास्तविक समय विश्लेषण:** त्वरित भविष्यवाणियां और अनुशंसाएं\n- **शैक्षणिक सामग्री:** विस्तृत स्पष्टीकरण और खेती युक्तियाँ",
            "institution_title": "🏫 संस्थान",
            "institution_text": "<strong>भारतीय सूचना प्रौद्योगिकी संस्थान, रायचूर</strong><br>कृषि प्रौद्योगिकी और टिकाऊ खेती के समाधानों में नवाचार।",
            "acknowledgements_title": "🙏 आभार",
            "acknowledgements_text": "इस कृषि एआई समाधान को विकसित करने में उनके समर्थन और मार्गदर्शन के लिए हमारे संकाय सलाहकारों डॉ.प्रियोद्युति प्रधान और आईआईआईटी रायचूर समुदाय को विशेष धन्यवाद।",
            "footer_title": "🌱 **दीपएग्रो**",
            "footer_slogan": "एआई और एमएल के साथ कृषि को सशक्त बनाना",
            "footer_credit": "❤️ टीम दीपएग्रो द्वारा निर्मित | आईआईआईटी रायचूर | 2025"
        }
    },
    "ta": {
      "page_title": "டீப்அக்ரோ - ஸ்மார்ட் விவசாயம்",
      "sidebar_title": "🌾 வழிசெலுத்தல்",
      "nav_home": "🏠 முகப்பு",
      "nav_crop": "🌾 பயிர் கணிப்பு",
      "nav_fertilizer": "🧪 உரப் பரிந்துரை",
      "nav_disease": "🔬 நோய் கண்டறிதல்",
      "nav_chat": "🤖 டீப்அக்ரோ AI உதவியாளர்",
      "nav_about": "👥 எங்களைப் பற்றி",
      "home": {
        "header_logo": "🌱 டீப்அக்ரோ",
        "header_tagline": "AI மற்றும் ML உடன் ஸ்மார்ட் விவசாய தீர்வுகள்",
        "welcome_header": "🌟 விவசாயத்தின் எதிர்காலத்திற்கு வரவேற்கிறோம்!",
        "welcome_text": "டீப்அக்ரோ விவசாய முறைகளில் புரட்சியை ஏற்படுத்த அதிநவீன **இயந்திர கற்றல்** மற்றும் **செயற்கை நுண்ணறிவு** ஆகியவற்றைப் பயன்படுத்துகிறது. எங்கள் தளம் இதற்கான புத்திசாலித்தனமான நுண்ணறிவுகளை வழங்குகிறது:",
        "card_crop_title": "🌾 ஸ்மார்ட் பயிர் பரிந்துரை",
        "card_crop_desc": "மேம்பட்ட ML அல்காரிதம்களைப் பயன்படுத்தி மண் நிலைமைகள், காலநிலை மற்றும் ஊட்டச்சத்துக்கள் ஆகியவற்றின் அடிப்படையில் தனிப்பயனாக்கப்பட்ட பயிர் பரிந்துரைகளைப் பெறுங்கள்.",
        "card_fert_title": "🧪 உர உகப்பாக்கம்",
        "card_fert_desc": "சுற்றுச்சூழல் பாதிப்பைக் குறைக்கும் அதே வேளையில் விளைச்சலை அதிகப்படுத்த துல்லியமான உரப் பரிந்துரைகளைப் பெறுங்கள்.",
        "card_disease_title": "🔬 AI-இயக்கப்படும் நோய் கண்டறிதல்",
        "card_disease_desc": "அதிநவீன CNN ஆழமான கற்றல் மாதிரிகளைப் பயன்படுத்தி உடனடி நோய் கண்டறிதலுக்காக இலை படங்களை பதிவேற்றவும்.",
        "metrics_header": "🚀 முக்கிய அம்சங்கள்",
        "metric_crops": "பயிர் வகைகள்",
        "metric_fertilizers": "உர வகைகள்",
        "metric_accuracy": "துல்லியம்",
        "metric_power": "இயங்குபவை",
        "why_choose_title": "🌟 ஏன் டீப்அக்ரோவைத் தேர்ந்தெடுக்க வேண்டும்?",
        "why_choose_desc": "எங்களின் அதிநவீன AI தொழில்நுட்பம், அதிகபட்ச விளைச்சல் மற்றும் நிலைத்தன்மைக்காக பாரம்பரிய விவசாயத்தை ஸ்மார்ட், தரவு சார்ந்த முடிவுகளாக மாற்றும் விவசாயத்தின் எதிர்காலத்தை அனுபவிக்கவும்।",
        "benefit_precision_title": "துல்லியமான விவசாயம்",
        "benefit_precision_desc": "உகந்த பயிர் தேர்வு மற்றும் வள மேலாண்மைக்காக துல்லியமான துல்லியத்துடன் தரவு சார்ந்த முடிவுகளை எடுங்கள்.",
        "benefit_sustain_title": "நிலையான விவசாயம்",
        "benefit_sustain_desc": "புத்திசாலித்தனமான பரிந்துரைகள் மூலம் உற்பத்தித்திறனை அதிகப்படுத்துவதன் மூலம் கழிவுகள் மற்றும் சுற்றுச்சூழல் பாதிப்பைக் குறைக்கவும்.",
        "benefit_realtime_title": "உண்மையான நேர பகுப்பாய்வு",
        "benefit_realtime_desc": "மேம்படுத்தப்பட்ட இயந்திர கற்றல் அல்காரிதம்கள் மற்றும் கணினி பார்வை மூலம் உடனடி நுண்ணறிவுகளையும் கணிப்புகளையும் பெறுங்கள்।"
      },
      "crop_prediction": {
        "main_title": "🌾 அறிவார்ந்த பயிர் பரிந்துரை அமைப்பு",
        "subtitle": "உங்கள் மண் மற்றும் சுற்றுச்சூழல் நிலைமைகளின் அடிப்படையில் AI-இயக்கப்படும் பயிர் பரிந்துரைகளைப் பெறுங்கள்.",
        "expander_header": "ℹ️ பயிர் கணிப்பு அளவுருக்களைப் புரிந்துகொள்ளுதல்",
        "expander_info_text": "எங்கள் AI மாதிரி உங்கள் நிலத்திற்கு சிறந்த பயிர்களைப் பரிந்துரைக்க பல காரணிகளை பகுப்பாய்வு செய்கிறது. ஒவ்வொரு அளவுருவும் பயிரின் பொருத்தத்தை தீர்மானிப்பதில் ஒரு முக்கிய பங்கை வகிக்கிறது:",
        "how_it_works": "📊 **இது எவ்வாறு செயல்படுகிறது:** எங்கள் இயந்திர கற்றல் அல்காரிதம் உங்கள் உள்ளீட்டுத் தரவைச் செயலாக்கி, தனிப்பயனாக்கப்பட்ட பரிந்துரைகளை வழங்க ஆயிரக்கணக்கான வெற்றிகரமான பயிர் சேர்க்கைகளுடன் அதை ஒப்பிடுகிறது।",
        "env_factors_header": "🌡️ சுற்றுச்சூழல் காரணிகள்",
        "temp_label": "🌡️ வெப்பநிலை (°C)",
        "temp_info": "<strong>வெப்பநிலை தாக்கம்:</strong> டிகிரி செல்சியஸில் சுற்றுப்புற வெப்பநிலை. வெவ்வேறு பயிர்கள் வெவ்வேறு வெப்பநிலை வரம்புகளில் வளரும் - வெப்பமண்டல பயிர்கள் 25-35°C விரும்புகின்றன, அதே நேரத்தில் மிதமான பயிர்கள் 15-25°C விரும்புகின்றன.",
        "hum_label": "💧 ஈரப்பதம் (%)",
        "hum_info": "<strong>ஈரப்பதம் தாக்கம்:</strong> காற்றில் உள்ள சார்பு ஈரப்பதம் சதவீதம். அதிக ஈரப்பதம் (>70%) அரிசி போன்ற பயிர்களுக்கு ஏற்றது, அதே நேரத்தில் குறைந்த ஈரப்பதம் (<50%) கோதுமை மற்றும் பார்லி போன்றவற்றுக்கு நல்லது.",
        "rain_label": "🌧️ மழைப்பொழிவு (மிமீ)",
        "rain_info": "<strong>மழைப்பொழிவு தாக்கம்:</strong> மில்லிமீட்டரில் சராசரி மழைப்பொழிவு அளவு. அரிசிக்கு 150-300 மிமீ தேவைப்படுகிறது, கோதுமைக்கு 30-100 மிமீ தேவைப்படுகிறது, அதே நேரத்தில் வறட்சியை எதிர்க்கும் பயிர்கள் <50 மிமீ உடன் வாழ முடியும்.",
        "ph_label": "⚗️ மண் pH அளவு",
        "ph_info": "<strong>pH தாக்கம்:</strong> மண் pH மதிப்பு அமிலத்தன்மை/காரத்தன்மையை அளவிடுகிறது. பெரும்பாலான பயிர்கள் 6.0-7.5 (லேசான அமிலத்தன்மை முதல் நடுநிலை) விரும்புகின்றன. அமில மண் (<6) ப்ளூபெர்ரிகளுக்கு ஏற்றது, அதே நேரத்தில் கார மண் (>7.5) அஸ்பாரகஸுக்கு ஏற்றது.",
        "nutrients_header": "🧪 மண் ஊட்டச்சத்துக்கள் (NPK மதிப்புகள்)",
        "n_label": "🔵 நைட்ரஜன் (N) உள்ளடக்கம்",
        "n_info": "<strong>நைட்ரஜன் (N) பங்கு:</strong> இலை வளர்ச்சி மற்றும் குளோரோபில் உற்பத்திக்கு அவசியம். இலை காய்கறிகளுக்கு அதிக N (80-120) தேவை, அதே நேரத்தில் வேர் காய்கறிகளுக்கு மிதமான N (40-80) தேவை.",
        "p_label": "🟡 பாஸ்பரஸ் (P) உள்ளடக்கம்",
        "p_info": "<strong>பாஸ்பரஸ் (P) பங்கு:</strong> வேர் வளர்ச்சி மற்றும் பூக்கும் முக்கியமானது. பழப் பயிர்களுக்கு அதிக P (60-100) தேவை, அதே நேரத்தில் புற்களுக்கு குறைந்த P (20-40) தேவை.",
        "k_label": "🔴 பொட்டாசியம் (K) உள்ளடக்கம்",
        "k_info": "<strong>பொட்டாசியம் (K) பங்கு:</strong> நோய் எதிர்ப்பு சக்தி மற்றும் நீர் ஒழுங்குமுறைக்கு முக்கியமானது. வேர் காய்கறிகள் மற்றும் பழங்களுக்கு அதிக K (80-150) தேவை, அதே நேரத்தில் தானியங்களுக்கு மிதமான K (40-80) தேவை.",
        "summary_header": "📊 தற்போதைய உள்ளீட்டுச் சுருக்கம்",
        "summary_temp": "🌡️ **வெப்பநிலை:**",
        "summary_hum": "💧 **ஈரப்பதம்:**",
        "summary_rain": "🌧️ **மழைப்பொழிவு:**",
        "summary_ph": "⚗️ **pH அளவு:**",
        "summary_n": "🔵 **நைட்ரஜன் (N):**",
        "summary_p": "🟡 **பாஸ்பரஸ் (P):**",
        "summary_k": "🔴 **பொட்டாசியம் (K):**",
        "reference_header": "📋 சிறந்த வரம்பு குறிப்பு",
        "ref_text": "<strong>உகந்த வளரும் நிலைமைகள்:</strong><br>• **வெப்பநிலை:** 20-30°C (பெரும்பாலான பயிர்கள்)<br>• **ஈரப்பதம்:** 40-70% (உகந்த வரம்பு)<br>• **மழைப்பொழிவு:** 50-200mm (பயிருக்கு பயிர் மாறுபடும்)<br>• **pH:** 6.0-7.5 (நடுநிலை முதல் லேசான அமிலத்தன்மை)<br>• **NPK:** ஆரோக்கியமான வளர்ச்சிக்கு சமநிலை விகிதம்",
        "warning_temp": "🌡️ வெப்பநிலை இயல்பான வளர்ச்சி வரம்பிற்கு (5-45°C) வெளியே உள்ளது",
        "warning_hum": "💧 ஈரப்பதம் அளவு பெரும்பாலான பயிர்களுக்கு சவாலானதாக இருக்கலாம்",
        "warning_ph": "⚗️ pH அளவு மிகவும் தீவிரமானது மற்றும் பயிர் விருப்பங்களை கட்டுப்படுத்தலாம்",
        "warning_n": "🔵 மிக அதிக நைட்ரஜன் அளவு அதிகப்படியான தாவர வளர்ச்சியை ஏற்படுத்தும்",
        "warning_p": "🟡 அதிக பாஸ்பரஸ் அளவு மற்ற ஊட்டச்சத்துக்களின் உறிஞ்சுவதில் தலையிடலாம்",
        "warning_k": "🔴 மிக அதிக பொட்டாசியம் அளவு மண் அமைப்பை பாதிக்கலாம்",
        "warnings_header": "⚠️ உள்ளீட்டு எச்சரிக்கைகள்:",
        "validation_header": "✅ சரிபார்ப்பு நிலை",
        "validation_text": "அனைத்து உள்ளீட்டு மதிப்புகளும் ஏற்கத்தக்க வரம்புகளுக்குள் உள்ளன! உங்கள் நிலைமைகள் பயிர் சாகுபடிக்கு மிகவும் நல்லது.",
        "predict_button": "🔮 சிறந்த பயிரைக் கணிக்கவும்",
        "loading_1": "மண் நிலைமைகளை பகுப்பாய்வு செய்கிறது...",
        "loading_2": "சுற்றுச்சூழல் தரவைச் செயலாக்குகிறது...",
        "loading_3": "பயிர் தரவுத்தளத்துடன் பொருந்துகிறது...",
        "loading_4": "பரிந்துரைகளை இறுதி செய்கிறது...",
        "result_header": "🎯 பரிந்துரைக்கப்பட்ட பயிர்:",
        "result_confidence": "📊 நம்பிக்கை மதிப்பெண்:",
        "result_quality": "🌟 பொருத்தத் தரம்:",
        "quality_excellent": "சிறந்தது",
        "quality_good": "நல்லது",
        "quality_fair": "சமநிலை",
        "top_3_header": "📈 சிறந்த 3 பயிர் பரிந்துரைகள்",
        "crop_season": "பருவம்",
        "crop_water": "நீர் தேவை",
        "crop_match": "பொருத்தம்",
        "crop_suitability": "பொருத்தமான தன்மை",
        "personalized_tips_header": "💡 தனிப்பயனாக்கப்பட்ட விவசாய குறிப்புகள்",
        "tips_climate_header": "🌡️ காலநிலை கருத்தாய்வுகள்",
        "tips_temp_high": "<strong>🌡️ அதிக வெப்பநிலை எச்சரிக்கை:</strong> வெப்பத்தை எதிர்க்கும் வகைகள், நிழல் வலைகள் மற்றும் அடிக்கடி நீர்ப்பாசன நேரத்தை கருத்தில் கொள்ளுங்கள். நீர் திறனுக்காக சொட்டு நீர்ப்பாசனம் அமைக்கவும்.",
        "tips_temp_low": "<strong>❄️ குளிர் வெப்பநிலை:</strong> குளிர்ந்த-காலநிலை பயிர்களுக்கு சிறந்தது. வரிசை கவர்கள் மற்றும் கிரீன்ஹவுஸ் விவசாயம் போன்ற உறைபனி பாதுகாப்பு நடவடிக்கைகளை கருத்தில் கொள்ளுங்கள்.",
        "tips_temp_ok": "<strong>🌡️ உகந்த வெப்பநிலை:</strong> பெரும்பாலான பயிர் வகைகளுக்கு சரியான நிலைமைகள். தொடர்ந்து நீர்ப்பாசனத்தை பராமரிக்கவும் மற்றும் பூச்சிகளை கண்காணிக்கவும்.",
        "tips_hum_high": "<strong>💧 அதிக ஈரப்பதம் எச்சரிக்கை:</strong> பூஞ்சை நோய்களைத் தடுக்க சரியான தாவர இடைவெளி மற்றும் காற்றோட்டத்தை உறுதி செய்யுங்கள். பூஞ்சைக்கொல்லி சிகிச்சைகளை கருத்தில் கொள்ளுங்கள்.",
        "tips_hum_low": "<strong>🏜️ குறைந்த ஈரப்பதம் எச்சரிக்கை:</strong> மண் ஈரப்பதத்தை பராமரிக்க மல்ச்சிங் மற்றும் அடிக்கடி லேசான நீர்ப்பாசனத்தை கருத்தில் கொள்ளுங்கள். ஈரப்பதத்தை பராமரிக்கும் நுட்பங்களைப் பயன்படுத்தவும்.",
        "tips_hum_ok": "<strong>💧 நல்ல ஈரப்பதம் அளவு:</strong> ஆரோக்கியமான தாவர வளர்ச்சிக்கு ஏற்ற நிலைமைகள். உகந்த தாவர வளர்ச்சிக்கு கண்காணிக்கவும்.",
        "tips_soil_header": "🧪 மண் மேலாண்மை",
        "tips_ph_acidic": "<strong>⚗️ அமில மண்:</strong> pH ஐ அதிகரிக்க சுண்ணாம்பு சேர்ப்பதைக் கருத்தில் கொள்ளுங்கள். அலுமினிய நச்சுத்தன்மைக்கு சோதித்து, மண் அமைப்பை மேம்படுத்த கரிமப் பொருட்களைச் சேர்க்கவும்.",
        "tips_ph_alkaline": "<strong>⚗️ கார மண்:</strong> pH ஐ குறைக்க சல்பர் அல்லது கரிமப் பொருட்களைச் சேர்ப்பதைக் கருத்தில் கொள்ளுங்கள். நுண் ஊட்டச்சத்து குறைபாடுகளை கண்காணிக்கவும்.",
        "tips_ph_ok": "<strong>⚗️ உகந்த pH வரம்பு:</strong> ஊட்டச்சத்துக்களின் கிடைக்கும் தன்மைக்கு சரியான நிலைமைகள். வழக்கமான கரிம மாற்றங்களுடன் மண் ஆரோக்கியத்தை பராமரிக்கவும்.",
        "tips_n_low": "<strong>🔵 குறைந்த நைட்ரஜன்:</strong> யூரியா அல்லது கரிம உரம் போன்ற நைட்ரஜன் நிறைந்த உரங்களைக் கருத்தில் கொள்ளுங்கள். சிறந்த உறிஞ்சுவதற்கு பிரிந்த அளவுகளில் பயன்படுத்தவும்.",
        "tips_n_high": "<strong>🔵 அதிக நைட்ரஜன்:</strong> அதிகப்படியான தாவர வளர்ச்சியை ஏற்படுத்தலாம். கவனமாக கண்காணிக்கவும் மற்றும் தேவைப்பட்டால் நைட்ரஜன் உள்ளீட்டைக் குறைக்கவும்.",
        "tips_p_low": "<strong>🟡 குறைந்த பாஸ்பரஸ்:</strong> டிஏபி அல்லது ராக் பாஸ்பேட் பயன்படுத்துவதைக் கருத்தில் கொள்ளுங்கள். வேர் வளர்ச்சி மற்றும் பூக்களுக்கு அவசியம்.",
        "tips_k_low": "<strong>🔴 குறைந்த பொட்டாசியம்:</strong> எம்ஓபி (பொட்டாஷின் மியூரேட்) பயன்படுத்துவதைக் கருத்தில் கொள்ளுங்கள். நோய் எதிர்ப்பு சக்தி மற்றும் நீர் ஒழுங்குமுறைக்கு முக்கியமானது।",
        "summary_box_header": "🌟 உங்கள் தனிப்பயனாக்கப்பட்ட பயிர் பரிந்துரையின் சுருக்கம்",
        "summary_box_text": "உங்கள் மண் மற்றும் சுற்றுச்சூழல் நிலைமைகளின் எங்கள் AI பகுப்பாய்வின் அடிப்படையில், **{}** என்பது உங்கள் நிலத்திற்கு **{:.1f}% நம்பிக்கை மதிப்பெண்** உடன் மிகவும் பொருத்தமான பயிர் ஆகும்.",
        "summary_match_quality": "🎯 பொருத்தத் தரம்",
        "summary_growth_potential": "🌱 வளர்ச்சி திறன்",
        "summary_econ_viability": "💰 பொருளாதார சாத்தியக்கூறு",
        "growth_high": "அதிகம்",
        "growth_medium": "நடுத்தரம்",
        "growth_moderate": "நடுத்தரம்",
        "econ_prof": "லாபகரமானது",
        "econ_good": "நல்லது"
      },
      "fertilizer_recommendation": {
        "main_title": "🧪 உரப் பரிந்துரை அமைப்பு",
        "subtitle": "உங்கள் பயிர் மற்றும் மண் நிலைமைகளின் அடிப்படையில் உகந்த உர பரிந்துரைகளை பெறுங்கள்.",
        "section_info": "🌱 பயிர் மற்றும் மண் தகவல்",
        "section_env": "🌡️ சுற்றுச்சூழல் நிலைமைகள்",
        "section_nutrients": "🧪 தற்போதைய மண் ஊட்டச்சத்துக்கள்",
        "crop_type_label": "பயிர் வகை",
        "soil_type_label": "மண் வகை",
        "temp_label": "வெப்பநிலை (°C)",
        "hum_label": "ஈரப்பதம் (%)",
        "moisture_label": "மண் ஈரப்பதம் (%)",
        "nitrogen_label": "நைட்ரஜன் உள்ளடக்கம்",
        "phosphorus_label": "பாஸ்பரஸ் உள்ளடக்கம்",
        "potassium_label": "பொட்டாசியம் உள்ளடக்கம்",
        "nutrient_status_header": "📊 ஊட்டச்சத்து நிலை",
        "low": "🔴 குறைவு",
        "medium": "🟡 நடுத்தரம்",
        "high": "🟢 அதிகம்",
        "predict_button": "💡 உரப் பரிந்துரையைப் பெறுங்கள்",
        "result_header": "🎯 பரிந்துரைக்கப்பட்ட உரம்:",
        "result_confidence": "📊 நம்பிக்கை:",
        "result_info_pre": "வகை ",
        "result_info_in": " மண்ணில்:",
        "result_info_apply": "- **{}** உரத்தைப் பயன்படுத்துங்கள்",
        "result_info_tips": "- அளவை தீர்மானிக்கும்போது தற்போதைய ஊட்டச்சத்து நிலைகளை கருத்தில் கொள்ளுங்கள்\n- பொருத்தமான வளர்ச்சி கட்டத்தில் பயன்படுத்தவும்\n- மண் ஈரப்பதம் மற்றும் வானிலை நிலைமைகளை கண்காணிக்கவும்",
        "error_message": "கணிப்பில் பிழை. உங்கள் உள்ளீடுகளை சரிபார்க்கவும்।"
      },
      "disease_detection": {
        "main_title": "🔬 நோய் கண்டறிதல்",
        "subtitle": "ஆழமான கற்றல் CNN மாதிரிகளைப் பயன்படுத்தி உடனடி நோய் கண்டறிதலுக்காக ஒரு இலை படத்தை பதிவேற்றவும்.",
        "upload_header": "📷 தாவர இலை படத்தை பதிவேற்றவும்",
        "upload_guidelines_title": "📸 பட பதிவேற்ற வழிகாட்டுதல்கள்:",
        "upload_guidelines_text": "✓ தெளிவான, நன்கு ஒளியூட்டப்பட்ட இலை படங்கள்<br>✓ பாதிக்கப்பட்ட பகுதிகள் அல்லது அறிகுறிகளில் கவனம் செலுத்துங்கள்<br>✓ ஆதரிக்கப்படும் வடிவங்கள்: JPG, PNG, JPEG<br>✓ அதிகபட்ச அளவு: 10MB",
        "file_uploader_label": "ஒரு இலை படத்தை தேர்வு செய்யவும்...",
        "file_uploader_help": "தாவர இலையின் தெளிவான படத்தை பதிவேற்றவும்",
        "uploaded_image_caption": "📷 பதிவேற்றப்பட்ட இலை படம்",
        "analyze_button": "🔍 நோய்களுக்காக பகுப்பாய்வு செய்யவும்",
        "loading_message": "🧠 AI படத்தை பகுப்பாய்வு செய்கிறது...",
        "analysis_complete": "பகுப்பாய்வு முடிந்தது!",
        "result_header": "🎯 கணிக்கப்பட்ட நோய்:",
        "result_confidence": "📊 நம்பிக்கை:",
        "disease_warning": "❗ உங்கள் தாவரம் நோய்வாய்ப்பட்டிருக்கலாம். உறுதிப்படுத்த ஒரு நிபுணரை அணுகவும்.",
        "healthy_message": "✅ தாவரம் ஆரோக்கியமாகத் தெரிகிறது!"
      },
      "about_page": {
        "main_title": "👥 எங்களைப் பற்றி",
        "subtitle": "ஸ்மார்ட் விவசாய புரட்சிக்கு பின்னால் உள்ள புதுமையான குழுவை சந்திக்கவும்!",
        "mission_title": "🌟 எங்கள் நோக்கம்",
        "mission_text": "டீப்அக்ரோ அதிநவீன AI மற்றும் இயந்திர கற்றல் தொழில்நுட்பங்கள் மூலம் பாரம்பரிய விவசாயத்தை மாற்றுவதற்கு அர்ப்பணிக்கப்பட்டுள்ளது. எங்கள் நோக்கம் சிறந்த பயிர் தேர்வு, உகந்த உரப் பயன்பாடு மற்றும் ஆரம்ப நோய் கண்டறிதலுக்கான புத்திசாலித்தனமான நுண்ணறிவுகளுடன் விவசாயிகளை மேம்படுத்துவதாகும்.",
        "team_header": "👨‍💻 எங்கள் வளர்ச்சி குழு",
        "team_desc": "ஐஐஐடி ராய்ச்சூரில் உள்ள ஒரு ஆர்வமுள்ள மாணவர் குழு தொழில்நுட்பத்துடன் விவசாயத்தில் புரட்சியை ஏற்படுத்த இணைந்து செயல்படுகிறது.",
        "tech_stack_header": "🛠️ தொழில்நுட்ப அடுக்கு",
        "ml_title": "🤖 இயந்திர கற்றல்",
        "ml_text": "• ரैंडம் ஃபாரஸ்ட் கிளாசிஃபையர்<br>• சிகாட்-லெர்ன்<br>• நும்பை மற்றும் பாண்டாஸ்<br>• ஃபீச்சர் இன்ஜினியரிங்",
        "web_title": "🌐 வலை கட்டமைப்பு",
        "web_text": "• ஸ்ட்ரீம்லிட்<br>• பைதான் பேக்கெண்ட்<br>• ஊடாடும் UI/UX<br>• உண்மையான நேர செயலாக்கம்",
        "data_title": "📊 தரவு மற்றும் காட்சிப்படுத்தல்",
        "data_text": "• விளக்கப்படங்களுக்கு ப்ளாட்லி<br>• பட செயலாக்கத்திற்கு PIL<br>• தனிப்பயன் CSS ஸ்டைலிங்<br>• ரெஸ்பான்சிவ் டிசைன்",
        "features_header": "✨ முக்கிய அம்சங்கள்",
        "smart_pred_header": "🎯 ஸ்மார்ட் கணிப்புகள்",
        "smart_pred_list": "- **பயிர் பரிந்துரை:** மண் மற்றும் காலநிலை நிலைமைகளின் அடிப்படையில் AI-இயக்கப்படும் பயிர் தேர்வு\n- **உர உகப்பாக்கம்:** அதிகபட்ச விளைச்சலுக்கான புத்திசாலித்தனமான உர பரிந்துரைகள்\n- **நோய் கண்டறிதல்:** தாவர நோய்களை அடையாளம் காண கணினி பார்வை",
        "ux_header": "🔧 பயனர் அனுபவம்",
        "ux_list": "- **ஊடாடும் இடைமுகம்:** பயன்படுத்த எளிதான ஸ்லைடர்கள் மற்றும் இன்புட் ஃபீல்ட்\n- **உண்மையான நேர பகுப்பாய்வு:** உடனடி கணிப்புகள் மற்றும் பரிந்துரைகள்\n- **கல்வி உள்ளடக்கம்:** விரிவான விளக்கங்கள் மற்றும் விவசாய குறிப்புகள்",
        "institution_title": "🏫 நிறுவனம்",
        "institution_text": "<strong>இந்திய தகவல் தொழில்நுட்ப நிறுவனம், ராய்ச்சூர்</strong><br>விவசாய தொழில்நுட்பம் மற்றும் நிலையான விவசாய தீர்வுகளில் புதுமை.",
        "acknowledgements_title": "🙏 அங்கீகாரங்கள்",
        "acknowledgements_text": "இந்த விவசாய AI தீர்வை உருவாக்குவதில் அவர்களின் ஆதரவு மற்றும் வழிகாட்டுதலுக்காக எங்கள் ஆசிரிய ஆலோசகர்களான டாக்டர். பிரியோத்யுதி பிரதான் மற்றும் ஐஐஐடி ராய்ச்சூர் சமூகத்திற்கு சிறப்பு நன்றி.",
        "footer_title": "🌱 **டீப்அக்ரோ**",
        "footer_slogan": "AI மற்றும் ML உடன் விவசாயத்தை மேம்படுத்துதல்",
        "footer_credit": "❤️ டீம் டீப்அக்ரோவால் கட்டப்பட்டது | ஐஐஐடி ராய்ச்சூர் | 2025"
      }
    }, 
  "tel": {
    "page_title": "డీప్‌అగ్రో - స్మార్ట్ వ్యవసాయం",
    "sidebar_title": "🌾 నావిగేషన్",
    "nav_home": "🏠 హోమ్",
    "nav_crop": "🌾 పంట అంచనా",
    "nav_fertilizer": "🧪 ఎరువుల సిఫార్సు",
    "nav_chat": "🤖 డీప్‌అగ్రో AI సహాయకుడు",
    "nav_disease": "🔬 వ్యాధి గుర్తింపు",
    "nav_about": "👥 మా గురించి",
    "home": {
      "header_logo": "🌱 డీప్‌అగ్రో",
      "header_tagline": "AI మరియు ML తో స్మార్ట్ వ్యవసాయ పరిష్కారాలు",
      "welcome_header": "🌟 వ్యవసాయ భవిష్యత్తుకు స్వాగతం!",
      "welcome_text": "డీప్‌అగ్రో వ్యవసాయ పద్ధతులలో విప్లవం సృష్టించడానికి అత్యాధునిక **యంత్ర అభ్యాసం** మరియు **కృత్రిమ మేధస్సు**ను ఉపయోగించుకుంటుంది. మా ప్లాట్‌ఫారమ్ దీని కోసం తెలివైన అంతర్దృష్టులను అందిస్తుంది:",
      "card_crop_title": "🌾 స్మార్ట్ పంట సిఫార్సు",
      "card_crop_desc": "అధునాతన ML అల్గారిథమ్‌లను ఉపయోగించి నేల పరిస్థితులు, వాతావరణం మరియు పోషకాల ఆధారంగా వ్యక్తిగతీకరించిన పంట సూచనలను పొందండి.",
      "card_fert_title": "🧪 ఎరువుల ఆప్టిమైజేషన్",
      "card_fert_desc": "పనితీరును పెంచుతూ పర్యావరణ ప్రభావాన్ని తగ్గించడానికి ఖచ్చితమైన ఎరువుల సిఫార్సులను పొందండి.",
      "card_disease_title": "🔬 AI-ఆధారిత వ్యాధి గుర్తింపు",
      "card_disease_desc": "అత్యాధునిక CNN డీప్ లెర్నింగ్ మోడళ్లను ఉపయోగించి తక్షణ వ్యాధి గుర్తింపు కోసం ఆకు చిత్రాలను అప్‌లోడ్ చేయండి।",
      "metrics_header": "🚀 ముఖ్య లక్షణాలు",
      "metric_crops": "పంట రకాలు",
      "metric_fertilizers": "ఎరువుల రకాలు",
      "metric_accuracy": "ఖచ్చితత్వం",
      "metric_power": "ఆధారితం",
      "why_choose_title": "🌟 డీప్‌అగ్రోను ఎందుకు ఎంచుకోవాలి?",
      "why_choose_desc": "గరిష్ట దిగుబడి మరియు స్థిరత్వం కోసం సాంప్రదాయ వ్యవసాయాన్ని స్మార్ట్, డేటా-ఆధారిత నిర్ణయాలుగా మార్చే మా అత్యాధునిక AI సాంకేతికతతో వ్యవసాయ భవిష్యత్తును అనుభవించండి.",
      "benefit_precision_title": "ఖచ్చితమైన వ్యవసాయం",
      "benefit_precision_desc": "సరైన పంట ఎంపిక మరియు వనరుల నిర్వహణ కోసం ఖచ్చితమైన ఖచ్చితత్వంతో డేటా-ఆధారిత నిర్ణయాలు తీసుకోండి.",
      "benefit_sustain_title": "స్థిరమైన వ్యవసాయం",
      "benefit_sustain_desc": "తెలివైన సిఫార్సుల ద్వారా ఉత్పాదకతను పెంచుతూ వ్యర్థాలు మరియు పర్యావరణ ప్రభావాన్ని తగ్గించండి.",
      "benefit_realtime_title": "రియల్-టైమ్ విశ్లేషణ",
      "benefit_realtime_desc": "అధునాతన యంత్ర అభ్యాస అల్గారిథమ్‌లు మరియు కంప్యూటర్ దృష్టి ద్వారా ఆధారితమైన తక్షణ అంతర్దృష్టులు మరియు అంచనాలను పొందండి."
    },
    "crop_prediction": {
      "main_title": "🌾 తెలివైన పంట సిఫార్సు వ్యవస్థ",
      "subtitle": "మీ నేల మరియు పర్యావరణ పరిస్థితుల ఆధారంగా AI-ఆధారిత పంట సూచనలను పొందండి.",
      "expander_header": "ℹ️ పంట అంచనా పారామీటర్‌లను అర్థం చేసుకోవడం",
      "expander_info_text": "మా AI మోడల్ మీ భూమికి ఉత్తమ పంటలను సిఫార్సు చేయడానికి అనేక అంశాలను విశ్లేషిస్తుంది. ప్రతి పారామీటర్ పంట అనుకూలతను నిర్ణయించడంలో కీలక పాత్ర పోషిస్తుంది:",
      "how_it_works": "📊 **ఇది ఎలా పనిచేస్తుంది:** మా యంత్ర అభ్యాస అల్గారిథమ్ మీ ఇన్‌పుట్ డేటాను ప్రాసెస్ చేస్తుంది మరియు వ్యక్తిగతీకరించిన సిఫార్సులను అందించడానికి వేలకొలది విజయవంతమైన పంట కలయికలతో దానిని పోల్చి చూస్తుంది.",
      "env_factors_header": "🌡️ పర్యావరణ కారకాలు",
      "temp_label": "🌡️ ఉష్ణోగ్రత (°C)",
      "temp_info": "<strong>ఉష్ణోగ్రత ప్రభావం:</strong> డిగ్రీల సెల్సియస్‌లో పరిసర ఉష్ణోగ్రత. వివిధ పంటలు వేర్వేరు ఉష్ణోగ్రత పరిధులలో పెరుగుతాయి - ఉష్ణమండల పంటలు 25-35°C ను ఇష్టపడతాయి, అయితే సమశీతోష్ణ పంటలు 15-25°C ను ఇష్టపడతాయి.",
      "hum_label": "💧 తేమ (%)",
      "hum_info": "<strong>తేమ ప్రభావం:</strong> గాలిలో సాపేక్ష తేమ శాతం. అధిక తేమ (>70%) వరి వంటి పంటలకు అనుకూలంగా ఉంటుంది, అయితే తక్కువ తేమ (<50%) గోధుమలు మరియు బార్లీ వంటి వాటికి మంచిది.",
      "rain_label": "🌧️ వర్షపాతం (మి.మీ)",
      "rain_info": "<strong>వర్షపాతం ప్రభావం:</strong> మిల్లీమీటర్లలో సగటు వర్షపాతం మొత్తం. వరికి 150-300 మిమీ అవసరం, గోధుమకు 30-100 మిమీ అవసరం, అయితే కరువు-నిరోధక పంటలు <50 మిమీ తో కూడా మనుగడ సాగించగలవు.",
      "ph_label": "⚗️ నేల pH స్థాయి",
      "ph_info": "<strong>pH ప్రభావం:</strong> నేల pH విలువ ఆమ్లత్వం/క్షారతను కొలుస్తుంది. చాలా పంటలు 6.0-7.5 (స్వల్పంగా ఆమ్లం నుండి తటస్థం) ను ఇష్టపడతాయి. ఆమ్ల నేల (<6) బ్లూబెర్రీలకు అనుకూలం, అయితే క్షార నేల (>7.5) ఆస్పరాగస్‌కు అనుకూలం.",
      "nutrients_header": "🧪 నేల పోషకాలు (NPK విలువలు)",
      "n_label": "🔵 నత్రజని (N) కంటెంట్",
      "n_info": "<strong>నత్రజని (N) పాత్ర:</strong> ఆకు పెరుగుదల మరియు క్లోరోఫిల్ ఉత్పత్తికి అవసరం. ఆకుకూరల కూరగాయలకు అధిక N (80-120) అవసరం, అయితే వేరు కూరగాయలకు మధ్యస్థ N (40-80) అవసరం.",
      "p_label": "🟡 భాస్వరం (P) కంటెంట్",
      "p_info": "<strong>భాస్వరం (P) పాత్ర:</strong> వేరు పెరుగుదల మరియు పుష్పించడానికి కీలకం. పండ్ల పంటలకు అధిక P (60-100) అవసరం, అయితే గడ్డికి తక్కువ P (20-40) అవసరం.",
      "k_label": "🔴 పొటాషియం (K) కంటెంట్",
      "k_info": "<strong>పొటాషియం (K) పాత్ర:</strong> వ్యాధి నిరోధకత మరియు నీటి నియంత్రణకు ముఖ్యమైనది. వేరు కూరగాయలు మరియు పండ్లకు అధిక K (80-150) అవసరం, అయితే ధాన్యాలకు మధ్యస్థ K (40-80) అవసరం.",
      "summary_header": "📊 ప్రస్తుత ఇన్‌పుట్ సారాంశం",
      "summary_temp": "🌡️ **ఉష్ణోగ్రత:**",
      "summary_hum": "💧 **తేమ:**",
      "summary_rain": "🌧️ **వర్షపాతం:**",
      "summary_ph": "⚗️ **pH స్థాయి:**",
      "summary_n": "🔵 **నత్రజని (N):**",
      "summary_p": "🟡 **భాస్వరం (P):**",
      "summary_k": "🔴 **పొటాషియం (K):**",
      "reference_header": "📋 ఆదర్శ పరిధి సూచన",
      "ref_text": "<strong>ఉత్తమ పెరుగుదల పరిస్థితులు:</strong><br>• **ఉష్ణోగ్రత:** 20-30°C (చాలా పంటలు)<br>• **తేమ:** 40-70% (ఉత్తమ పరిధి)<br>• **వర్షపాతం:** 50-200mm (పంటను బట్టి మారుతుంది)<br>• **pH:** 6.0-7.5 (తటస్థం నుండి స్వల్పంగా ఆమ్లం)<br>• **NPK:** ఆరోగ్యకరమైన పెరుగుదల కోసం సమతుల్య నిష్పత్తి",
      "warning_temp": "🌡️ ఉష్ణోగ్రత సాధారణ పెరుగుదల పరిధి (5-45°C) వెలుపల ఉంది",
      "warning_hum": "💧 తేమ స్థాయి చాలా పంటలకు సవాలుగా ఉండవచ్చు",
      "warning_ph": "⚗️ pH స్థాయి చాలా తీవ్రంగా ఉంది మరియు పంట ఎంపికలను పరిమితం చేయవచ్చు",
      "warning_n": "🔵 చాలా అధిక నత్రజని స్థాయి అధిక వృక్షసంపద పెరుగుదలకు దారితీయవచ్చు",
      "warning_p": "🟡 అధిక భాస్వరం స్థాయి ఇతర పోషకాల శోషణకు ఆటంకం కలిగించవచ్చు",
      "warning_k": "🔴 చాలా అధిక పొటాషియం స్థాయి నేల నిర్మాణాన్ని ప్రభావితం చేయవచ్చు",
      "warnings_header": "⚠️ ఇన్‌పుట్ హెచ్చరికలు:",
      "validation_header": "✅ ధృవీకరణ స్థితి",
      "validation_text": "అన్ని ఇన్‌పుట్ విలువలు ఆమోదయోగ్యమైన పరిమితులలో ఉన్నాయి! మీ పరిస్థితులు పంట సాగుకు చాలా మంచివి.",
      "predict_button": "🔮 ఉత్తమ పంటను అంచనా వేయండి",
      "loading_1": "నేల పరిస్థితులను విశ్లేషిస్తోంది...",
      "loading_2": "పర్యావరణ డేటాను ప్రాసెస్ చేస్తోంది...",
      "loading_3": "పంట డేటాబేస్‌తో సరిపోల్చడం...",
      "loading_4": "సిఫార్సులను ఖరారు చేస్తోంది...",
      "result_header": "🎯 సిఫార్సు చేయబడిన పంట:",
      "result_confidence": "📊 విశ్వాస స్కోర్:",
      "result_quality": "🌟 సరిపోలే నాణ్యత:",
      "quality_excellent": "అద్భుతమైనది",
      "quality_good": "మంచిది",
      "quality_fair": "సమం",
      "top_3_header": "📈 టాప్ 3 పంట సిఫార్సులు",
      "crop_season": "పంట కాలం",
      "crop_water": "నీటి అవసరం",
      "crop_match": "సరిపోలిక",
      "crop_suitability": "అనుకూలత",
      "personalized_tips_header": "💡 వ్యక్తిగతీకరించిన వ్యవసాయ చిట్కాలు",
      "tips_climate_header": "🌡️ వాతావరణ పరిగణనలు",
      "tips_temp_high": "<strong>🌡️ అధిక ఉష్ణోగ్రత హెచ్చరిక:</strong> వేడి-నిరోధక రకాలు, షేడ్ నెట్‌లు మరియు తరచుగా నీటిపారుదల సమయాలను పరిగణించండి. నీటి సామర్థ్యం కోసం డ్రిప్ నీటిపారుదలని ఏర్పాటు చేయండి.",
      "tips_temp_low": "<strong>❄️ చల్లని ఉష్ణోగ్రత:</strong> చల్లని-వాతావరణ పంటలకు అనువైనది. వరుస కవర్లు మరియు గ్రీన్‌హౌస్ వ్యవసాయం వంటి మంచు రక్షణ చర్యలను పరిగణించండి.",
      "tips_temp_ok": "<strong>🌡️ సరైన ఉష్ణోగ్రత:</strong> చాలా పంట రకాలకు సరైన పరిస్థితులు. నిరంతరం నీటిపారుదలని నిర్వహించండి మరియు తెగుళ్లను పర్యవేక్షించండి.",
      "tips_hum_high": "<strong>💧 అధిక తేమ హెచ్చరిక:</strong> ఫంగల్ వ్యాధులు రాకుండా సరైన మొక్కల దూరం మరియు వెంటిలేషన్ ఉండేలా చూసుకోండి. ఫంగసైడ్ చికిత్సలను పరిగణించండి.",
      "tips_hum_low": "<strong>🏜️ తక్కువ తేమ హెచ్చరిక:</strong> నేల తేమను నిలుపుకోవడానికి మల్చింగ్ మరియు తరచుగా తేలికపాటి నీటిపారుదలని పరిగణించండి. తేమను నిలుపుకునే పద్ధతులను ఉపయోగించండి.",
      "tips_hum_ok": "<strong>💧 మంచి తేమ స్థాయి:</strong> ఆరోగ్యకరమైన మొక్కల పెరుగుదలకు అనుకూలమైన పరిస్థితులు. సరైన మొక్కల పెరుగుదల కోసం పర్యవేక్షించండి.",
      "tips_soil_header": "🧪 నేల నిర్వహణ",
      "tips_ph_acidic": "<strong>⚗️ ఆమ్ల నేల:</strong> pH పెంచడానికి సున్నం కలపడం గురించి ఆలోచించండి. అల్యూమినియం విషపూరితం కోసం పరీక్షించండి మరియు నేల నిర్మాణాన్ని మెరుగుపరచడానికి సేంద్రీయ పదార్థాన్ని జోడించండి.",
      "tips_ph_alkaline": "<strong>⚗️ క్షార నేల:</strong> pH తగ్గించడానికి సల్ఫర్ లేదా సేంద్రీయ పదార్థాన్ని కలపడం గురించి ఆలోచించండి. సూక్ష్మపోషకాల లోపాలను పర్యవేక్షించండి.",
      "tips_ph_ok": "<strong>⚗️ సరైన pH పరిధి:</strong> పోషకాల లభ్యతకు సరైన పరిస్థితులు. సాధారణ సేంద్రీయ సవరణలతో నేల ఆరోగ్యాన్ని నిర్వహించండి.",
      "tips_n_low": "<strong>🔵 తక్కువ నత్రజని:</strong> యూరియా లేదా సేంద్రీయ ఎరువు వంటి నత్రజని-రిచ్ ఎరువులను పరిగణించండి. మెరుగైన శోషణ కోసం విభాజిత మోతాదులలో వర్తించండి.",
      "tips_n_high": "<strong>🔵 అధిక నత్రజని:</strong> అధిక వృక్షసంపద పెరుగుదలకు దారితీయవచ్చు. జాగ్రత్తగా పర్యవేక్షించండి మరియు అవసరమైతే నత్రజని ఇన్‌పుట్‌ను తగ్గించండి.",
      "tips_p_low": "<strong>🟡 తక్కువ భాస్వరం:</strong> DAP లేదా రాక్ ఫాస్ఫేట్ వర్తించండి. వేరు పెరుగుదల మరియు పుష్పించడానికి అవసరం.",
      "tips_k_low": "<strong>🔴 తక్కువ పొటాషియం:</strong> MOP (మ్యూరేట్ ఆఫ్ పొటాష్) వర్తించండి. వ్యాధి నిరోధకత మరియు నీటి నియంత్రణకు ముఖ్యమైనది.",
      "summary_box_header": "🌟 మీ వ్యక్తిగతీకరించిన పంట సిఫార్సు సారాంశం",
      "summary_box_text": "మీ నేల మరియు పర్యావరణ పరిస్థితుల మా AI విశ్లేషణ ఆధారంగా, **{}** మీ భూమికి **{:.1f}% విశ్వాస స్కోర్**తో అత్యంత అనువైన పంట.",
      "summary_match_quality": "🎯 సరిపోలే నాణ్యత",
      "summary_growth_potential": "🌱 పెరుగుదల సామర్థ్యం",
      "summary_econ_viability": "💰 ఆర్థిక సాధ్యత",
      "growth_high": "అధికం",
      "growth_medium": "మధ్యస్థం",
      "growth_moderate": "మధ్యస్థం",
      "econ_prof": "లాభదాయకం",
      "econ_good": "మంచిది"
    },
    "fertilizer_recommendation": {
      "main_title": "🧪 ఎరువుల సిఫార్సు వ్యవస్థ",
      "subtitle": "మీ పంట మరియు నేల పరిస్థితుల ఆధారంగా ఉత్తమ ఎరువుల సూచనలను పొందండి.",
      "section_info": "🌱 పంట మరియు నేల సమాచారం",
      "section_env": "🌡️ పర్యావరణ పరిస్థితులు",
      "section_nutrients": "🧪 ప్రస్తుత నేల పోషకాలు",
      "crop_type_label": "పంట రకం",
      "soil_type_label": "నేల రకం",
      "temp_label": "ఉష్ణోగ్రత (°C)",
      "hum_label": "తేమ (%)",
      "moisture_label": "నేల తేమ (%)",
      "nitrogen_label": "నత్రజని కంటెంట్",
      "phosphorus_label": "భాస్వరం కంటెంట్",
      "potassium_label": "పొటాషియం కంటెంట్",
      "nutrient_status_header": "📊 పోషక స్థితి",
      "low": "🔴 తక్కువ",
      "medium": "🟡 మధ్యస్థం",
      "high": "🟢 అధికం",
      "predict_button": "💡 ఎరువుల సిఫార్సు పొందండి",
      "result_header": "🎯 సిఫార్సు చేయబడిన ఎరువు:",
      "result_confidence": "📊 విశ్వాసం:",
      "result_info_pre": "",
      "result_info_in": " నేలలోని ",
      "result_info_apply": "- **{}** ఎరువును వర్తించండి",
      "result_info_tips": "- పరిమాణాన్ని నిర్ణయించేటప్పుడు ప్రస్తుత పోషకాల స్థాయిని పరిగణించండి\n- తగిన పెరుగుదల దశలో వర్తించండి\n- నేల తేమ మరియు వాతావరణ పరిస్థితులను పర్యవేక్షించండి",
      "error_message": "అంచనాలో లోపం. దయచేసి మీ ఇన్‌పుట్‌లను తనిఖీ చేయండి."
    },
    "disease_detection": {
      "main_title": "🔬 వ్యాధి గుర్తింపు",
      "subtitle": "డీప్ లెర్నింగ్ CNN మోడళ్లను ఉపయోగించి తక్షణ వ్యాధి గుర్తింపు కోసం ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి.",
      "upload_header": "📷 మొక్క ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
      "upload_guidelines_title": "📸 చిత్ర అప్‌లోడ్ మార్గదర్శకాలు:",
      "upload_guidelines_text": "✓ స్పష్టమైన, బాగా వెలుతురు ఉన్న ఆకు చిత్రాలు<br>✓ ప్రభావిత ప్రాంతాలు లేదా లక్షణాలపై దృష్టి పెట్టండి<br>✓ మద్దతు ఉన్న ఫార్మాట్‌లు: JPG, PNG, JPEG<br>✓ గరిష్ట పరిమాణం: 10MB",
      "file_uploader_label": "ఒక ఆకు చిత్రాన్ని ఎంచుకోండి...",
      "file_uploader_help": "మొక్క ఆకు యొక్క స్పష్టమైన చిత్రాన్ని అప్‌లోడ్ చేయండి",
      "uploaded_image_caption": "📷 అప్‌లోడ్ చేయబడిన ఆకు చిత్రం",
      "analyze_button": "🔍 వ్యాధుల కోసం విశ్లేషించండి",
      "loading_message": "🧠 AI చిత్రాన్ని విశ్లేషిస్తోంది...",
      "analysis_complete": "విశ్లేషణ పూర్తయింది!",
      "result_header": "🎯 అంచనా వేయబడిన వ్యాధి:",
      "result_confidence": "📊 విశ్వాసం:",
      "disease_warning": "❗ మీ మొక్కకు వ్యాధి సోకి ఉండవచ్చు. దయచేసి ధృవీకరణ కోసం నిపుణుడిని సంప్రదించండి.",
      "healthy_message": "✅ మొక్క ఆరోగ్యంగా ఉన్నట్లు కనిపిస్తుంది!"
    },
    "about_page": {
      "main_title": "👥 మా గురించి",
      "subtitle": "స్మార్ట్ వ్యవసాయ విప్లవం వెనుక ఉన్న వినూత్న బృందాన్ని కలవండి!",
      "mission_title": "🌟 మా లక్ష్యం",
      "mission_text": "డీప్‌అగ్రో అధునాతన AI మరియు యంత్ర అభ్యాస సాంకేతికతలతో సాంప్రదాయ వ్యవసాయాన్ని మార్చడానికి అంకితం చేయబడింది. మా లక్ష్యం మెరుగైన పంట ఎంపిక, సరైన ఎరువుల వాడకం మరియు ముందస్తు వ్యాధి గుర్తింపు కోసం తెలివైన అంతర్దృష్టులతో రైతులను శక్తివంతం చేయడమే.",
      "team_header": "👨‍💻 మా అభివృద్ధి బృందం",
      "team_desc": "ఐఐఐటి రాయచూర్‌కు చెందిన విద్యార్థుల ఉద్వేగభరితమైన బృందం, సాంకేతికతతో వ్యవసాయంలో విప్లవం సృష్టించడానికి కలిసి పనిచేస్తుంది.",
      "tech_stack_header": "🛠️ సాంకేతిక స్టాక్",
      "ml_title": "🤖 యంత్ర అభ్యాసం",
      "ml_text": "• రాండమ్ ఫారెస్ట్ క్లాసిఫైయర్<br>• స్కికిట్-లెర్న్<br>• నంపై & పాండాలు<br>• ఫీచర్ ఇంజనీరింగ్",
      "web_title": "🌐 వెబ్ ఫ్రేమ్‌వర్క్",
      "web_text": "• స్ట్రీమ్‌లిట్<br>• పైథాన్ బ్యాకెండ్<br>• ఇంటరాక్టివ్ UI/UX<br>• రియల్-టైమ్ ప్రాసెసింగ్",
      "data_title": "📊 డేటా & విజువలైజేషన్",
      "data_text": "• చార్ట్‌ల కోసం ప్లోట్లీ<br>• ఇమేజ్ ప్రాసెసింగ్ కోసం PIL<br>• కస్టమ్ CSS స్టైలింగ్<br>• రెస్పాన్సివ్ డిజైన్",
      "features_header": "✨ ముఖ్య లక్షణాలు",
      "smart_pred_header": "🎯 స్మార్ట్ అంచనాలు",
      "smart_pred_list": "- **పంట సిఫార్సు:** నేల మరియు వాతావరణ పరిస్థితుల ఆధారంగా AI-ఆధారిత పంట ఎంపిక\n- **ఎరువుల ఆప్టిమైజేషన్:** గరిష్ట దిగుబడి కోసం తెలివైన ఎరువుల సిఫార్సులు\n- **వ్యాధి గుర్తింపు:** మొక్కల వ్యాధి గుర్తింపు కోసం కంప్యూటర్ దృష్టి",
      "ux_header": "🔧 వినియోగదారు అనుభవం",
      "ux_list": "- **ఇంటరాక్టివ్ ఇంటర్‌ఫేస్:** ఉపయోగించడానికి సులభమైన స్లైడర్‌లు మరియు ఇన్‌పుట్ ఫీల్డ్‌లు\n- **రియల్-టైమ్ విశ్లేషణ:** తక్షణ అంచనాలు మరియు సిఫార్సులు\n- **విద్యాపరమైన కంటెంట్:** వివరణాత్మక వివరణలు మరియు వ్యవసాయ చిట్కాలు",
      "institution_title": "🏫 సంస్థ",
      "institution_text": "<strong>ఇండియన్ ఇన్‌స్టిట్యూట్ ఆఫ్ ఇన్ఫర్మేషన్ టెక్నాలజీ, రాయచూర్</strong><br>వ్యవసాయ సాంకేతికత మరియు స్థిరమైన వ్యవసాయ పరిష్కారాలలో ఆవిష్కరణ.",
      "acknowledgements_title": "🙏 అభినందనలు",
      "acknowledgements_text": "ఈ వ్యవసాయ AI పరిష్కారాన్ని అభివృద్ధి చేయడంలో వారి మద్దతు మరియు మార్గదర్శకత్వం కోసం మా ఫ్యాకల్టీ సలహాదారులు డా. ప్రియోద్యుతి ప్రధాన్ మరియు ఐఐఐటి రాయచూర్ కమ్యూనిటీకి ప్రత్యేక ధన్యవాదాలు.",
      "footer_title": "🌱 **డీప్‌అగ్రో**",
      "footer_slogan": "AI మరియు ML తో వ్యవసాయాన్ని శక్తివంతం చేస్తోంది",
      "footer_credit": "❤️ టీమ్ డీప్‌అగ్రో ద్వారా నిర్మించబడింది | ఐఐఐటి రాయచూర్ | 2025"
    }
  },
  "pa": {
    "page_title": "ਦੀਪਐਗਰੋ - ਸਮਾਰਟ ਖੇਤੀਬਾੜੀ",
    "sidebar_title": "🌾 ਨੇਵੀਗੇਸ਼ਨ",
    "nav_home": "🏠 ਹੋਮ",
    "nav_crop": "🌾 ਫਸਲ ਦੀ ਭਵਿੱਖਬਾਣੀ",
    "nav_fertilizer": "🧪 ਖਾਦ ਦੀ ਸਿਫਾਰਸ਼",
    "nav_disease": "🔬 ਰੋਗ ਦਾ ਪਤਾ ਲਗਾਉਣਾ",
    "nav_chat": "🤖 ਦੀਪਐਗਰੋ AI ਸਹਾਇਕ",
    "nav_about": "👥 ਸਾਡੇ ਬਾਰੇ",
    "home": {
      "header_logo": "🌱 ਦੀਪਐਗਰੋ",
      "header_tagline": "AI ਅਤੇ ML ਨਾਲ ਸਮਾਰਟ ਖੇਤੀਬਾੜੀ ਹੱਲ",
      "welcome_header": "🌟 ਖੇਤੀਬਾੜੀ ਦੇ ਭਵਿੱਖ ਵਿੱਚ ਤੁਹਾਡਾ ਸੁਆਗਤ ਹੈ!",
      "welcome_text": "ਦੀਪਐਗਰੋ ਖੇਤੀਬਾੜੀ ਦੇ ਤਰੀਕਿਆਂ ਵਿੱਚ ਕ੍ਰਾਂਤੀ ਲਿਆਉਣ ਲਈ ਅਤਿ-ਆਧੁਨਿਕ **ਮਸ਼ੀਨ ਲਰਨਿੰਗ** ਅਤੇ **ਆਰਟੀਫੀਸ਼ੀਅਲ ਇੰਟੈਲੀਜੈਂਸ** ਦਾ ਲਾਭ ਉਠਾਉਂਦਾ ਹੈ। ਸਾਡਾ ਪਲੇਟਫਾਰਮ ਇਸ ਲਈ ਬੁੱਧੀਮਾਨ ਅੰਤਰ-ਦ੍ਰਿਸ਼ਟੀਆਂ ਪ੍ਰਦਾਨ ਕਰਦਾ ਹੈ:",
      "card_crop_title": "🌾 ਸਮਾਰਟ ਫਸਲ ਦੀ ਸਿਫਾਰਸ਼",
      "card_crop_desc": "ਅਤਿ-ਆਧੁਨਿਕ ML ਐਲਗੋਰਿਦਮ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਮਿੱਟੀ ਦੀਆਂ ਸਥਿਤੀਆਂ, ਜਲਵਾਯੂ ਅਤੇ ਪੌਸ਼ਟਿਕ ਤੱਤਾਂ ਦੇ ਆਧਾਰ 'ਤੇ ਵਿਅਕਤੀਗਤ ਫਸਲ ਦੇ ਸੁਝਾਅ ਪ੍ਰਾਪਤ ਕਰੋ।",
      "card_fert_title": "🧪 ਖਾਦ ਦਾ ਅਨੁਕੂਲਨ",
      "card_fert_desc": "ਵਾਤਾਵਰਣ ਦੇ ਪ੍ਰਭਾਵ ਨੂੰ ਘਟਾਉਂਦੇ ਹੋਏ ਉਪਜ ਨੂੰ ਵੱਧ ਤੋਂ ਵੱਧ ਕਰਨ ਲਈ ਸਹੀ ਖਾਦ ਦੀਆਂ ਸਿਫਾਰਸ਼ਾਂ ਪ੍ਰਾਪਤ ਕਰੋ।",
      "card_disease_title": "🔬 AI-ਸੰਚਾਲਿਤ ਰੋਗ ਦਾ ਪਤਾ ਲਗਾਉਣਾ",
      "card_disease_desc": "ਅਤਿ-ਆਧੁਨਿਕ CNN ਡੀਪ ਲਰਨਿੰਗ ਮਾਡਲਾਂ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਤੁਰੰਤ ਰੋਗ ਦੀ ਪਛਾਣ ਲਈ ਪੱਤਿਆਂ ਦੀਆਂ ਤਸਵੀਰਾਂ ਅਪਲੋਡ ਕਰੋ।",
      "metrics_header": "🚀 ਮੁੱਖ ਵਿਸ਼ੇਸ਼ਤਾਵਾਂ",
      "metric_crops": "ਫਸਲਾਂ ਦੀਆਂ ਕਿਸਮਾਂ",
      "metric_fertilizers": "ਖਾਦਾਂ ਦੀਆਂ ਕਿਸਮਾਂ",
      "metric_accuracy": "ਸ਼ੁੱਧਤਾ",
      "metric_power": "ਸੰਚਾਲਿਤ",
      "why_choose_title": "🌟 ਦੀਪਐਗਰੋ ਕਿਉਂ ਚੁਣੋ?",
      "why_choose_desc": "ਸਾਡੀ ਅਤਿ-ਆਧੁਨਿਕ AI ਤਕਨਾਲੋਜੀ ਨਾਲ ਖੇਤੀਬਾੜੀ ਦੇ ਭਵਿੱਖ ਦਾ ਅਨੁਭਵ ਕਰੋ ਜੋ ਵੱਧ ਤੋਂ ਵੱਧ ਉਪਜ ਅਤੇ ਸਥਿਰਤਾ ਲਈ ਰਵਾਇਤੀ ਖੇਤੀਬਾੜੀ ਨੂੰ ਸਮਾਰਟ, ਡੇਟਾ-ਸੰਚਾਲਿਤ ਫੈਸਲਿਆਂ ਵਿੱਚ ਬਦਲ ਦਿੰਦੀ ਹੈ।",
      "benefit_precision_title": "ਸਟੀਕ ਖੇਤੀਬਾੜੀ",
      "benefit_precision_desc": "ਅਨੁਕੂਲ ਫਸਲ ਦੀ ਚੋਣ ਅਤੇ ਸਰੋਤ ਪ੍ਰਬੰਧਨ ਲਈ ਸਹੀ ਸ਼ੁੱਧਤਾ ਨਾਲ ਡੇਟਾ-ਸੰਚਾਲਿਤ ਫੈਸਲੇ ਲਓ।",
      "benefit_sustain_title": "ਟਿਕਾਊ ਖੇਤੀ",
      "benefit_sustain_desc": "ਬੁੱਧੀਮਾਨ ਸਿਫਾਰਸ਼ਾਂ ਦੁਆਰਾ ਉਤਪਾਦਕਤਾ ਨੂੰ ਵੱਧ ਤੋਂ ਵੱਧ ਕਰਦੇ ਹੋਏ ਰਹਿੰਦ-ਖੂੰਹਦ ਅਤੇ ਵਾਤਾਵਰਣ ਦੇ ਪ੍ਰਭਾਵ ਨੂੰ ਘਟਾਓ।",
      "benefit_realtime_title": "ਰੀਅਲ-ਟਾਈਮ ਵਿਸ਼ਲੇਸ਼ਣ",
      "benefit_realtime_desc": "ਐਡਵਾਂਸਡ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਐਲਗੋਰਿਦਮ ਅਤੇ ਕੰਪਿਊਟਰ ਵਿਜ਼ਨ ਦੁਆਰਾ ਸੰਚਾਲਿਤ ਤੁਰੰਤ ਅੰਤਰ-ਦ੍ਰਿਸ਼ਟੀਆਂ ਅਤੇ ਭਵਿੱਖਬਾਣੀਆਂ ਪ੍ਰਾਪਤ ਕਰੋ।"
    },
    "crop_prediction": {
      "main_title": "🌾 ਬੁੱਧੀਮਾਨ ਫਸਲ ਸਿਫਾਰਸ਼ ਪ੍ਰਣਾਲੀ",
      "subtitle": "ਆਪਣੀ ਮਿੱਟੀ ਅਤੇ ਵਾਤਾਵਰਣ ਦੀਆਂ ਸਥਿਤੀਆਂ ਦੇ ਆਧਾਰ 'ਤੇ AI-ਸੰਚਾਲਿਤ ਫਸਲ ਦੇ ਸੁਝਾਅ ਪ੍ਰਾਪਤ ਕਰੋ।",
      "expander_header": "ℹ️ ਫਸਲ ਦੀ ਭਵਿੱਖਬਾਣੀ ਦੇ ਮਾਪਦੰਡਾਂ ਨੂੰ ਸਮਝਣਾ",
      "expander_info_text": "ਸਾਡਾ AI ਮਾਡਲ ਤੁਹਾਡੀ ਜ਼ਮੀਨ ਲਈ ਸਭ ਤੋਂ ਵਧੀਆ ਫਸਲਾਂ ਦੀ ਸਿਫਾਰਸ਼ ਕਰਨ ਲਈ ਕਈ ਕਾਰਕਾਂ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰਦਾ ਹੈ। ਹਰੇਕ ਮਾਪਦੰਡ ਫਸਲ ਦੀ ਅਨੁਕੂਲਤਾ ਨਿਰਧਾਰਤ ਕਰਨ ਵਿੱਚ ਇੱਕ ਮਹੱਤਵਪੂਰਨ ਭੂਮਿਕਾ ਨਿਭਾਉਂਦਾ ਹੈ:",
      "how_it_works": "📊 **ਇਹ ਕਿਵੇਂ ਕੰਮ ਕਰਦਾ ਹੈ:** ਸਾਡਾ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਐਲਗੋਰਿਦਮ ਤੁਹਾਡੇ ਇਨਪੁਟ ਡੇਟਾ ਨੂੰ ਪ੍ਰੋਸੈਸ ਕਰਦਾ ਹੈ ਅਤੇ ਵਿਅਕਤੀਗਤ ਸਿਫਾਰਸ਼ਾਂ ਪ੍ਰਦਾਨ ਕਰਨ ਲਈ ਇਸਨੂੰ ਹਜ਼ਾਰਾਂ ਸਫਲ ਫਸਲੀ ਸੰਜੋਗਾਂ ਨਾਲ ਤੁਲਨਾ ਕਰਦਾ ਹੈ।",
      "env_factors_header": "🌡️ ਵਾਤਾਵਰਣ ਦੇ ਕਾਰਕ",
      "temp_label": "🌡️ ਤਾਪਮਾਨ (°C)",
      "temp_info": "<strong>ਤਾਪਮਾਨ ਦਾ ਪ੍ਰਭਾਵ:</strong> ਡਿਗਰੀ ਸੈਲਸੀਅਸ ਵਿੱਚ ਵਾਤਾਵਰਣ ਦਾ ਤਾਪਮਾਨ। ਵੱਖ-ਵੱਖ ਫਸਲਾਂ ਵੱਖ-ਵੱਖ ਤਾਪਮਾਨ ਰੇਂਜਾਂ ਵਿੱਚ ਵਧਦੀਆਂ ਹਨ - ਗਰਮ ਦੇਸ਼ਾਂ ਦੀਆਂ ਫਸਲਾਂ 25-35°C ਨੂੰ ਤਰਜੀਹ ਦਿੰਦੀਆਂ ਹਨ ਜਦੋਂ ਕਿ ਸ਼ਾਂਤ ਫਸਲਾਂ 15-25°C ਨੂੰ ਤਰਜੀਹ ਦਿੰਦੀਆਂ ਹਨ।",
      "hum_label": "💧 ਨਮੀ (%)",
      "hum_info": "<strong>ਨਮੀ ਦਾ ਪ੍ਰਭਾਵ:</strong> ਹਵਾ ਵਿੱਚ ਸਾਪੇਖਿਕ ਨਮੀ ਦਾ ਪ੍ਰਤੀਸ਼ਤ। ਉੱਚ ਨਮੀ (>70%) ਚੌਲਾਂ ਵਰਗੀਆਂ ਫਸਲਾਂ ਲਈ ਢੁਕਵੀਂ ਹੈ, ਜਦੋਂ ਕਿ ਘੱਟ ਨਮੀ (<50%) ਕਣਕ ਅਤੇ ਜੌਂ ਵਰਗੀਆਂ ਫਸਲਾਂ ਲਈ ਬਿਹਤਰ ਹੈ।",
      "rain_label": "🌧️ ਵਰਖਾ (ਮਿ.ਮੀ.)",
      "rain_info": "<strong>ਵਰਖਾ ਦਾ ਪ੍ਰਭਾਵ:</strong> ਮਿਲੀਮੀਟਰ ਵਿੱਚ ਔਸਤ ਵਰਖਾ ਦੀ ਮਾਤਰਾ। ਚੌਲਾਂ ਨੂੰ 150-300 ਮਿਲੀਮੀਟਰ ਦੀ ਲੋੜ ਹੁੰਦੀ ਹੈ, ਕਣਕ ਨੂੰ 30-100 ਮਿਲੀਮੀਟਰ ਦੀ ਲੋੜ ਹੁੰਦੀ ਹੈ, ਜਦੋਂ ਕਿ ਸੋਕੇ-ਰੋਧੀ ਫਸਲਾਂ <50 ਮਿਲੀਮੀਟਰ ਨਾਲ ਬਚ ਸਕਦੀਆਂ ਹਨ।",
      "ph_label": "⚗️ ਮਿੱਟੀ ਦਾ ਪੀ.ਐੱਚ. ਪੱਧਰ",
      "ph_info": "<strong>ਪੀਐੱਚ ਦਾ ਪ੍ਰਭਾਵ:</strong> ਮਿੱਟੀ ਦਾ ਪੀਐੱਚ ਮੁੱਲ ਤੇਜ਼ਾਬੀਤਾ/ਖਾਰੀਅਤ ਨੂੰ ਮਾਪਦਾ ਹੈ। ਜ਼ਿਆਦਾਤਰ ਫਸਲਾਂ 6.0-7.5 (ਥੋੜ੍ਹਾ ਤੇਜ਼ਾਬੀ ਤੋਂ ਨਿਰਪੱਖ) ਨੂੰ ਪਸੰਦ ਕਰਦੀਆਂ ਹਨ। ਤੇਜ਼ਾਬੀ ਮਿੱਟੀ (<6) ਬਲੂਬੇਰੀ ਲਈ ਢੁਕਵੀਂ ਹੈ, ਜਦੋਂ ਕਿ ਖਾਰੀ ਮਿੱਟੀ (>7.5) ਐਸਪੈਰਗਸ ਲਈ ਢੁਕਵੀਂ ਹੈ।",
      "nutrients_header": "🧪 ਮਿੱਟੀ ਦੇ ਪੌਸ਼ਟਿਕ ਤੱਤ (NPK ਮੁੱਲ)",
      "n_label": "🔵 ਨਾਈਟ੍ਰੋਜਨ (N) ਸਮੱਗਰੀ",
      "n_info": "<strong>ਨਾਈਟ੍ਰੋਜਨ (N) ਦੀ ਭੂਮਿਕਾ:</strong> ਪੱਤਿਆਂ ਦੇ ਵਿਕਾਸ ਅਤੇ ਕਲੋਰੋਫਿਲ ਦੇ ਉਤਪਾਦਨ ਲਈ ਜ਼ਰੂਰੀ। ਪੱਤੇਦਾਰ ਸਬਜ਼ੀਆਂ ਨੂੰ ਉੱਚ N (80-120) ਦੀ ਲੋੜ ਹੁੰਦੀ ਹੈ, ਜਦੋਂ ਕਿ ਜੜ੍ਹਾਂ ਵਾਲੀਆਂ ਸਬਜ਼ੀਆਂ ਨੂੰ ਮੱਧਮ N (40-80) ਦੀ ਲੋੜ ਹੁੰਦੀ ਹੈ।",
      "p_label": "🟡 ਫਾਸਫੋਰਸ (P) ਸਮੱਗਰੀ",
      "p_info": "<strong>ਫਾਸਫੋਰਸ (P) ਦੀ ਭੂਮਿਕਾ:</strong> ਜੜ੍ਹਾਂ ਦੇ ਵਿਕਾਸ ਅਤੇ ਫੁੱਲ ਆਉਣ ਲਈ ਮਹੱਤਵਪੂਰਨ। ਫਲਾਂ ਦੀਆਂ ਫਸਲਾਂ ਨੂੰ ਉੱਚ P (60-100) ਦੀ ਲੋੜ ਹੁੰਦੀ ਹੈ, ਜਦੋਂ ਕਿ ਘਾਹ ਨੂੰ ਘੱਟ P (20-40) ਦੀ ਲੋੜ ਹੁੰਦੀ ਹੈ।",
      "k_label": "🔴 ਪੋਟਾਸ਼ੀਅਮ (K) ਸਮੱਗਰੀ",
      "k_info": "<strong>ਪੋਟਾਸ਼ੀਅਮ (K) ਦੀ ਭੂਮਿਕਾ:</strong> ਰੋਗ ਪ੍ਰਤੀਰੋਧ ਅਤੇ ਪਾਣੀ ਦੇ ਨਿਯਮ ਲਈ ਮਹੱਤਵਪੂਰਨ। ਜੜ੍ਹਾਂ ਵਾਲੀਆਂ ਸਬਜ਼ੀਆਂ ਅਤੇ ਫਲਾਂ ਨੂੰ ਉੱਚ K (80-150) ਦੀ ਲੋੜ ਹੁੰਦੀ ਹੈ, ਜਦੋਂ ਕਿ ਅਨਾਜ ਨੂੰ ਮੱਧਮ K (40-80) ਦੀ ਲੋੜ ਹੁੰਦੀ ਹੈ।",
      "summary_header": "📊 ਮੌਜੂਦਾ ਇਨਪੁਟ ਸੰਖੇਪ",
      "summary_temp": "🌡️ **ਤਾਪਮਾਨ:**",
      "summary_hum": "💧 **ਨਮੀ:**",
      "summary_rain": "🌧️ **ਵਰਖਾ:**",
      "summary_ph": "⚗️ **ਪੀ.ਐੱਚ. ਪੱਧਰ:**",
      "summary_n": "🔵 **ਨਾਈਟ੍ਰੋਜਨ (N):**",
      "summary_p": "🟡 **ਫਾਸਫੋਰਸ (P):**",
      "summary_k": "🔴 **ਪੋਟਾਸ਼ੀਅਮ (K):**",
      "reference_header": "📋 ਆਦਰਸ਼ ਸੀਮਾ ਸੰਦਰਭ",
      "ref_text": "<strong>ਅਨੁਕੂਲ ਵਧਣ ਦੀਆਂ ਸਥਿਤੀਆਂ:</strong><br>• **ਤਾਪਮਾਨ:** 20-30°C (ਜ਼ਿਆਦਾਤਰ ਫਸਲਾਂ)<br>• **ਨਮੀ:** 40-70% (ਅਨੁਕੂਲ ਸੀਮਾ)<br>• **ਵਰਖਾ:** 50-200mm (ਫਸਲ ਅਨੁਸਾਰ ਵੱਖ-ਵੱਖ ਹੁੰਦਾ ਹੈ)<br>• **ਪੀਐੱਚ:** 6.0-7.5 (ਨਿਰਪੱਖ ਤੋਂ ਥੋੜ੍ਹਾ ਤੇਜ਼ਾਬੀ)<br>• **NPK:** ਸਿਹਤਮੰਦ ਵਿਕਾਸ ਲਈ ਸੰਤੁਲਿਤ ਅਨੁਪਾਤ",
      "warning_temp": "🌡️ ਤਾਪਮਾਨ ਆਮ ਵਿਕਾਸ ਸੀਮਾ (5-45°C) ਤੋਂ ਬਾਹਰ ਹੈ",
      "warning_hum": "💧 ਨਮੀ ਦਾ ਪੱਧਰ ਜ਼ਿਆਦਾਤਰ ਫਸਲਾਂ ਲਈ ਚੁਣੌਤੀਪੂਰਨ ਹੋ ਸਕਦਾ ਹੈ",
      "warning_ph": "⚗️ ਪੀ.ਐੱਚ. ਪੱਧਰ ਕਾਫ਼ੀ ਅਤਿਅੰਤ ਹੈ ਅਤੇ ਫਸਲ ਦੇ ਵਿਕਲਪਾਂ ਨੂੰ ਸੀਮਤ ਕਰ ਸਕਦਾ ਹੈ",
      "warning_n": "🔵 ਬਹੁਤ ਉੱਚ ਨਾਈਟ੍ਰੋਜਨ ਦਾ ਪੱਧਰ ਬਹੁਤ ਜ਼ਿਆਦਾ ਸਬਜ਼ੀਆਂ ਦੇ ਵਾਧੇ ਦਾ ਕਾਰਨ ਬਣ ਸਕਦਾ ਹੈ",
      "warning_p": "🟡 ਉੱਚ ਫਾਸਫੋਰਸ ਪੱਧਰ ਹੋਰ ਪੌਸ਼ਟਿਕ ਤੱਤਾਂ ਦੇ ਸਮਾਈ ਵਿੱਚ ਦਖਲ ਦੇ ਸਕਦਾ ਹੈ",
      "warning_k": "🔴 ਬਹੁਤ ਉੱਚ ਪੋਟਾਸ਼ੀਅਮ ਪੱਧਰ ਮਿੱਟੀ ਦੀ ਬਣਤਰ ਨੂੰ ਪ੍ਰਭਾਵਿਤ ਕਰ ਸਕਦਾ ਹੈ",
      "warnings_header": "⚠️ ਇਨਪੁਟ ਚੇਤਾਵਨੀਆਂ:",
      "validation_header": "✅ ਪ੍ਰਮਾਣਿਕਤਾ ਸਥਿਤੀ",
      "validation_text": "ਸਾਰੇ ਇਨਪੁਟ ਮੁੱਲ ਸਵੀਕਾਰਯੋਗ ਸੀਮਾਵਾਂ ਦੇ ਅੰਦਰ ਹਨ! ਤੁਹਾਡੀਆਂ ਸਥਿਤੀਆਂ ਫਸਲ ਦੀ ਕਾਸ਼ਤ ਲਈ ਬਹੁਤ ਵਧੀਆ ਹਨ।",
      "predict_button": "🔮 ਸਭ ਤੋਂ ਵਧੀਆ ਫਸਲ ਦੀ ਭਵਿੱਖਬਾਣੀ ਕਰੋ",
      "loading_1": "ਮਿੱਟੀ ਦੀਆਂ ਸਥਿਤੀਆਂ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰ ਰਿਹਾ ਹੈ...",
      "loading_2": "ਵਾਤਾਵਰਣ ਦੇ ਡੇਟਾ ਨੂੰ ਪ੍ਰੋਸੈਸ ਕਰ ਰਿਹਾ ਹੈ...",
      "loading_3": "ਫਸਲ ਡੇਟਾਬੇਸ ਨਾਲ ਮੇਲ ਖਾਂਦਾ ਹੈ...",
      "loading_4": "ਸਿਫਾਰਸ਼ਾਂ ਨੂੰ ਅੰਤਿਮ ਰੂਪ ਦੇ ਰਿਹਾ ਹੈ...",
      "result_header": "🎯 ਸਿਫਾਰਸ਼ ਕੀਤੀ ਫਸਲ:",
      "result_confidence": "📊 ਵਿਸ਼ਵਾਸ ਸਕੋਰ:",
      "result_quality": "🌟 ਮੈਚ ਗੁਣਵੱਤਾ:",
      "quality_excellent": "ਸ਼ਾਨਦਾਰ",
      "quality_good": "ਚੰਗਾ",
      "quality_fair": "ਠੀਕ",
      "top_3_header": "📈 ਚੋਟੀ ਦੀਆਂ 3 ਫਸਲਾਂ ਦੀਆਂ ਸਿਫਾਰਸ਼ਾਂ",
      "crop_season": "ਸੀਜ਼ਨ",
      "crop_water": "ਪਾਣੀ ਦੀ ਲੋੜ",
      "crop_match": "ਮੇਲ",
      "crop_suitability": "ਅਨੁਕੂਲਤਾ",
      "personalized_tips_header": "💡 ਵਿਅਕਤੀਗਤ ਖੇਤੀ ਦੇ ਸੁਝਾਅ",
      "tips_climate_header": "🌡️ ਜਲਵਾਯੂ ਵਿਚਾਰ",
      "tips_temp_high": "<strong>🌡️ ਉੱਚ ਤਾਪਮਾਨ ਚੇਤਾਵਨੀ:</strong> ਗਰਮੀ-ਰੋਧਕ ਕਿਸਮਾਂ, ਸ਼ੇਡ ਨੈੱਟ, ਅਤੇ ਅਕਸਰ ਸਿੰਚਾਈ ਦੇ ਸਮੇਂ ਬਾਰੇ ਵਿਚਾਰ ਕਰੋ। ਪਾਣੀ ਦੀ ਕੁਸ਼ਲਤਾ ਲਈ ਡ੍ਰਿੱਪ ਸਿੰਚਾਈ ਸਥਾਪਿਤ ਕਰੋ।",
      "tips_temp_low": "<strong>❄️ ਠੰਡਾ ਤਾਪਮਾਨ:</strong> ਠੰਡੇ-ਮੌਸਮ ਦੀਆਂ ਫਸਲਾਂ ਲਈ ਆਦਰਸ਼। ਕਤਾਰ ਕਵਰ ਅਤੇ ਗ੍ਰੀਨਹਾਉਸ ਖੇਤੀ ਵਰਗੇ ਠੰਡ ਸੁਰੱਖਿਆ ਉਪਾਵਾਂ ਬਾਰੇ ਵਿਚਾਰ ਕਰੋ।",
      "tips_temp_ok": "<strong>🌡️ ਅਨੁਕੂਲ ਤਾਪਮਾਨ:</strong> ਜ਼ਿਆਦਾਤਰ ਫਸਲਾਂ ਦੀਆਂ ਕਿਸਮਾਂ ਲਈ ਸੰਪੂਰਨ ਸਥਿਤੀਆਂ। ਨਿਯਮਤ ਤੌਰ 'ਤੇ ਪਾਣੀ ਦੇਣਾ ਜਾਰੀ ਰੱਖੋ ਅਤੇ ਕੀੜਿਆਂ ਦੀ ਨਿਗਰਾਨੀ ਕਰੋ।",
      "tips_hum_high": "<strong>💧 ਉੱਚ ਨਮੀ ਚੇਤਾਵਨੀ:</strong> ਫੰਗਲ ਰੋਗਾਂ ਨੂੰ ਰੋਕਣ ਲਈ ਸਹੀ ਪੌਦਿਆਂ ਦੀ ਦੂਰੀ ਅਤੇ ਹਵਾਦਾਰੀ ਨੂੰ ਯਕੀਨੀ ਬਣਾਓ। ਉੱਲੀਨਾਸ਼ਕ ਇਲਾਜਾਂ ਬਾਰੇ ਵਿਚਾਰ ਕਰੋ।",
      "tips_hum_low": "<strong>🏜️ ਘੱਟ ਨਮੀ ਚੇਤਾਵਨੀ:</strong> ਮਿੱਟੀ ਦੀ ਨਮੀ ਬਰਕਰਾਰ ਰੱਖਣ ਲਈ ਮਲਚਿੰਗ ਅਤੇ ਲਗਾਤਾਰ ਹਲਕੀ ਸਿੰਚਾਈ ਬਾਰੇ ਵਿਚਾਰ ਕਰੋ। ਨਮੀ ਬਰਕਰਾਰ ਰੱਖਣ ਦੀਆਂ ਤਕਨੀਕਾਂ ਦੀ ਵਰਤੋਂ ਕਰੋ।",
      "tips_hum_ok": "<strong>💧 ਚੰਗੀ ਨਮੀ ਦਾ ਪੱਧਰ:</strong> ਸਿਹਤਮੰਦ ਪੌਦੇ ਦੇ ਵਾਧੇ ਲਈ ਅਨੁਕੂਲ ਹਾਲਾਤ। ਅਨੁਕੂਲ ਪੌਦੇ ਦੇ ਵਾਧੇ ਲਈ ਨਿਗਰਾਨੀ ਕਰੋ।",
      "tips_soil_header": "🧪 ਮਿੱਟੀ ਪ੍ਰਬੰਧਨ",
      "tips_ph_acidic": "<strong>⚗️ ਤੇਜ਼ਾਬੀ ਮਿੱਟੀ:</strong> ਪੀਐੱਚ ਵਧਾਉਣ ਲਈ ਚੂਨਾ ਜੋੜਨ ਬਾਰੇ ਵਿਚਾਰ ਕਰੋ। ਐਲੂਮੀਨੀਅਮ ਜ਼ਹਿਰੀਲੇਪਨ ਲਈ ਜਾਂਚ ਕਰੋ ਅਤੇ ਮਿੱਟੀ ਦੀ ਬਣਤਰ ਨੂੰ ਬਿਹਤਰ ਬਣਾਉਣ ਲਈ ਜੈਵਿਕ ਪਦਾਰਥ ਸ਼ਾਮਲ ਕਰੋ।",
      "tips_ph_alkaline": "<strong>⚗️ ਖਾਰੀ ਮਿੱਟੀ:</strong> ਪੀਐੱਚ ਘਟਾਉਣ ਲਈ ਸਲਫਰ ਜਾਂ ਜੈਵਿਕ ਪਦਾਰਥ ਜੋੜਨ ਬਾਰੇ ਵਿਚਾਰ ਕਰੋ। ਸੂਖਮ ਪੌਸ਼ਟਿਕ ਤੱਤਾਂ ਦੀ ਘਾਟ ਲਈ ਨਿਗਰਾਨੀ ਕਰੋ।",
      "tips_ph_ok": "<strong>⚗️ ਅਨੁਕੂਲ ਪੀਐੱਚ ਰੇਂਜ:</strong> ਪੌਸ਼ਟਿਕ ਤੱਤਾਂ ਦੀ ਉਪਲਬਧਤਾ ਲਈ ਸੰਪੂਰਨ ਸਥਿਤੀਆਂ। ਨਿਯਮਤ ਜੈਵਿਕ ਸੋਧਾਂ ਨਾਲ ਮਿੱਟੀ ਦੀ ਸਿਹਤ ਨੂੰ ਬਣਾਈ ਰੱਖੋ।",
      "tips_n_low": "<strong>🔵 ਘੱਟ ਨਾਈਟ੍ਰੋਜਨ:</strong> ਯੂਰੀਆ ਜਾਂ ਜੈਵਿਕ ਖਾਦ ਵਰਗੀਆਂ ਨਾਈਟ੍ਰੋਜਨ-ਭਰਪੂਰ ਖਾਦਾਂ ਬਾਰੇ ਵਿਚਾਰ ਕਰੋ। ਬਿਹਤਰ ਸਮਾਈ ਲਈ ਵੰਡੀਆਂ ਖੁਰਾਕਾਂ ਵਿੱਚ ਲਾਗੂ ਕਰੋ।",
      "tips_n_high": "<strong>🔵 ਉੱਚ ਨਾਈਟ੍ਰੋਜਨ:</strong> ਬਹੁਤ ਜ਼ਿਆਦਾ ਸਬਜ਼ੀਆਂ ਦੇ ਵਾਧੇ ਦਾ ਕਾਰਨ ਬਣ ਸਕਦਾ ਹੈ। ਧਿਆਨ ਨਾਲ ਨਿਗਰਾਨੀ ਕਰੋ ਅਤੇ ਜੇ ਲੋੜ ਹੋਵੇ ਤਾਂ ਨਾਈਟ੍ਰੋਜਨ ਇਨਪੁਟ ਨੂੰ ਘਟਾਓ।",
      "tips_p_low": "<strong>🟡 ਘੱਟ ਫਾਸਫੋਰਸ:</strong> DAP ਜਾਂ ਰੌਕ ਫਾਸਫੇਟ ਲਾਗੂ ਕਰਨ ਬਾਰੇ ਵਿਚਾਰ ਕਰੋ। ਜੜ੍ਹਾਂ ਦੇ ਵਿਕਾਸ ਅਤੇ ਫੁੱਲ ਆਉਣ ਲਈ ਜ਼ਰੂਰੀ।",
      "tips_k_low": "<strong>🔴 ਘੱਟ ਪੋਟਾਸ਼ੀਅਮ:</strong> MOP (ਪੋਟਾਸ਼ ਦਾ ਮਿਉਰੇਟ) ਲਾਗੂ ਕਰਨ ਬਾਰੇ ਵਿਚਾਰ ਕਰੋ। ਰੋਗ ਪ੍ਰਤੀਰੋਧ ਅਤੇ ਪਾਣੀ ਦੇ ਨਿਯਮ ਲਈ ਮਹੱਤਵਪੂਰਨ।",
      "summary_box_header": "🌟 ਤੁਹਾਡੀ ਵਿਅਕਤੀਗਤ ਫਸਲ ਸਿਫਾਰਸ਼ ਦਾ ਸੰਖੇਪ",
      "summary_box_text": "ਤੁਹਾਡੀ ਮਿੱਟੀ ਅਤੇ ਵਾਤਾਵਰਣ ਦੀਆਂ ਸਥਿਤੀਆਂ ਦੇ ਸਾਡੇ AI ਵਿਸ਼ਲੇਸ਼ਣ ਦੇ ਆਧਾਰ ਤੇ, **{}** ਤੁਹਾਡੀ ਜ਼ਮੀਨ ਲਈ **{:.1f}% ਵਿਸ਼ਵਾਸ ਸਕੋਰ** ਨਾਲ ਸਭ ਤੋਂ ਢੁਕਵੀਂ ਫਸਲ ਹੈ।",
      "summary_match_quality": "🎯 ਮੈਚ ਗੁਣਵੱਤਾ",
      "summary_growth_potential": "🌱 ਵਿਕਾਸ ਦੀ ਸੰਭਾਵਨਾ",
      "summary_econ_viability": "💰 ਆਰਥਿਕ ਵਿਵਹਾਰਕਤਾ",
      "growth_high": "ਉੱਚ",
      "growth_medium": "ਮੱਧਮ",
      "growth_moderate": "ਮੱਧਮ",
      "econ_prof": "ਲਾਭਦਾਇਕ",
      "econ_good": "ਚੰਗਾ"
    },
    "fertilizer_recommendation": {
      "main_title": "🧪 ਖਾਦ ਸਿਫਾਰਸ਼ ਪ੍ਰਣਾਲੀ",
      "subtitle": "ਆਪਣੀ ਫਸਲ ਅਤੇ ਮਿੱਟੀ ਦੀਆਂ ਸਥਿਤੀਆਂ ਦੇ ਆਧਾਰ ਤੇ ਅਨੁਕੂਲ ਖਾਦ ਦੇ ਸੁਝਾਅ ਪ੍ਰਾਪਤ ਕਰੋ।",
      "section_info": "🌱 ਫਸਲ ਅਤੇ ਮਿੱਟੀ ਦੀ ਜਾਣਕਾਰੀ",
      "section_env": "🌡️ ਵਾਤਾਵਰਣ ਦੀਆਂ ਸਥਿਤੀਆਂ",
      "section_nutrients": "🧪 ਮੌਜੂਦਾ ਮਿੱਟੀ ਦੇ ਪੌਸ਼ਟਿਕ ਤੱਤ",
      "crop_type_label": "ਫਸਲ ਦੀ ਕਿਸਮ",
      "soil_type_label": "ਮਿੱਟੀ ਦੀ ਕਿਸਮ",
      "temp_label": "ਤਾਪਮਾਨ (°C)",
      "hum_label": "ਨਮੀ (%)",
      "moisture_label": "ਮਿੱਟੀ ਦੀ ਨਮੀ (%)",
      "nitrogen_label": "ਨਾਈਟ੍ਰੋਜਨ ਸਮੱਗਰੀ",
      "phosphorus_label": "ਫਾਸਫੋਰਸ ਸਮੱਗਰੀ",
      "potassium_label": "ਪੋਟਾਸ਼ੀਅਮ ਸਮੱਗਰੀ",
      "nutrient_status_header": "📊 ਪੌਸ਼ਟਿਕ ਤੱਤਾਂ ਦੀ ਸਥਿਤੀ",
      "low": "🔴 ਘੱਟ",
      "medium": "🟡 ਮੱਧਮ",
      "high": "🟢 ਉੱਚ",
      "predict_button": "💡 ਖਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਪ੍ਰਾਪਤ ਕਰੋ",
      "result_header": "🎯 ਸਿਫਾਰਸ਼ ਕੀਤੀ ਖਾਦ:",
      "result_confidence": "📊 ਵਿਸ਼ਵਾਸ:",
      "result_info_pre": "",
      "result_info_in": " ਮਿੱਟੀ ਵਿੱਚ ",
      "result_info_apply": "- **{}** ਖਾਦ ਲਾਗੂ ਕਰੋ",
      "result_info_tips": "- ਮਾਤਰਾ ਨਿਰਧਾਰਤ ਕਰਦੇ ਸਮੇਂ ਮੌਜੂਦਾ ਪੌਸ਼ਟਿਕ ਤੱਤਾਂ ਦੇ ਪੱਧਰਾਂ 'ਤੇ ਵਿਚਾਰ ਕਰੋ\n- ਉਚਿਤ ਵਿਕਾਸ ਦੇ ਪੜਾਅ ਦੌਰਾਨ ਲਾਗੂ ਕਰੋ\n- ਮਿੱਟੀ ਦੀ ਨਮੀ ਅਤੇ ਮੌਸਮ ਦੀਆਂ ਸਥਿਤੀਆਂ ਦੀ ਨਿਗਰਾਨੀ ਕਰੋ",
      "error_message": "ਭਵਿੱਖਬਾਣੀ ਵਿੱਚ ਗਲਤੀ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੇ ਇਨਪੁਟਸ ਦੀ ਜਾਂਚ ਕਰੋ।"
    },
    "disease_detection": {
      "main_title": "🔬 ਰੋਗ ਦਾ ਪਤਾ ਲਗਾਉਣਾ",
      "subtitle": "ਡੀਪ ਲਰਨਿੰਗ CNN ਮਾਡਲਾਂ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਤੁਰੰਤ ਰੋਗ ਦੀ ਪਛਾਣ ਲਈ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ।",
      "upload_header": "📷 ਪੌਦੇ ਦੇ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ",
      "upload_guidelines_title": "📸 ਤਸਵੀਰ ਅਪਲੋਡ ਦਿਸ਼ਾ-ਨਿਰਦੇਸ਼:",
      "upload_guidelines_text": "✓ ਸਾਫ਼, ਚੰਗੀ ਤਰ੍ਹਾਂ ਰੋਸ਼ਨੀ ਵਾਲੀਆਂ ਪੱਤੇ ਦੀਆਂ ਤਸਵੀਰਾਂ<br>✓ ਪ੍ਰਭਾਵਿਤ ਖੇਤਰਾਂ ਜਾਂ ਲੱਛਣਾਂ 'ਤੇ ਧਿਆਨ ਕੇਂਦਰਿਤ ਕਰੋ<br>✓ ਸਮਰਥਿਤ ਫਾਰਮੈਟ: JPG, PNG, JPEG<br>✓ ਅਧਿਕਤਮ ਆਕਾਰ: 10MB",
      "file_uploader_label": "ਇੱਕ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਚੁਣੋ...",
      "file_uploader_help": "ਪੌਦੇ ਦੇ ਪੱਤੇ ਦੀ ਇੱਕ ਸਾਫ਼ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ",
      "uploaded_image_caption": "ਅਪਲੋਡ ਕੀਤੀ ਪੱਤੇ ਦੀ ਤਸਵੀਰ",
      "analyze_button": "🔍 ਰੋਗਾਂ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰੋ",
      "loading_message": "🧠 AI ਤਸਵੀਰ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰ ਰਿਹਾ ਹੈ...",
      "analysis_complete": "ਵਿਸ਼ਲੇਸ਼ਣ ਪੂਰਾ ਹੋ ਗਿਆ!",
      "result_header": "🎯 ਭਵਿੱਖਬਾਣੀ ਕੀਤਾ ਰੋਗ:",
      "result_confidence": "📊 ਵਿਸ਼ਵਾਸ:",
      "disease_warning": "❗ ਤੁਹਾਡਾ ਪੌਦਾ ਰੋਗੀ ਹੋ ਸਕਦਾ ਹੈ। ਪੁਸ਼ਟੀ ਲਈ ਕਿਸੇ ਪੇਸ਼ੇਵਰ ਨਾਲ ਸਲਾਹ ਕਰੋ।",
      "healthy_message": "✅ ਪੌਦਾ ਸਿਹਤਮੰਦ ਲੱਗਦਾ ਹੈ!"
    },
    "about_page": {
      "main_title": "👥 ਸਾਡੇ ਬਾਰੇ",
      "subtitle": "ਸਮਾਰਟ ਖੇਤੀਬਾੜੀ ਕ੍ਰਾਂਤੀ ਦੇ ਪਿੱਛੇ ਦੀ ਨਵੀਨਤਾਕਾਰੀ ਟੀਮ ਨੂੰ ਮਿਲੋ!",
      "mission_title": "🌟 ਸਾਡਾ ਮਿਸ਼ਨ",
      "mission_text": "ਦੀਪਐਗਰੋ ਅਤਿ-ਆਧੁਨਿਕ AI ਅਤੇ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਤਕਨਾਲੋਜੀਆਂ ਰਾਹੀਂ ਰਵਾਇਤੀ ਖੇਤੀਬਾੜੀ ਨੂੰ ਬਦਲਣ ਲਈ ਸਮਰਪਿਤ ਹੈ। ਸਾਡਾ ਟੀਚਾ ਬਿਹਤਰ ਫਸਲ ਦੀ ਚੋਣ, ਅਨੁਕੂਲ ਖਾਦ ਦੀ ਵਰਤੋਂ, ਅਤੇ ਸ਼ੁਰੂਆਤੀ ਰੋਗ ਦਾ ਪਤਾ ਲਗਾਉਣ ਲਈ ਬੁੱਧੀਮਾਨ ਅੰਤਰ-ਦ੍ਰਿਸ਼ਟੀਆਂ ਨਾਲ ਕਿਸਾਨਾਂ ਨੂੰ ਸ਼ਕਤੀ ਪ੍ਰਦਾਨ ਕਰਨਾ ਹੈ।",
      "team_header": "👨‍💻 ਸਾਡੀ ਵਿਕਾਸ ਟੀਮ",
      "team_desc": "ਆਈ.ਆਈ.ਆਈ.ਟੀ. ਰਾਇਚੂਰ ਦੇ ਵਿਦਿਆਰਥੀਆਂ ਦਾ ਇੱਕ ਭਾਵੁਕ ਸਮੂਹ ਜੋ ਤਕਨਾਲੋਜੀ ਨਾਲ ਖੇਤੀਬਾੜੀ ਵਿੱਚ ਕ੍ਰਾਂਤੀ ਲਿਆਉਣ ਲਈ ਮਿਲ ਕੇ ਕੰਮ ਕਰ ਰਿਹਾ ਹੈ।",
      "tech_stack_header": "🛠️ ਤਕਨਾਲੋਜੀ ਸਟੈਕ",
      "ml_title": "🤖 ਮਸ਼ੀਨ ਲਰਨਿੰਗ",
      "ml_text": "• ਰੈਂਡਮ ਫਾਰੈਸਟ ਕਲਾਸੀਫਾਇਰ<br>• ਸਕਿਟ-ਲਰਨ<br>• ਨੰਪਾਈ ਅਤੇ ਪਾਂਡਾ<br>• ਵਿਸ਼ੇਸ਼ਤਾ ਇੰਜੀਨੀਅਰਿੰਗ",
      "web_title": "🌐 ਵੈੱਬ ਫਰੇਮਵਰਕ",
      "web_text": "• ਸਟ੍ਰੀਮਲਿਟ<br>• ਪਾਈਥਨ ਬੈਕਐਂਡ<br>• ਇੰਟਰਐਕਟਿਵ UI/UX<br>• ਰੀਅਲ-ਟਾਈਮ ਪ੍ਰੋਸੈਸਿੰਗ",
      "data_title": "📊 ਡੇਟਾ ਅਤੇ ਵਿਜ਼ੂਅਲਾਈਜੇਸ਼ਨ",
      "data_text": "• ਚਾਰਟਾਂ ਲਈ ਪਲੋਟਲੀ<br>• ਚਿੱਤਰ ਪ੍ਰੋਸੈਸਿੰਗ ਲਈ ਪੀ.ਆਈ.ਐਲ.<br>• ਕਸਟਮ ਸੀ.ਐਸ.ਐਸ. ਸਟਾਈਲਿੰਗ<br>• ਰੈਸਪੌਂਸਿਵ ਡਿਜ਼ਾਈਨ",
      "features_header": "✨ ਮੁੱਖ ਵਿਸ਼ੇਸ਼ਤਾਵਾਂ",
      "smart_pred_header": "🎯 ਸਮਾਰਟ ਭਵਿੱਖਬਾਣੀਆਂ",
      "smart_pred_list": "- **ਫਸਲ ਦੀ ਸਿਫਾਰਸ਼:** ਮਿੱਟੀ ਅਤੇ ਜਲਵਾਯੂ ਦੀਆਂ ਸਥਿਤੀਆਂ ਦੇ ਆਧਾਰ 'ਤੇ AI-ਸੰਚਾਲਿਤ ਫਸਲ ਦੀ ਚੋਣ\n- **ਖਾਦ ਦਾ ਅਨੁਕੂਲਨ:** ਵੱਧ ਤੋਂ ਵੱਧ ਉਪਜ ਲਈ ਬੁੱਧੀਮਾਨ ਖਾਦ ਦੀਆਂ ਸਿਫਾਰਸ਼ਾਂ\n- **ਰੋਗ ਦਾ ਪਤਾ ਲਗਾਉਣਾ:** ਪੌਦੇ ਦੇ ਰੋਗ ਦੀ ਪਛਾਣ ਲਈ ਕੰਪਿਊਟਰ ਵਿਜ਼ਨ",
      "ux_header": "🔧 ਉਪਭੋਗਤਾ ਅਨੁਭਵ",
      "ux_list": "- **ਇੰਟਰਐਕਟਿਵ ਇੰਟਰਫੇਸ:** ਵਰਤਣ ਵਿੱਚ ਆਸਾਨ ਸਲਾਈਡਰ ਅਤੇ ਇਨਪੁਟ ਫੀਲਡ\n- **ਰੀਅਲ-ਟਾਈਮ ਵਿਸ਼ਲੇਸ਼ਣ:** ਤੁਰੰਤ ਭਵਿੱਖਬਾਣੀਆਂ ਅਤੇ ਸਿਫਾਰਸ਼ਾਂ\n- **ਵਿਦਿਅਕ ਸਮੱਗਰੀ:** ਵਿਸਤ੍ਰਿਤ ਵਿਆਖਿਆਵਾਂ ਅਤੇ ਖੇਤੀ ਦੇ ਸੁਝਾਅ',",
      "institution_title": "🏫 ਸੰਸਥਾ",
      "institution_text": "<strong>ਇੰਡੀਅਨ ਇੰਸਟੀਚਿਊਟ ਆਫ ਇਨਫਰਮੇਸ਼ਨ ਟੈਕਨਾਲੋਜੀ, ਰਾਇਚੂਰ</strong><br>ਖੇਤੀਬਾੜੀ ਤਕਨਾਲੋਜੀ ਅਤੇ ਟਿਕਾਊ ਖੇਤੀਬਾੜੀ ਹੱਲਾਂ ਵਿੱਚ ਨਵੀਨਤਾ।",
      "acknowledgements_title": "🙏 ਧੰਨਵਾਦ",
      "acknowledgements_text": "ਇਸ ਖੇਤੀਬਾੜੀ AI ਹੱਲ ਨੂੰ ਵਿਕਸਤ ਕਰਨ ਵਿੱਚ ਉਨ੍ਹਾਂ ਦੇ ਸਮਰਥਨ ਅਤੇ ਮਾਰਗਦਰਸ਼ਨ ਲਈ ਸਾਡੇ ਫੈਕਲਟੀ ਸਲਾਹਕਾਰਾਂ ਡਾ. ਪ੍ਰਿਓਦਯੁਤੀ ਪ੍ਰਧਾਨ ਅਤੇ ਆਈ.ਆਈ.ਆਈ.ਟੀ. ਰਾਇਚੂਰ ਕਮਿਊਨਿਟੀ ਦਾ ਵਿਸ਼ੇਸ਼ ਧੰਨਵਾਦ।",
      "footer_title": "🌱 **ਦੀਪਐਗਰੋ**",
      "footer_slogan": "AI ਅਤੇ ML ਨਾਲ ਖੇਤੀਬਾੜੀ ਨੂੰ ਸ਼ਕਤੀ ਪ੍ਰਦਾਨ ਕਰਨਾ",
      "footer_credit": "❤️ ਦੀਪਐਗਰੋ ਟੀਮ ਦੁਆਰਾ ਬਣਾਇਆ ਗਿਆ | ਆਈ.ਆਈ.ਆਈ.ਟੀ. ਰਾਇਚੂਰ | 2025"
    }
  }, 
   'or': {
    'page_title': "ଦୀପଆଗ୍ରୋ - ସ୍ମାର୍ଟ କୃଷି",
    'sidebar_title': "🌾ନାଭିଗେସନ",
    'nav_home': "🏠 ହୋମ",
    'nav_crop': "🌾 ଫସଲ ପୂର୍ବାନୁମାନ",
    'nav_fertilizer': "🧪 ସାର ସୁପାରିଶ",
    'nav_disease': "🔬 ରୋଗ ଚିହ୍ନଟ",
    "nav_chat": "🤖 ଦୀପଆଗ୍ରୋ AI ସହାୟକ",
    'nav_about': "👥 ଆମ ବିଷୟରେ",
    'home': {
        'header_logo': '🌱 ଦୀପଆଗ୍ରୋ',
        'header_tagline': 'AI ଏବଂ ML ସହିତ ସ୍ମାର୍ଟ କୃଷି ସମାଧାନ',
        'welcome_header': "🌟 କୃଷିର ଭବିଷ୍ୟତକୁ ସ୍ୱାଗତ!",
        'welcome_text': "ଦୀପଆଗ୍ରୋ କୃଷି ପଦ୍ଧତିରେ ବିପ୍ଳବ ଆଣିବା ପାଇଁ ଅତ୍ୟାଧୁନିକ **ମେସିନ ଲର୍ନିଂ** ଏବଂ **ଆର୍ଟିଫିସିଆଲ ଇଣ୍ଟେଲିଜେନ୍ସ**କୁ ଉପଯୋଗ କରେ। ଆମର ପ୍ଲାଟଫର୍ମ ଏଥିପାଇଁ ବୁଦ୍ଧିମାନ ଅନ୍ତର୍ଦୃଷ୍ଟି ପ୍ରଦାନ କରେ:",
        'card_crop_title': '🌾 ସ୍ମାର୍ଟ ଫସଲ ସୁପାରିଶ',
        'card_crop_desc': 'ଉନ୍ନତ ML ଆଲଗୋରିଦମ ବ୍ୟବହାର କରି ମାଟି ଅବସ୍ଥା, ଜଳବାୟୁ ଏବଂ ପୁଷ୍ଟିସାର ଉପରେ ଆଧାର କରି ବ୍ୟକ୍ତିଗତ ଫସଲ ପରାମର୍ଶ ପାଆନ୍ତୁ।',
        'card_fert_title': '🧪 ସାର ଅପ୍ଟିମାଇଜେସନ',
        'card_fert_desc': 'ପରିବେଶ ପ୍ରଭାବକୁ ହ୍ରାସ କରି ଉତ୍ପାଦନକୁ ସର୍ବାଧିକ କରିବା ପାଇଁ ସଠିକ୍ ସାର ସୁପାରିଶ ପାଆନ୍ତୁ।',
        'card_disease_title': '🔬 AI-ଚାଳିତ ରୋଗ ଚିହ୍ନଟ',
        'card_disease_desc': 'ଅତ୍ୟାଧୁନିକ CNN ଗଭୀର ଶିକ୍ଷଣ ମଡେଲ ବ୍ୟବହାର କରି ତୁରନ୍ତ ରୋଗ ଚିହ୍ନଟ ପାଇଁ ପତ୍ରର ଛବି ଅପଲୋଡ କରନ୍ତୁ।',
        'metrics_header': '🚀 ମୁଖ୍ୟ ବୈଶିଷ୍ଟ୍ୟ',
        'metric_crops': 'ଫସଲ ପ୍ରକାର',
        'metric_fertilizers': 'ସାର ପ୍ରକାର',
        'metric_accuracy': 'ସଠିକତା',
        'metric_power': 'ଚାଳିତ',
        'why_choose_title': '🌟 ଦୀପଆଗ୍ରୋ କାହିଁକି ବାଛିବେ?',
        'why_choose_desc': 'ଆମର ଅତ୍ୟାଧୁନିକ AI ପ୍ରଯୁକ୍ତି ସହିତ କୃଷିର ଭବିଷ୍ୟତ ଅନୁଭବ କରନ୍ତୁ ଯାହା ସର୍ବାଧିକ ଉତ୍ପାଦନ ଏବଂ ସ୍ଥାୟୀତା ପାଇଁ ପାରମ୍ପରିକ କୃଷିକୁ ସ୍ମାର୍ଟ, ଡାଟା-ଚାଳିତ ନିଷ୍ପତ୍ତିରେ ପରିବର୍ତ୍ତନ କରେ।',
        'benefit_precision_title': 'ସଠିକ୍ କୃଷି',
        'benefit_precision_desc': 'ଉପଯୁକ୍ତ ଫସଲ ଚୟନ ଏବଂ ସମ୍ବଳ ପରିଚାଳନା ପାଇଁ ସଠିକ୍ ସଠିକତା ସହିତ ଡାଟା-ଚାଳିତ ନିଷ୍ପତ୍ତି ନିଅନ୍ତୁ।',
        'benefit_sustain_title': 'ସ୍ଥାୟୀ ଚାଷ',
        'benefit_sustain_desc': 'ବୁଦ୍ଧିମାନ ସୁପାରିଶ ମାଧ୍ୟମରେ ଉତ୍ପାଦକତାକୁ ସର୍ବାଧିକ କରି ବର୍ଜ୍ୟବସ୍ତୁ ଏବଂ ପରିବେଶ ପ୍ରଭାବକୁ ହ୍ରାସ କରନ୍ତୁ।',
        'benefit_realtime_title': 'ରିଅଲ-ଟାଇମ ବିଶ୍ଳେଷଣ',
        'benefit_realtime_desc': 'ଉନ୍ନତ ମେସିନ ଲର୍ନିଂ ଆଲଗୋରିଦମ ଏବଂ କମ୍ପ୍ୟୁଟର ଭିଜନ ଦ୍ୱାରା ଚାଳିତ ତୁରନ୍ତ ଅନ୍ତର୍ଦୃଷ୍ଟି ଏବଂ ପୂର୍ବାନୁମାନ ପାଆନ୍ତୁ।',
    },
    'crop_prediction': {
        'main_title': '🌾 ବୁଦ୍ଧିମାନ ଫସଲ ସୁପାରିଶ ପ୍ରଣାଳୀ',
        'subtitle': 'ଆପଣଙ୍କ ମାଟି ଏବଂ ପରିବେଶ ଅବସ୍ଥା ଉପରେ ଆଧାର କରି AI-ଚାଳିତ ଫସଲ ପରାମର୍ଶ ପାଆନ୍ତୁ।',
        'expander_header': 'ℹ️ ଫସଲ ପୂର୍ବାନୁମାନ ପାରାମିଟରକୁ ବୁଝିବା',
        'expander_info_text': 'ଆମର AI ମଡେଲ ଆପଣଙ୍କ ଜମି ପାଇଁ ସର୍ବୋତ୍ତମ ଫସଲ ସୁପାରିଶ କରିବା ପାଇଁ ଅନେକ କାରକକୁ ବିଶ୍ଳେଷଣ କରେ। ପ୍ରତ୍ୟେକ ପାରାମିଟର ଫସଲର ଉପଯୁକ୍ତତା ନିର୍ଣ୍ଣୟ କରିବାରେ ଏକ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ ଭୂମିକା ଗ୍ରହଣ କରେ:',
        'how_it_works': '📊 **ଏହା କିପରି କାର୍ଯ୍ୟ କରେ:** ଆମର ମେସିନ ଲର୍ନିଂ ଆଲଗୋରିଦମ ଆପଣଙ୍କର ଇନପୁଟ ଡାଟାକୁ ପ୍ରକ୍ରିୟାକରଣ କରେ ଏବଂ ବ୍ୟକ୍ତିଗତ ସୁପାରିଶ ପ୍ରଦାନ କରିବା ପାଇଁ ଏହାକୁ ହଜାର ହଜାର ସଫଳ ଫସଲ ସଂଯୋଗ ସହିତ ତୁଳନା କରେ।',
        'env_factors_header': '🌡️ ପରିବେଶ କାରକ',
        'temp_label': '🌡️ ତାପମାତ୍ରା (°C)',
        'temp_info': '<strong>ତାପମାତ୍ରା ପ୍ରଭାବ:</strong> ଡିଗ୍ରୀ ସେଲସିୟସରେ ପରିବେଶର ତାପମାତ୍ରା। ବିଭିନ୍ନ ଫସଲ ବିଭିନ୍ନ ତାପମାତ୍ରା ସୀମାରେ ବଢ଼ନ୍ତି - ଟ୍ରୋପିକାଲ ଫସଲ 25-35°C ପସନ୍ଦ କରନ୍ତି ଯେତେବେଳେକି ସାମାନ୍ୟ ଫସଲ 15-25°C ପସନ୍ଦ କରନ୍ତି।',
        'hum_label': '💧 ଆର୍ଦ୍ରତା (%)',
        'hum_info': '<strong>ଆର୍ଦ୍ରତା ପ୍ରଭାବ:</strong> ବାୟୁରେ ସାପେକ୍ଷ ଆର୍ଦ୍ରତା ପ୍ରତିଶତ। ଅଧିକ ଆର୍ଦ୍ରତା (>70%) ଧାନ ଭଳି ଫସଲ ପାଇଁ ଉପଯୁକ୍ତ, ଯେତେବେଳେକି କମ ଆର୍ଦ୍ରତା (<50%) ଗହମ ଏବଂ ବାର୍ଲି ଭଳି ଫସଲ ପାଇଁ ଭଲ।',
        'rain_label': '🌧️ ବର୍ଷା (ମି.ମି.)',
        'rain_info': '<strong>ବର୍ଷା ପ୍ରଭାବ:</strong> ମିଲିମିଟରରେ ହାରାହାରି ବର୍ଷା ପରିମାଣ। ଧାନକୁ 150-300 ମିମି ଆବଶ୍ୟକ, ଗହମକୁ 30-100 ମିମି ଆବଶ୍ୟକ, ଯେତେବେଳକି ମରୁଡି-ପ୍ରତିରୋଧୀ ଫସଲ <50 ମିମି ସହିତ ବଞ୍ଚିପାରନ୍ତି।',
        'ph_label': '⚗️ ମାଟିର pH ସ୍ତର',
        'ph_info': '<strong>pH ପ୍ରଭାବ:</strong> ମାଟିର pH ମୂଲ୍ୟ ଅମ୍ଳତା/କ୍ଷାରୀୟତାକୁ ମାପେ। ଅଧିକାଂଶ ଫସଲ 6.0-7.5 (ସ୍ୱଳ୍ପ ଅମ୍ଳୀୟରୁ ନିରପେକ୍ଷ) ପସନ୍ଦ କରନ୍ତି। ଅମ୍ଳୀୟ ମାଟି (<6) ବ୍ଲୁବେରୀ ପାଇଁ ଉପଯୁକ୍ତ, ଯେତେବେଳେକି କ୍ଷାରୀୟ ମାଟି (>7.5) ଆସ୍ପାରାଗସ୍ ପାଇଁ ଉପଯୁକ୍ତ।',
        'nutrients_header': '🧪 ମାଟି ପୁଷ୍ଟିସାର (NPK ମୂଲ୍ୟ)',
        'n_label': '🔵 ନାଇଟ୍ରୋଜେନ (N) ପରିମାଣ',
        'n_info': '<strong>ନାଇଟ୍ରୋଜେନ (N) ର ଭୂମିକା:</strong> ପତ୍ର ବୃଦ୍ଧି ଏବଂ କ୍ଲୋରୋଫିଲ ଉତ୍ପାଦନ ପାଇଁ ଅତ୍ୟାବଶ୍ୟକ। ପତ୍ରଯୁକ୍ତ ପନିପରିବାକୁ ଅଧିକ N (80-120) ଆବଶ୍ୟକ, ଯେତେବେଳେକି ମୂଳ ପନିପରିବାକୁ ମଧ୍ୟମ N (40-80) ଆବଶ୍ୟକ।',
        'p_label': '🟡 ଫସଫରସ (P) ପରିମାଣ',
        'p_info': '<strong>ଫସଫରସ (P) ର ଭୂମିକା:</strong> ମୂଳ ବୃଦ୍ଧି ଏବଂ ଫୁଲ ଫୁଟିବା ପାଇଁ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ। ଫଳ ଫସଲକୁ ଅଧିକ P (60-100) ଆବଶ୍ୟକ, ଯେତେବେଳେକି ଘାସକୁ କମ P (20-40) ଆବଶ୍ୟକ।',
        'k_label': '🔴 ପୋଟାସିୟମ (K) ପରିମାଣ',
        'k_info': '<strong>ପୋଟାସିୟମ (K) ର ଭୂମିକା:</strong> ରୋଗ ପ୍ରତିରୋଧ ଏବଂ ଜଳ ନିୟନ୍ତ୍ରଣ ପାଇଁ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ। ମୂଳ ପନିପରିବା ଏବଂ ଫଳକୁ ଅଧିକ K (80-150) ଆବଶ୍ୟକ, ଯେତେବେଳେକି ଶସ୍ୟକୁ ମଧ୍ୟମ K (40-80) ଆବଶ୍ୟକ।',
        'summary_header': '📊 ବର୍ତ୍ତମାନର ଇନପୁଟ ସାରାଂଶ',
        'summary_temp': '🌡️ **ତାପମାତ୍ରା:**',
        'summary_hum': '💧 **ଆର୍ଦ୍ରତା:**',
        'summary_rain': '🌧️ **ବର୍ଷା:**',
        'summary_ph': '⚗️ **pH ସ୍ତର:**',
        'summary_n': '🔵 **ନାଇଟ୍ରୋଜେନ (N):**',
        'summary_p': '🟡 **ଫସଫରସ (P):**',
        'summary_k': '🔴 **ପୋଟାସିୟମ (K):**',
        'reference_header': '📋 ଆଦର୍ଶ ସୀମା ସନ୍ଦର୍ଭ',
        'ref_text': '<strong>ଉପଯୁକ୍ତ ବଢ଼ିବା ପରିସ୍ଥିତି:</strong><br>• **ତାପମାତ୍ରା:** 20-30°C (ଅଧିକାଂଶ ଫସଲ)<br>• **ଆର୍ଦ୍ରତା:** 40-70% (ଉପଯୁକ୍ତ ସୀମା)<br>• **ବର୍ଷା:** 50-200mm (ଫସଲ ଅନୁଯାୟୀ ଭିନ୍ନ ହୁଏ)<br>• **pH:** 6.0-7.5 (ନିରପେକ୍ଷରୁ ସ୍ୱଳ୍ପ ଅମ୍ଳୀୟ)<br>• **NPK:** ସୁସ୍ଥ ବୃଦ୍ଧି ପାଇଁ ସନ୍ତୁଳିତ ଅନୁପାତ',
        'warning_temp': '🌡️ ତାପମାତ୍ରା ସାଧାରଣ ବୃଦ୍ଧି ସୀମା (5-45°C) ବାହାରେ ଅଛି',
        'warning_hum': '💧 ଆର୍ଦ୍ରତା ସ୍ତର ଅଧିକାଂଶ ଫସଲ ପାଇଁ ଚ୍ୟାଲେଞ୍ଜିଂ ହୋଇପାରେ',
        'warning_ph': '⚗️ pH ସ୍ତର ବହୁତ ଚରମ ଅଟେ ଏବଂ ଫସଲର ବିକଳ୍ପକୁ ସୀମିତ କରିପାରେ',
        'warning_n': '🔵 ଅତ୍ୟଧିକ ନାଇଟ୍ରୋଜେନ ସ୍ତର ଅତ୍ୟଧିକ ଉଦ୍ଭିଦ ବୃଦ୍ଧିର କାରଣ ହୋଇପାରେ',
        'warning_p': '🟡 ଅଧିକ ଫସଫରସ ସ୍ତର ଅନ୍ୟ ପୁଷ୍ଟିସାର ଅବଶୋଷଣରେ ହସ୍ତକ୍ଷେପ କରିପାରେ',
        'warning_k': '🔴 ଅତ୍ୟଧିକ ପୋଟାସିୟମ ସ୍ତର ମାଟିର ଗଠନକୁ ପ୍ରଭାବିତ କରିପାରେ',
        'warnings_header': '⚠️ ଇନପୁଟ ଚେତାବନୀ:',
        'validation_header': '✅ ବୈଧତା ସ୍ଥିତି',
        'validation_text': 'ସମସ୍ତ ଇନପୁଟ ମୂଲ୍ୟ ଗ୍ରହଣୀୟ ସୀମା ମଧ୍ୟରେ ଅଛି! ଆପଣଙ୍କର ପରିସ୍ଥିତି ଫସଲ ଚାଷ ପାଇଁ ବହୁତ ଭଲ।',
        'predict_button': '🔮 ସର୍ବୋତ୍ତମ ଫସଲର ପୂର୍ବାନୁମାନ କରନ୍ତୁ',
        'loading_1': 'ମାଟି ଅବସ୍ଥା ବିଶ୍ଳେଷଣ କରୁଛି...',
        'loading_2': 'ପରିବେଶ ଡାଟା ପ୍ରକ୍ରିୟାକରଣ କରୁଛି...',
        'loading_3': 'ଫସଲ ଡାଟାବେସ ସହିତ ମେଳ କରୁଛି...',
        'loading_4': 'ସୁପାରିଶକୁ ଚୂଡାନ୍ତ କରୁଛି...',
        'result_header': '🎯 ସୁପାରିଶ କରାଯାଇଥିବା ଫସଲ:',
        'result_confidence': '📊 ଆତ୍ମବିଶ୍ୱାସ ସ୍କୋର:',
        'result_quality': '🌟 ମ୍ୟାଚ ଗୁଣ:',
        'quality_excellent': 'ଉତ୍କୃଷ୍ଟ',
        'quality_good': 'ଭଲ',
        'quality_fair': 'ଠିକ',
        'top_3_header': '📈 ଶୀର୍ଷ 3 ଫସଲ ସୁପାରିଶ',
        'crop_season': 'ଋତୁ',
        'crop_water': 'ଜଳ ଆବଶ୍ୟକତା',
        'crop_match': 'ମେଳ',
        'crop_suitability': 'ଉପଯୁକ୍ତତା',
        'personalized_tips_header': '💡 ବ୍ୟକ୍ତିଗତ ଚାଷ ଟିପ୍ସ',
        'tips_climate_header': '🌡️ ଜଳବାୟୁ ବିଚାର',
        'tips_temp_high': '<strong>🌡️ ଅଧିକ ତାପମାତ୍ରା ଚେତାବନୀ:</strong> ଉତ୍ତାପ-ପ୍ରତିରୋଧୀ କିସମ, ଛାୟା ନେଟ ଏବଂ ବାରମ୍ବାର ଜଳସେଚନ ସମୟ ବିଷୟରେ ବିଚାର କରନ୍ତୁ। ଜଳ ଦକ୍ଷତା ପାଇଁ ଡ୍ରିପ ଜଳସେଚନ ସ୍ଥାପନ କରନ୍ତୁ।',
        'tips_temp_low': '<strong>❄️ ଥଣ୍ଡା ତାପମାତ୍ରା:</strong> ଥଣ୍ଡା-ପାଗ ଫସଲ ପାଇଁ ଆଦର୍ଶ। ଧାଡ଼ି କଭର ଏବଂ ଗ୍ରୀନହାଉସ ଚାଷ ଭଳି ତୁଷାର ସୁରକ୍ଷା ପଦକ୍ଷେପ ଉପରେ ବିଚାର କରନ୍ତୁ।',
        'tips_temp_ok': '<strong>🌡️ ଉପଯୁକ୍ତ ତାପମାତ୍ରା:</strong> ଅଧିକାଂଶ ଫସଲ କିସମ ପାଇଁ ସଠିକ୍ ଅବସ୍ଥା। ନିୟମିତ ଜଳସେଚନ ବଜାୟ ରଖନ୍ତୁ ଏବଂ କୀଟଗୁଡିକୁ ତଦାରଖ କରନ୍ତୁ।',
        'tips_hum_high': '<strong>💧 ଅଧିକ ଆର୍ଦ୍ରତା ଚେତାବନୀ:</strong> ଫଙ୍ଗଲ ରୋଗକୁ ରୋକିବା ପାଇଁ ଉପଯୁକ୍ତ ଗଛ ଦୂରତା ଏବଂ ଭେଣ୍ଟିଲେସନ ନିଶ୍ଚିତ କରନ୍ତୁ। ଫଙ୍ଗିସାଇଡ ଚିକିତ୍ସା ବିଷୟରେ ବିଚାର କରନ୍ତୁ।',
        'tips_hum_low': '<strong>🏜️ କମ ଆର୍ଦ୍ରତା ଚେତାବନୀ:</strong> ମାଟିର ଆର୍ଦ୍ରତା ବଜାୟ ରଖିବା ପାଇଁ ମଲଚିଂ ଏବଂ ବାରମ୍ବାର ହାଲୁକା ଜଳସେଚନ ବିଷୟରେ ବିଚାର କରନ୍ତୁ। ଆର୍ଦ୍ରତା ବଜାୟ ରଖିବା ପ୍ରଯୁକ୍ତି ବ୍ୟବହାର କରନ୍ତୁ।',
        'tips_hum_ok': '<strong>💧 ଭଲ ଆର୍ଦ୍ରତା ସ୍ତର:</strong> ସୁସ୍ଥ ଗଛ ବୃଦ୍ଧି ପାଇଁ ଅନୁକୂଳ ଅବସ୍ଥା। ଉପଯୁକ୍ତ ଗଛ ବୃଦ୍ଧି ପାଇଁ ତଦାରଖ କରନ୍ତୁ।',
        'tips_soil_header': '🧪 ମାଟି ପରିଚାଳନା',
        'tips_ph_acidic': '<strong>⚗️ ଅମ୍ଳୀୟ ମାଟି:</strong> pH ବଢ଼ାଇବା ପାଇଁ ଚୂନ ଯୋଗ କରିବା ବିଷୟରେ ବିଚାର କରନ୍ତୁ। ଆଲୁମିନିୟମ ବିଷାକ୍ତତା ପାଇଁ ପରୀକ୍ଷା କରନ୍ତୁ ଏବଂ ମାଟି ଗଠନରେ ଉନ୍ନତି ପାଇଁ ଜୈବିକ ପଦାର୍ଥ ଯୋଗ କରନ୍ତୁ।',
        'tips_ph_alkaline': '<strong>⚗️ କ୍ଷାରୀୟ ମାଟି:</strong> pH ହ୍ରାସ କରିବା ପାଇଁ ସଲଫର କିମ୍ବା ଜୈବିକ ପଦାର୍ଥ ଯୋଗ କରିବା ବିଷୟରେ ବିଚାର କରନ୍ତୁ। ମାଇକ୍ରୋନିଉଟ୍ରିଏଣ୍ଟ ଅଭାବ ପାଇଁ ତଦାରଖ କରନ୍ତୁ।',
        'tips_ph_ok': '<strong>⚗️ ଉପଯୁକ୍ତ pH ସୀମା:</strong> ପୁଷ୍ଟିସାର ଉପଲବ୍ଧତା ପାଇଁ ସଠିକ୍ ଅବସ୍ଥା। ନିୟମିତ ଜୈବିକ ସଂଶୋଧନ ସହିତ ମାଟିର ସ୍ୱାସ୍ଥ୍ୟ ବଜାୟ ରଖନ୍ତୁ।',
        'tips_n_low': '<strong>🔵 କମ ନାଇଟ୍ରୋଜେନ:</strong> ୟୁରିଆ କିମ୍ବା ଜୈବିକ ସାର ଭଳି ନାଇଟ୍ରୋଜେନ-ସମୃଦ୍ଧ ସାର ଉପରେ ବିଚାର କରନ୍ତୁ। ଉନ୍ନତ ଅବଶୋଷଣ ପାଇଁ ବିଭାଜିତ ମାତ୍ରାରେ ପ୍ରୟୋଗ କରନ୍ତୁ।',
        'tips_n_high': '<strong>🔵 ଅଧିକ ନାଇଟ୍ରୋଜେନ:</strong> ଅତ୍ୟଧିକ ଉଦ୍ଭିଦ ବୃଦ୍ଧିର କାରଣ ହୋଇପାରେ। ସାବଧାନତାର ସହିତ ତଦାରଖ କରନ୍ତୁ ଏବଂ ଆବଶ୍ୟକ ହେଲେ ନାଇଟ୍ରୋଜେନ ଇନପୁଟକୁ ହ୍ରାସ କରନ୍ତୁ।',
        'tips_p_low': '<strong>🟡 କମ ଫସଫରସ:</strong> DAP କିମ୍ବା ରକ ଫସଫେଟ ପ୍ରୟୋଗ କରିବା ବିଷୟରେ ବିଚାର କରନ୍ତୁ। ମୂଳ ବୃଦ୍ଧି ଏବଂ ଫୁଲ ଫୁଟିବା ପାଇଁ ଅତ୍ୟାବଶ୍ୟକ।',
        'tips_k_low': '<strong>🔴 କମ ପୋଟାସିୟମ:</strong> MOP (ପୋଟାଶର ମ୍ୟୁରେଟ) ପ୍ରୟୋଗ କରିବା ବିଷୟରେ ବିଚାର କରନ୍ତୁ। ରୋଗ ପ୍ରତିରୋଧ ଏବଂ ଜଳ ନିୟନ୍ତ୍ରଣ ପାଇଁ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ।',
        'summary_box_header': '🌟 ଆପଣଙ୍କର ବ୍ୟକ୍ତିଗତ ଫସଲ ସୁପାରିଶର ସାରାଂଶ',
        'summary_box_text': 'ଆପଣଙ୍କ ମାଟି ଏବଂ ପରିବେଶ ଅବସ୍ଥାର ଆମର AI ବିଶ୍ଳେଷଣ ଉପରେ ଆଧାର କରି, **{}** ଆପଣଙ୍କ ଜମି ପାଇଁ **{:.1f}% ଆତ୍ମବିଶ୍ୱାସ ସ୍କୋର** ସହିତ ସବୁଠାରୁ ଉପଯୁକ୍ତ ଫସଲ ଅଟେ।',
        'summary_match_quality': '🎯 ମ୍ୟାଚ ଗୁଣ',
        'summary_growth_potential': '🌱 ବୃଦ୍ଧି ସମ୍ଭାବନା',
        'summary_econ_viability': '💰 ଆର୍ଥିକ ବ୍ୟବହାର୍ଯ୍ୟତା',
        'growth_high': 'ଅଧିକ',
        'growth_medium': 'ମଧ୍ୟମ',
        'growth_moderate': 'ମଧ୍ୟମ',
        'econ_prof': 'ଲାଭଜନକ',
        'econ_good': 'ଭଲ',
    },
    'fertilizer_recommendation': {
        'main_title': '🧪 ସାର ସୁପାରିଶ ପ୍ରଣାଳୀ',
        'subtitle': 'ଆପଣଙ୍କ ଫସଲ ଏବଂ ମାଟି ଅବସ୍ଥା ଉପରେ ଆଧାର କରି ଉପଯୁକ୍ତ ସାର ପରାମର୍ଶ ପାଆନ୍ତୁ।',
        'section_info': '🌱 ଫସଲ ଏବଂ ମାଟି ସୂଚନା',
        'section_env': '🌡️ ପରିବେଶ ଅବସ୍ଥା',
        'section_nutrients': '🧪 ବର୍ତ୍ତମାନର ମାଟି ପୁଷ୍ଟିସାର',
        'crop_type_label': 'ଫସଲ ପ୍ରକାର',
        'soil_type_label': 'ମାଟି ପ୍ରକାର',
        'temp_label': 'ତାପମାତ୍ରା (°C)',
        'hum_label': 'ଆର୍ଦ୍ରତା (%)',
        'moisture_label': 'ମାଟି ଆର୍ଦ୍ରତା (%)',
        'nitrogen_label': 'ନାଇଟ୍ରୋଜେନ ପରିମାଣ',
        'phosphorus_label': 'ଫସଫରସ ପରିମାଣ',
        'potassium_label': 'ପୋଟାସିୟମ ପରିମାଣ',
        'nutrient_status_header': '📊 ପୁଷ୍ଟିସାର ସ୍ଥିତି',
        'low': '🔴 କମ',
        'medium': '🟡 ମଧ୍ୟମ',
        'high': '🟢 ଅଧିକ',
        'predict_button': '💡 ସାର ସୁପାରିଶ ପାଆନ୍ତୁ',
        'result_header': '🎯 ସୁପାରିଶ କରାଯାଇଥିବା ସାର:',
        'result_confidence': '📊 ଆତ୍ମବିଶ୍ୱାସ:',
        'result_info_pre': 'ପାଇଁ ',
        'result_info_in': ' ମାଟିରେ:',
        'result_info_apply': '- **{}** ସାର ପ୍ରୟୋଗ କରନ୍ତୁ',
        'result_info_tips': '- ପରିମାଣ ନିର୍ଣ୍ଣୟ କରିବାବେଳେ ବର୍ତ୍ତମାନର ପୁଷ୍ଟିସାର ସ୍ତରକୁ ବିଚାର କରନ୍ତୁ\n- ଉପଯୁକ୍ତ ବୃଦ୍ଧି ପର୍ଯ୍ୟାୟରେ ପ୍ରୟୋଗ କରନ୍ତୁ\n- ମାଟି ଆର୍ଦ୍ରତା ଏବଂ ପାଗ ଅବସ୍ଥାକୁ ତଦାରଖ କରନ୍ତୁ',
        'error_message': 'ପୂର୍ବାନୁମାନରେ ତ୍ରୁଟି। ଦୟାକରି ଆପଣଙ୍କ ଇନପୁଟ୍ସ ଯାଞ୍ଚ କରନ୍ତୁ।',
    },
    'disease_detection': {
        'main_title': '🔬 ରୋଗ ଚିହ୍ନଟ',
        'subtitle': 'ଗଭୀର ଶିକ୍ଷଣ CNN ମଡେଲ ବ୍ୟବହାର କରି ତୁରନ୍ତ ରୋଗ ଚିହ୍ନଟ ପାଇଁ ପତ୍ରର ଛବି ଅପଲୋଡ କରନ୍ତୁ।',
        'upload_header': '📷 ଗଛର ପତ୍ରର ଛବି ଅପଲୋଡ କରନ୍ତୁ',
        'upload_guidelines_title': '📸 ଛବି ଅପଲୋଡ ମାର୍ଗଦର୍ଶିକା:',
        'upload_guidelines_text': '✓ ସ୍ୱଚ୍ଛ, ଉଜ୍ଜ୍ୱଳ ପତ୍ରର ଛବି<br>✓ ପ୍ରଭାବିତ କ୍ଷେତ୍ର କିମ୍ବା ଲକ୍ଷଣ ଉପରେ ଧ୍ୟାନ ଦିଅନ୍ତୁ<br>✓ ସମର୍ଥିତ ଫର୍ମାଟ: JPG, PNG, JPEG<br>✓ ସର୍ବାଧିକ ଆକାର: 10MB',
        'file_uploader_label': 'ଏକ ପତ୍ରର ଛବି ବାଛନ୍ତୁ...',
        'file_uploader_help': 'ଗଛର ପତ୍ରର ଏକ ସ୍ୱଚ୍ଛ ଛବି ଅପଲୋଡ କରନ୍ତୁ',
        'uploaded_image_caption': 'ଅପଲୋଡ କରାଯାଇଥିବା ପତ୍ରର ଛବି',
        'analyze_button': '🔍 ରୋଗ ବିଶ୍ଳେଷଣ କରନ୍ତୁ',
        'loading_message': '🧠 AI ଛବିକୁ ବିଶ୍ଳେଷଣ କରୁଛି...',
        'analysis_complete': 'ବିଶ୍ଳେଷଣ ସମାପ୍ତ!',
        'result_header': '🎯 ପୂର୍ବାନୁମାନିତ ରୋଗ:',
        'result_confidence': '📊 ଆତ୍ମବିଶ୍ୱାସ:',
        'disease_warning': '❗ ଆପଣଙ୍କ ଗଛ ରୋଗଗ୍ରସ୍ତ ହୋଇପାରେ। ନିଶ୍ଚିତତା ପାଇଁ ଜଣେ ପେଶାଦାରଙ୍କ ସହିତ ପରାମର୍ଶ କରନ୍ତୁ।',
        'healthy_message': '✅ ଗଛ ସୁସ୍ଥ ଦେଖାଯାଉଛି!',
    },
    'about_page': {
        'main_title': '👥 ଆମ ବିଷୟରେ',
        'subtitle': 'ସ୍ମାର୍ଟ କୃଷି ବିପ୍ଳବ ପଛରେ ଥିବା ଅଭିନବ ଦଳକୁ ଭେଟନ୍ତୁ!',
        'mission_title': '🌟 ଆମର ଲକ୍ଷ୍ୟ',
        'mission_text': 'ଦୀପଆଗ୍ରୋ ଅତ୍ୟାଧୁନିକ AI ଏବଂ ମେସିନ ଲର୍ନିଂ ପ୍ରଯୁକ୍ତି ମାଧ୍ୟମରେ ପାରମ୍ପରିକ କୃଷିକୁ ରୂପାନ୍ତର କରିବାକୁ ସମର୍ପିତ। ଆମର ଲକ୍ଷ୍ୟ ହେଉଛି ଉତ୍ତମ ଫସଲ ଚୟନ, ଉପଯୁକ୍ତ ସାର ବ୍ୟବହାର, ଏବଂ ପ୍ରାରମ୍ଭିକ ରୋଗ ଚିହ୍ନଟ ପାଇଁ ବୁଦ୍ଧିମାନ ଅନ୍ତର୍ଦୃଷ୍ଟି ସହିତ ଚାଷୀମାନଙ୍କୁ ସଶକ୍ତ କରିବା।',
        'team_header': '👨‍💻 ଆମର ବିକାଶ ଦଳ',
        'team_desc': 'ଆଇଆଇଆଇଟି ରାୟଚୁରରୁ ଏକ ଉତ୍ସାହୀ ଛାତ୍ର ଦଳ ପ୍ରଯୁକ୍ତି ସହିତ କୃଷିରେ ବିପ୍ଳବ ଆଣିବା ପାଇଁ ଏକତ୍ର କାର୍ଯ୍ୟ କରୁଛନ୍ତି।',
        'tech_stack_header': '🛠️ ପ୍ରଯୁକ୍ତିଗତ ଷ୍ଟାକ',
        'ml_title': '🤖 ମେସିନ ଲର୍ନିଂ',
        'ml_text': '• ରାଣ୍ଡମ ଫରେଷ୍ଟ କ୍ଲାସିଫାୟର<br>• ସ୍କିକିଟ-ଲର୍ନ<br>• ନମ୍ପାଇ ଏବଂ ପାଣ୍ଡା<br>• ଫିଚର ଇଞ୍ଜିନିୟରିଂ',
        'web_title': '🌐 ୱେବ ଫ୍ରେମୱାର୍କ',
        'web_text': '• ଷ୍ଟ୍ରିମଲିଟ<br>• ପାଇଥନ ବ୍ୟାକେଣ୍ଡ<br>• ଇଣ୍ଟରାକ୍ଟିଭ UI/UX<br>• ରିଅଲ-ଟାଇମ ପ୍ରୋସେସିଂ',
        'data_title': '📊 ଡାଟା ଏବଂ ଭିଜୁଆଲାଇଜେସନ',
        'data_text': '• ଚାର୍ଟ ପାଇଁ ପ୍ଲଟଲି<br>• ଛବି ପ୍ରକ୍ରିୟାକରଣ ପାଇଁ PIL<br>• କଷ୍ଟମ CSS ଷ୍ଟାଇଲିଂ<br>• ରେସପନସିଭ ଡିଜାଇନ',
        'features_header': '✨ ମୁଖ୍ୟ ବୈଶିଷ୍ଟ୍ୟ',
        'smart_pred_header': '🎯 ସ୍ମାର୍ଟ ପୂର୍ବାନୁମାନ',
        'smart_pred_list': '- **ଫସଲ ସୁପାରିଶ:** ମାଟି ଏବଂ ଜଳବାୟୁ ଅବସ୍ଥା ଉପରେ ଆଧାରିତ AI-ଚାଳିତ ଫସଲ ଚୟନ\n- **ସାର ଅପ୍ଟିମାଇଜେସନ:** ସର୍ବାଧିକ ଉତ୍ପାଦନ ପାଇଁ ବୁଦ୍ଧିମାନ ସାର ସୁପାରିଶ\n- **ରୋଗ ଚିହ୍ନଟ:** ଗଛର ରୋଗ ଚିହ୍ନଟ ପାଇଁ କମ୍ପ୍ୟୁଟର ଭିଜନ',
        'ux_header': 'ଉପଭୋକ୍ତା ଅନୁଭବ',
        'ux_list': '- **ଇଣ୍ଟରାକ୍ଟିଭ ଇଣ୍ଟରଫେସ:** ବ୍ୟବହାର କରିବାକୁ ସହଜ ସ୍ଲାଇଡର ଏବଂ ଇନପୁଟ ଫିଲ୍ଡ\n- **ରିଅଲ-ଟାଇମ ବିଶ୍ଳେଷଣ:** ତୁରନ୍ତ ପୂର୍ବାନୁମାନ ଏବଂ ସୁପାରିଶ\n- **ଶିକ୍ଷାମୂଳକ ବିଷୟବସ୍ତୁ:** ବିସ୍ତୃତ ବ୍ୟାଖ୍ୟା ଏବଂ ଚାଷ ଟିପ୍ସ',
        'institution_title': '🏫 ଅନୁଷ୍ଠାନ',
        'institution_text': '<strong>ଇଣ୍ଡିଆନ ଇନଷ୍ଟିଚ୍ୟୁଟ ଅଫ ଇନଫରମେସନ ଟେକ୍ନୋଲୋଜି, ରାୟଚୁର</strong><br>କୃଷି ପ୍ରଯୁକ୍ତି ଏବଂ ସ୍ଥାୟୀ ଚାଷ ସମାଧାନରେ ଅଭିନବ।',
        'acknowledgements_title': 'ଧନ୍ୟବାଦ',
        'acknowledgements_text': 'ଏହି କୃଷି AI ସମାଧାନ ବିକାଶରେ ସେମାନଙ୍କ ସମର୍ଥନ ଏବଂ ମାର୍ଗଦର୍ଶନ ପାଇଁ ଆମର ଫ୍ୟାକଲ୍ଟି ପରାମର୍ଶଦାତା ଡକ୍ଟର ପ୍ରିୟୋଦ୍ୟୁତି ପ୍ରଧାନ ଏବଂ ଆଇଆଇଆଇଟି ରାୟଚୁର ସମ୍ପ୍ରଦାୟକୁ ବିଶେଷ ଧନ୍ୟବାଦ।',
        'footer_title': '🌱 **ଦୀପଆଗ୍ରୋ**',
        'footer_slogan': 'AI ଏବଂ ML ସହିତ କୃଷିକୁ ସଶକ୍ତ କରିବା',
        'footer_credit': '❤️ ଦୀପଆଗ୍ରୋ ଦଳ ଦ୍ୱାରା ନିର୍ମିତ | ଆଇଆଇଆଇଟି ରାୟଚୁର | 2025'
    }
}, 
'bn': {
    'page_title': "ডীপএগ্রো - স্মার্ট কৃষি",
    'sidebar_title': "🌾 নেভিগেশন",
    'nav_home': "🏠 হোম",
    'nav_crop': "🌾 ফসল পূর্বাভাস",
    'nav_fertilizer': "🧪 সারের সুপারিশ",
    'nav_disease': "🔬 রোগ শনাক্তকরণ",
    "nav_chat": "🤖 ডীপএগ্রো এআই সহকারী",
    'nav_about': "👥 আমাদের সম্পর্কে",
    'home': {
        'header_logo': '🌱 ডীপএগ্রো',
        'header_tagline': 'এআই এবং এমএল সহ স্মার্ট কৃষি সমাধান',
        'welcome_header': "🌟 কৃষির ভবিষ্যতে স্বাগতম!",
        'welcome_text': "ডীপএগ্রো অত্যাধুনিক **মেশিন লার্নিং** এবং **কৃত্রিম বুদ্ধিমত্তা** ব্যবহার করে কৃষিকাজে বিপ্লব ঘটাচ্ছে। আমাদের প্ল্যাটফর্ম এর জন্য বুদ্ধিদীপ্ত অন্তর্দৃষ্টি প্রদান করে:",
        'card_crop_title': '🌾 স্মার্ট ফসল সুপারিশ',
        'card_crop_desc': 'উন্নত এমএল অ্যালগরিদম ব্যবহার করে মাটির অবস্থা, জলবায়ু এবং পুষ্টির উপর ভিত্তি করে ব্যক্তিগতকৃত ফসলের পরামর্শ পান।',
        'card_fert_title': '🧪 সারের অপটিমাইজেশন',
        'card_fert_desc': 'পরিবেশের উপর প্রভাব কমানোর পাশাপাশি ফলন বাড়ানোর জন্য সঠিক সারের সুপারিশ পান।',
        'card_disease_title': '🔬 এআই-চালিত রোগ শনাক্তকরণ',
        'card_disease_desc': 'অত্যাধুনিক সিএনএন ডিপ লার্নিং মডেল ব্যবহার করে তাৎক্ষণিক রোগ শনাক্তকরণের জন্য পাতার ছবি আপলোড করুন।',
        'metrics_header': '🚀 প্রধান বৈশিষ্ট্য',
        'metric_crops': 'ফসল প্রকার',
        'metric_fertilizers': 'সারের প্রকার',
        'metric_accuracy': 'নির্ভুলতা',
        'metric_power': 'দ্বারা চালিত',
        'why_choose_title': '🌟 কেন ডীপএগ্রো বেছে নেবেন?',
        'why_choose_desc': 'আমাদের অত্যাধুনিক এআই প্রযুক্তির সাথে কৃষির ভবিষ্যতের অভিজ্ঞতা নিন যা সর্বোচ্চ ফলন এবং স্থায়িত্বের জন্য ঐতিহ্যবাহী কৃষিকাজকে স্মার্ট, ডেটা-চালিত সিদ্ধান্তে রূপান্তরিত করে।',
        'benefit_precision_title': 'সঠিক কৃষিকাজ',
        'benefit_precision_desc': 'সর্বোত্তম ফসল নির্বাচন এবং সম্পদ ব্যবস্থাপনার জন্য নিখুঁত নির্ভুলতার সাথে ডেটা-চালিত সিদ্ধান্ত নিন।',
        'benefit_sustain_title': 'টেকসই কৃষিকাজ',
        'benefit_sustain_desc': 'বুদ্ধিদীপ্ত সুপারিশের মাধ্যমে উৎপাদনশীলতা বাড়ানোর সময় অপচয় এবং পরিবেশগত প্রভাব হ্রাস করুন।',
        'benefit_realtime_title': 'রিয়েল-টাইম বিশ্লেষণ',
        'benefit_realtime_desc': 'উন্নত মেশিন লার্নিং অ্যালগরিদম এবং কম্পিউটার ভিশন দ্বারা চালিত তাৎক্ষণিক অন্তর্দৃষ্টি এবং পূর্বাভাস পান।',
    },
    'crop_prediction': {
        'main_title': '🌾 বুদ্ধিদীপ্ত ফসল সুপারিশ সিস্টেম',
        'subtitle': 'আপনার মাটি এবং পরিবেশগত অবস্থার উপর ভিত্তি করে এআই-চালিত ফসলের পরামর্শ পান।',
        'expander_header': 'ℹ️ ফসল পূর্বাভাস প্যারামিটার বোঝা',
        'expander_info_text': 'আমাদের এআই মডেল আপনার জমির জন্য সেরা ফসল সুপারিশ করতে একাধিক কারণ বিশ্লেষণ করে। প্রতিটি প্যারামিটার ফসলের উপযুক্ততা নির্ধারণে একটি গুরুত্বপূর্ণ ভূমিকা পালন করে:',
        'how_it_works': '📊 **এটি কীভাবে কাজ করে:** আমাদের মেশিন লার্নিং অ্যালগরিদম আপনার ইনপুট ডেটা প্রক্রিয়া করে এবং ব্যক্তিগতকৃত সুপারিশ প্রদানের জন্য হাজার হাজার সফল ফসলের সংমিশ্রণের সাথে এটি তুলনা করে।',
        'env_factors_header': '🌡️ পরিবেশগত কারণ',
        'temp_label': '🌡️ তাপমাত্রা (°C)',
        'temp_info': '<strong>তাপমাত্রার প্রভাব:</strong> ডিগ্রি সেলসিয়াসে পরিবেষ্টিত তাপমাত্রা। বিভিন্ন ফসল বিভিন্ন তাপমাত্রা পরিসরে বৃদ্ধি পায় - ক্রান্তীয় ফসল ২৫-৩৫°সে পছন্দ করে যখন নাতিশীতোষ্ণ ফসল ১৫-২৫°সে পছন্দ করে।',
        'hum_label': '💧 আর্দ্রতা (%)',
        'hum_info': '<strong>আর্দ্রতার প্রভাব:</strong> বাতাসে আপেক্ষিক আর্দ্রতার শতাংশ। উচ্চ আর্দ্রতা (>৭০%) ধান এর মতো ফসলের জন্য উপযুক্ত, যখন কম আর্দ্রতা (<৫০%) গম এবং বার্লি এর জন্য ভালো।',
        'rain_label': '🌧️ বৃষ্টিপাত (মিমি)',
        'rain_info': '<strong>বৃষ্টিপাতের প্রভাব:</strong> মিলিমিটারে গড় বৃষ্টিপাতের পরিমাণ। ধানের জন্য ১৫০-৩০০ মিমি প্রয়োজন, গমের জন্য ৩০-১০০ মিমি প্রয়োজন, যখন খরা-প্রতিরোধী ফসল <৫০ মিমি তেও টিকে থাকতে পারে।',
        'ph_label': '⚗️ মাটির পিএইচ স্তর',
        'ph_info': '<strong>পিএইচ-এর প্রভাব:</strong> মাটির পিএইচ মান অম্লতা/ক্ষারত্ব পরিমাপ করে। বেশিরভাগ ফসল ৬.০-৭.৫ (সামান্য অম্লীয় থেকে নিরপেক্ষ) পছন্দ করে। অম্লীয় মাটি (<৬) ব্লুবেরি এর জন্য উপযুক্ত, যখন ক্ষারীয় মাটি (>৭.৫) অ্যাস্পারাগাস এর জন্য উপযুক্ত।',
        'nutrients_header': '🧪 মাটির পুষ্টি (NPK মান)',
        'n_label': '🔵 নাইট্রোজেন (N) উপাদান',
        'n_info': '<strong>নাইট্রোজেন (N) এর ভূমিকা:</strong> পাতা বৃদ্ধি এবং ক্লোরোফিল উৎপাদনের জন্য অপরিহার্য। পাতাযুক্ত শাকসব্জির জন্য উচ্চ N (৮০-১২০) প্রয়োজন, যখন মূল শাকসব্জির জন্য মাঝারি N (৪০-৮০) প্রয়োজন।',
        'p_label': '🟡 ফসফরাস (P) উপাদান',
        'p_info': '<strong>ফসফরাস (P) এর ভূমিকা:</strong> মূলের বৃদ্ধি এবং ফুলের জন্য গুরুত্বপূর্ণ। ফল ফসলের জন্য উচ্চ P (৬০-১০০) প্রয়োজন, যখন ঘাসের জন্য কম P (২০-৪০) প্রয়োজন।',
        'k_label': '🔴 পটাশিয়াম (K) উপাদান',
        'k_info': '<strong>পটাশিয়াম (K) এর ভূমিকা:</strong> রোগ প্রতিরোধ এবং জল নিয়ন্ত্রণের জন্য গুরুত্বপূর্ণ। মূল শাকসব্জি এবং ফলের জন্য উচ্চ K (৮০-১৫০) প্রয়োজন, যখন শস্যের জন্য মাঝারি K (৪০-৮০) প্রয়োজন।',
        'summary_header': '📊 বর্তমান ইনপুট সারসংক্ষেপ',
        'summary_temp': '🌡️ **তাপমাত্রা:**',
        'summary_hum': '💧 **আর্দ্রতা:**',
        'summary_rain': '🌧️ **বৃষ্টিপাত:**',
        'summary_ph': '⚗️ **পিএইচ স্তর:**',
        'summary_n': '🔵 **নাইট্রোজেন (N):**',
        'summary_p': '🟡 **ফসফরাস (P):**',
        'summary_k': '🔴 **পটাশিয়াম (K):**',
        'reference_header': '📋 আদর্শ পরিসীমা নির্দেশিকা',
        'ref_text': '<strong>সর্বোত্তম বৃদ্ধির শর্তাবলী:</strong><br>• **তাপমাত্রা:** ২০-৩০°C (বেশিরভাগ ফসল)<br>• **আর্দ্রতা:** ৪০-৭০% (সর্বোত্তম পরিসর)<br>• **বৃষ্টিপাত:** ৫০-২০০ মিমি (ফসল অনুযায়ী পরিবর্তিত হয়)<br>• **পিএইচ:** ৬.০-৭.৫ (নিরপেক্ষ থেকে সামান্য অম্লীয়)<br>• **NPK:** সুস্থ বৃদ্ধির জন্য সুষম অনুপাত',
        'warning_temp': '🌡️ তাপমাত্রা সাধারণ বৃদ্ধির পরিসরের (৫-৪৫°সে) বাইরে আছে',
        'warning_hum': '💧 আর্দ্রতার মাত্রা বেশিরভাগ ফসলের জন্য চ্যালেঞ্জিং হতে পারে',
        'warning_ph': '⚗️ পিএইচ স্তর বেশ চরম এবং ফসলের বিকল্প সীমিত করতে পারে',
        'warning_n': '🔵 খুব উচ্চ নাইট্রোজেন স্তর অত্যধিক উদ্ভিজ্জ বৃদ্ধির কারণ হতে পারে',
        'warning_p': '🟡 উচ্চ ফসফরাস স্তর অন্যান্য পুষ্টির শোষণে হস্তক্ষেপ করতে পারে',
        'warning_k': '🔴 খুব উচ্চ পটাশিয়াম স্তর মাটির গঠনকে প্রভাবিত করতে পারে',
        'warnings_header': '⚠️ ইনপুট সতর্কতা:',
        'validation_header': '✅ বৈধতার স্থিতি',
        'validation_text': 'সমস্ত ইনপুট মান গ্রহণযোগ্য সীমার মধ্যে রয়েছে! আপনার অবস্থা ফসল চাষের জন্য খুব ভালো দেখাচ্ছে।',
        'predict_button': '🔮 সেরা ফসলের পূর্বাভাস দিন',
        'loading_1': 'মাটির অবস্থা বিশ্লেষণ করা হচ্ছে...',
        'loading_2': 'পরিবেশগত ডেটা প্রক্রিয়াকরণ করা হচ্ছে...',
        'loading_3': 'ফসলের ডেটাবেসের সাথে মিলানো হচ্ছে...',
        'loading_4': 'সুপারিশ চূড়ান্ত করা হচ্ছে...',
        'result_header': '🎯 প্রস্তাবিত ফসল:',
        'result_confidence': '📊 আত্মবিশ্বাসের স্কোর:',
        'result_quality': '🌟 মিলের গুণমান:',
        'quality_excellent': 'চমৎকার',
        'quality_good': 'ভালো',
        'quality_fair': 'মোটামুটি',
        'top_3_header': '📈 শীর্ষ ৩ ফসলের সুপারিশ',
        'crop_season': 'ঋতু',
        'crop_water': 'জলের প্রয়োজন',
        'crop_match': 'মিল',
        'crop_suitability': 'উপযোগিতা',
        'personalized_tips_header': '💡 ব্যক্তিগতকৃত কৃষি টিপস',
        'tips_climate_header': '🌡️ জলবায়ু বিবেচনা',
        'tips_temp_high': '<strong>🌡️ উচ্চ তাপমাত্রা সতর্কতা:</strong> তাপ-প্রতিরোধী জাত, শেড নেট এবং ঘন ঘন সেচ সূচি বিবেচনা করুন। জল সাশ্রয়ের জন্য ড্রিপ সেচ স্থাপন করুন।',
        'tips_temp_low': '<strong>❄️ ঠান্ডা তাপমাত্রা:</strong> শীতল-আবহাওয়ার ফসলের জন্য আদর্শ। সারি কভার এবং গ্রিনহাউস চাষের মতো তুষার প্রতিরোধের ব্যবস্থা বিবেচনা করুন।',
        'tips_temp_ok': '<strong>🌡️ সর্বোত্তম তাপমাত্রা:</strong> বেশিরভাগ ফসলের জাতের জন্য উপযুক্ত অবস্থা। নিয়মিত জল দেওয়া বজায় রাখুন এবং কীটপতঙ্গ পর্যবেক্ষণ করুন।',
        'tips_hum_high': '<strong>💧 উচ্চ আর্দ্রতা সতর্কতা:</strong> ছত্রাকজনিত রোগ প্রতিরোধের জন্য সঠিক উদ্ভিদ ব্যবধান এবং বায়ুচলাচল নিশ্চিত করুন। ছত্রাকনাশক চিকিৎসার কথা বিবেচনা করুন।',
        'tips_hum_low': '<strong>🏜️ কম আর্দ্রতা সতর্কতা:</strong> মাটির আর্দ্রতা বজায় রাখার জন্য মালচিং এবং ঘন ঘন হালকা জল দেওয়া বিবেচনা করুন। আর্দ্রতা ধরে রাখার কৌশল ব্যবহার করুন।',
        'tips_hum_ok': '<strong>💧 ভালো আর্দ্রতার স্তর:</strong> সুস্থ উদ্ভিদ বৃদ্ধির জন্য অনুকূল অবস্থা। সর্বোত্তম উদ্ভিদ বৃদ্ধির জন্য পর্যবেক্ষণ করুন।',
        'tips_soil_header': '🧪 মাটি ব্যবস্থাপনা',
        'tips_ph_acidic': '<strong>⚗️ অম্লীয় মাটি:</strong> পিএইচ বাড়ানোর জন্য চুন যোগ করার কথা ভাবুন। অ্যালুমিনিয়ামের বিষাক্ততার জন্য পরীক্ষা করুন এবং মাটির গঠন উন্নত করতে জৈব পদার্থ যোগ করুন।',
        'tips_ph_alkaline': '<strong>⚗️ ক্ষারীয় মাটি:</strong> পিএইচ কমাতে সালফার বা জৈব পদার্থ যোগ করার কথা ভাবুন। মাইক্রোনিউট্রিয়েন্ট এর ঘাটতি পর্যবেক্ষণ করুন।',
        'tips_ph_ok': '<strong>⚗️ সর্বোত্তম পিএইচ পরিসীমা:</strong> পুষ্টির প্রাপ্যতার জন্য নিখুঁত অবস্থা। নিয়মিত জৈব সংশোধনের সাথে মাটির স্বাস্থ্য বজায় রাখুন।',
        'tips_n_low': '<strong>🔵 কম নাইট্রোজেন:</strong> ইউরিয়া বা জৈব সারের মতো নাইট্রোজেন-সমৃদ্ধ সার বিবেচনা করুন। ভাল শোষণের জন্য বিভক্ত ডোজ-এ প্রয়োগ করুন।',
        'tips_n_high': '<strong>🔵 উচ্চ নাইট্রোজেন:</strong> অত্যধিক উদ্ভিজ্জ বৃদ্ধির কারণ হতে পারে। সতর্কতার সাথে পর্যবেক্ষণ করুন এবং প্রয়োজন হলে নাইট্রোজেন ইনপুট হ্রাস করুন।',
        'tips_p_low': '<strong>🟡 কম ফসফরাস:</strong> ডিএপি বা রক ফসফেট প্রয়োগ করার কথা ভাবুন। মূল বৃদ্ধি এবং ফুলের জন্য অপরিহার্য।',
        'tips_k_low': '<strong>🔴 কম পটাশিয়াম:</strong> এমওপি (মিউরিয়েট অফ পটাশ) প্রয়োগ করার কথা ভাবুন। রোগ প্রতিরোধ এবং জল নিয়ন্ত্রণের জন্য গুরুত্বপূর্ণ।',
        'summary_box_header': '🌟 আপনার ব্যক্তিগতকৃত ফসল সুপারিশের সারসংক্ষেপ',
        'summary_box_text': 'আপনার মাটি এবং পরিবেশগত অবস্থার আমাদের এআই বিশ্লেষণের উপর ভিত্তি করে, **{}** হলো আপনার জমির জন্য **{:.1f}% আত্মবিশ্বাসের স্কোর** সহ সবচেয়ে উপযুক্ত ফসল।',
        'summary_match_quality': '🎯 মিলের গুণমান',
        'summary_growth_potential': '🌱 বৃদ্ধির সম্ভাবনা',
        'summary_econ_viability': '💰 অর্থনৈতিক সম্ভাব্যতা',
        'growth_high': 'অধিক',
        'growth_medium': 'মাঝারি',
        'growth_moderate': 'মাঝারি',
        'econ_prof': 'লাভজনক',
        'econ_good': 'ভালো',
    },
    'fertilizer_recommendation': {
        'main_title': '🧪 সারের সুপারিশ সিস্টেম',
        'subtitle': 'আপনার ফসল এবং মাটির অবস্থার উপর ভিত্তি করে সর্বোত্তম সারের পরামর্শ পান।',
        'section_info': '🌱 ফসল এবং মাটির তথ্য',
        'section_env': '🌡️ পরিবেশগত অবস্থা',
        'section_nutrients': '🧪 বর্তমান মাটির পুষ্টি',
        'crop_type_label': 'ফসল প্রকার',
        'soil_type_label': 'মাটির প্রকার',
        'temp_label': 'তাপমাত্রা (°C)',
        'hum_label': 'আর্দ্রতা (%)',
        'moisture_label': 'মাটির আর্দ্রতা (%)',
        'nitrogen_label': 'নাইট্রোজেন উপাদান',
        'phosphorus_label': 'ফসফরাস উপাদান',
        'potassium_label': 'পটাশিয়াম উপাদান',
        'nutrient_status_header': '📊 পুষ্টির স্থিতি',
        'low': '🔴 কম',
        'medium': '🟡 মাঝারি',
        'high': '🟢 উচ্চ',
        'predict_button': '💡 সারের সুপারিশ পান',
        'result_header': '🎯 প্রস্তাবিত সার:',
        'result_confidence': '📊 আত্মবিশ্বাস:',
        'result_info_pre': '',
        'result_info_in': ' মাটিতে:',
        'result_info_apply': '- **{}** সার প্রয়োগ করুন',
        'result_info_tips': '- পরিমাণ নির্ধারণ করার সময় বর্তমান পুষ্টির মাত্রা বিবেচনা করুন\n- উপযুক্ত বৃদ্ধির পর্যায়ে প্রয়োগ করুন\n- মাটির আর্দ্রতা এবং আবহাওয়ার অবস্থা পর্যবেক্ষণ করুন',
        'error_message': 'পূর্বাভাসে ত্রুটি। আপনার ইনপুটগুলি পরীক্ষা করুন।',
    },
    'disease_detection': {
        'main_title': '🔬 রোগ শনাক্তকরণ',
        'subtitle': 'ডিপ লার্নিং সিএনএন মডেল ব্যবহার করে একটি পাতার ছবি থেকে তাৎক্ষণিকভাবে রোগ শনাক্ত করুন।',
        'upload_header': '📷 গাছের পাতার ছবি আপলোড করুন',
        'upload_guidelines_title': '📸 ছবি আপলোড নির্দেশিকা:',
        'upload_guidelines_text': '✓ পরিষ্কার, ভালোভাবে আলোকিত পাতার ছবি<br>✓ আক্রান্ত স্থান বা লক্ষণের উপর মনোযোগ দিন<br>✓ সমর্থিত ফরম্যাট: JPG, PNG, JPEG<br>✓ সর্বোচ্চ আকার: 10MB',
        'file_uploader_label': 'একটি পাতার ছবি বেছে নিন...',
        'file_uploader_help': 'একটি গাছের পাতার পরিষ্কার ছবি আপলোড করুন',
        'uploaded_image_caption': 'আপলোড করা পাতার ছবি',
        'analyze_button': '🔍 রোগের জন্য বিশ্লেষণ করুন',
        'loading_message': '🧠 এআই ছবিটি বিশ্লেষণ করছে...',
        'analysis_complete': 'বিশ্লেষণ সম্পূর্ণ!',
        'result_header': '🎯 পূর্বাভাসিত রোগ:',
        'result_confidence': '📊 আত্মবিশ্বাস:',
        'disease_warning': '❗ আপনার গাছে রোগ থাকতে পারে। নিশ্চিতকরণের জন্য একজন পেশাদারের সাথে পরামর্শ করুন।',
        'healthy_message': '✅ গাছটি সুস্থ বলে মনে হচ্ছে!',
    },
    'about_page': {
        'main_title': '👥 আমাদের সম্পর্কে',
        'subtitle': 'স্মার্ট কৃষি বিপ্লবের পেছনের উদ্ভাবনী দলের সাথে দেখা করুন!',
        'mission_title': '🌟 আমাদের লক্ষ্য',
        'mission_text': 'ডীপএগ্রো অত্যাধুনিক এআই এবং মেশিন লার্নিং প্রযুক্তির মাধ্যমে ঐতিহ্যবাহী কৃষিকাজকে রূপান্তরিত করতে নিবেদিত। আমাদের লক্ষ্য হলো উন্নত ফসল নির্বাচন, সর্বোত্তম সার ব্যবহার, এবং প্রাথমিক রোগ শনাক্তকরণের জন্য বুদ্ধিদীপ্ত অন্তর্দৃষ্টি দিয়ে কৃষকদের ক্ষমতায়ন করা।',
        'team_header': '👨‍💻 আমাদের উন্নয়ন দল',
        'team_desc': 'আইআইআইটি রায়চুর এর একদল উত্সাহী ছাত্র প্রযুক্তির মাধ্যমে কৃষিতে বিপ্লব ঘটাতে একসাথে কাজ করছে।',
        'tech_stack_header': '🛠️ প্রযুক্তি স্ট্যাক',
        'ml_title': '🤖 মেশিন লার্নিং',
        'ml_text': '• র্যান্ডম ফরেস্ট ক্লাসিফায়ার<br>• সাইকিট-লার্ন<br>• নামপাই এবং প্যান্ডাস<br>• ফিচার ইঞ্জিনিয়ারিং',
        'web_title': '🌐 ওয়েব ফ্রেমওয়ার্ক',
        'web_text': '• স্ট্রিমলিট<br>• পাইথন ব্যাকএন্ড<br>• ইন্টারেক্টিভ ইউআই/ইউএক্স<br>• রিয়েল-টাইম প্রক্রিয়াকরণ',
        'data_title': '📊 ডেটা এবং ভিজ্যুয়ালাইজেশন',
        'data_text': '• চার্টের জন্য প্লটলি<br>• চিত্র প্রক্রিয়াকরণের জন্য পিআইএল<br>• কাস্টম সিএসএস স্টাইলিং<br>• রেসপনসিভ ডিজাইন',
        'features_header': '✨ প্রধান বৈশিষ্ট্য',
        'smart_pred_header': '🎯 স্মার্ট পূর্বাভাস',
        'smart_pred_list': '- **ফসল সুপারিশ:** মাটি এবং জলবায়ু অবস্থার উপর ভিত্তি করে এআই-চালিত ফসল নির্বাচন\n- **সারের অপটিমাইজেশন:** সর্বোচ্চ ফলনের জন্য বুদ্ধিদীপ্ত সারের সুপারিশ\n- **রোগ শনাক্তকরণ:** গাছের রোগ শনাক্তকরণের জন্য কম্পিউটার ভিশন',
        'ux_header': '🔧 ব্যবহারকারী অভিজ্ঞতা',
        'ux_list': '- **ইন্টারেক্টিভ ইন্টারফেস:** ব্যবহার করা সহজ স্লাইডার এবং ইনপুট ফিল্ড\n- **রিয়েল-টাইম বিশ্লেষণ:** তাৎক্ষণিক পূর্বাভাস এবং সুপারিশ\n- **শিক্ষামূলক সামগ্রী:** বিস্তারিত ব্যাখ্যা এবং কৃষি টিপস',
        'institution_title': '🏫 প্রতিষ্ঠান',
        'institution_text': '<strong>ইন্ডিয়ান ইনস্টিটিউট অফ ইনফরমেশন টেকনোলজি, রায়চুর</strong><br>কৃষি প্রযুক্তি এবং টেকসই কৃষি সমাধানে উদ্ভাবন।',
        'acknowledgements_title': '🙏 কৃতজ্ঞতা',
        'acknowledgements_text': 'এই কৃষি এআই সমাধানটি বিকাশে তাদের সমর্থন এবং নির্দেশনার জন্য আমাদের অনুষদ উপদেষ্টা ড. প্রিয়োদ্যুতি প্রধান এবং আইআইআইটি রায়চুর কমিউনিটিকে বিশেষ ধন্যবাদ।',
        'footer_title': '🌱 **ডীপএগ্রো**',
        'footer_slogan': 'এআই এবং এমএল দিয়ে কৃষিকে ক্ষমতায়ন',
        'footer_credit': '❤️ টিম ডীপএগ্রো দ্বারা নির্মিত | আইআইআইটি রায়চুর | ২০২৫'
    }
}, 
'mr': {
    'page_title': "डीपअॅग्रो - स्मार्ट शेती",
    'sidebar_title': "🌾नेव्हिगेशन",
    'nav_home': "🏠 मुख्यपृष्ठ",
    'nav_crop': "🌾 पिकाचा अंदाज",
    'nav_fertilizer': "🧪 खताची शिफारस",
    'nav_disease': "🔬 रोगाची ओळख",
    "nav_chat": "🤖 डीपअॅग्रो एआय सहाय्यक",
    'nav_about': "👥 आमच्याबद्दल",
    'home': {
        'header_logo': '🌱 डीपअॅग्रो',
        'header_tagline': 'एआय आणि एमएल सह स्मार्ट शेती उपाय',
        'welcome_header': "🌟 शेतीच्या भविष्यात आपले स्वागत आहे!",
        'welcome_text': "डीपअॅग्रो शेतीच्या पद्धतींमध्ये क्रांती घडवण्यासाठी अत्याधुनिक **यंत्र शिक्षण (मशीन लर्निंग)** आणि **कृत्रिम बुद्धिमत्ता (आर्टिफिशियल इंटेलिजन्स)** चा वापर करते. आमचे प्लॅटफॉर्म यासाठी उपयुक्त अंतर्दृष्टी प्रदान करते:",
        'card_crop_title': '🌾 स्मार्ट पीक शिफारस',
        'card_crop_desc': 'अत्याधुनिक एमएल अल्गॉरिदम वापरून मातीची स्थिती, हवामान आणि पोषक तत्वांवर आधारित वैयक्तिकृत पिकाचे अंदाज मिळवा.',
        'card_fert_title': '🧪 खत अनुकूलन',
        'card_fert_desc': 'पर्यावरणाचा परिणाम कमी करताना उत्पादन वाढवण्यासाठी अचूक खतांच्या शिफारशी मिळवा.',
        'card_disease_title': '🔬 एआय-समर्थित रोग ओळख',
        'card_disease_desc': 'अत्याधुनिक सीएनएन डीप लर्निंग मॉडेल्स वापरून त्वरित रोग ओळखण्यासाठी पानांचे फोटो अपलोड करा.',
        'metrics_header': '🚀 प्रमुख वैशिष्ट्ये',
        'metric_crops': 'पिकांचे प्रकार',
        'metric_fertilizers': 'खतांचे प्रकार',
        'metric_accuracy': 'अचूकता',
        'metric_power': 'समर्थित',
        'why_choose_title': '🌟 डीपअॅग्रो का निवडायचे?',
        'why_choose_desc': 'आमच्या अत्याधुनिक एआय तंत्रज्ञानासह शेतीच्या भविष्याचा अनुभव घ्या, जे जास्तीत जास्त उत्पादन आणि टिकाऊपणासाठी पारंपारिक शेतीला स्मार्ट, डेटा-चालित निर्णयांमध्ये बदलते.',
        'benefit_precision_title': 'अचूक शेती',
        'benefit_precision_desc': 'योग्य पीक निवड आणि संसाधनांच्या व्यवस्थापनासाठी अचूकतेसह डेटा-चालित निर्णय घ्या.',
        'benefit_sustain_title': 'टिकाऊ शेती',
        'benefit_sustain_desc': 'बुद्धिमत्तापूर्ण शिफारशींद्वारे उत्पादनक्षमता वाढवताना कचरा आणि पर्यावरणाचा प्रभाव कमी करा.',
        'benefit_realtime_title': 'रिअल-टाइम विश्लेषण',
        'benefit_realtime_desc': 'अत्याधुनिक यंत्र शिक्षण अल्गॉरिदम आणि संगणकीय दृष्टी (कॉम्प्युटर व्हिजन) द्वारे समर्थित त्वरित अंतर्दृष्टी आणि अंदाज मिळवा.',
    },
    'crop_prediction': {
        'main_title': '🌾 बुद्धिमान पीक शिफारस प्रणाली',
        'subtitle': 'तुमच्या माती आणि पर्यावरणीय परिस्थितीनुसार एआय-समर्थित पिकाचे अंदाज मिळवा.',
        'expander_header': 'ℹ️ पीक अंदाजाचे पॅरामीटर्स समजून घेणे',
        'expander_info_text': 'आमचा एआय मॉडेल तुमच्या जमिनीसाठी सर्वोत्तम पिकाची शिफारस करण्यासाठी अनेक घटकांचे विश्लेषण करते. प्रत्येक पॅरामीटर पिकाच्या योग्यतेमध्ये महत्त्वाची भूमिका बजावते:',
        'how_it_works': '📊 **हे कसे काम करते:** आमचा यंत्र शिक्षण अल्गॉरिदम तुमच्या इनपुट डेटावर प्रक्रिया करतो आणि वैयक्तिकृत शिफारशी देण्यासाठी हजारो यशस्वी पीक संयोजनांशी त्याची तुलना करतो.',
        'env_factors_header': '🌡️ पर्यावरणीय घटक',
        'temp_label': '🌡️ तापमान (°C)',
        'temp_info': '<strong>तापमानाचा परिणाम:</strong> सेल्सिअसमध्ये सभोवतालचे तापमान. उष्णकटिबंधीय पिकांना २५-३५°C आवडते, तर समशीतोष्ण पिकांना १५-२५°C आवडते.',
        'hum_label': '💧 आर्द्रता (%)',
        'hum_info': '<strong>आर्द्रतेचा परिणाम:</strong> हवेतील सापेक्ष आर्द्रता टक्केवारी. जास्त आर्द्रता (>७०%) भात सारख्या पिकासाठी योग्य आहे, तर कमी आर्द्रता (<५०%) गहू आणि बार्ली सारख्या पिकासाठी चांगली आहे.',
        'rain_label': '🌧️ पर्जन्यमान (मिमी)',
        'rain_info': '<strong>पर्जन्यमानाचा परिणाम:</strong> मिलीमीटरमध्ये सरासरी पर्जन्यमान. भाताला १५०-३०० मिमी आवश्यक आहे, गव्हाला ३०-१०० मिमी आवश्यक आहे, तर दुष्काळ-प्रतिरोधक पिके <५० मिमी मध्ये देखील टिकू शकतात.',
        'ph_label': '⚗️ मातीचा pH स्तर',
        'ph_info': '<strong>pH चा परिणाम:</strong> मातीचा pH मूल्य आम्लता/अल्कधर्मीयता मोजतो. बहुतेक पिकांना ६.०-७.५ (किंचित आम्ल ते तटस्थ) आवडते. आम्लयुक्त माती (<६) ब्ल्यूबेरीसाठी योग्य आहे, तर अल्कधर्मीय माती (>७.५) शतावरीसाठी योग्य आहे.',
        'nutrients_header': '🧪 मातीतील पोषक तत्व (NPK मूल्य)',
        'n_label': '🔵 नायट्रोजन (N) सामग्री',
        'n_info': '<strong>नायट्रोजनची (N) भूमिका:</strong> पानांच्या वाढीसाठी आणि क्लोरोफिल उत्पादनासाठी आवश्यक. पालेभाज्यांना जास्त N (८०-१२०) आवश्यक असते, तर मुळांच्या भाज्यांना मध्यम N (४०-८०) आवश्यक असते.',
        'p_label': '🟡 फॉस्फरस (P) सामग्री',
        'p_info': '<strong>फॉस्फरसची (P) भूमिका:</strong> मुळांच्या वाढीसाठी आणि फुलांसाठी महत्त्वपूर्ण. फळांच्या पिकांना जास्त P (६०-१००) आवश्यक आहे, तर गवतांना कमी P (२०-४०) आवश्यक आहे.',
        'k_label': '🔴 पोटॅशियम (K) सामग्री',
        'k_info': '<strong>पोटॅशियमची (K) भूमिका:</strong> रोगप्रतिकार आणि जल नियंत्रणासाठी महत्त्वपूर्ण. मुळांच्या भाज्यांना आणि फळांना जास्त K (८०-१५०) आवश्यक आहे, तर धान्यांना मध्यम K (४०-८०) आवश्यक आहे.',
        'summary_header': '📊 वर्तमान इनपुट सारांश',
        'summary_temp': '🌡️ **तापमान:**',
        'summary_hum': '💧 **आर्द्रता:**',
        'summary_rain': '🌧️ **पर्जन्यमान:**',
        'summary_ph': '⚗️ **pH स्तर:**',
        'summary_n': '🔵 **नायट्रोजन (N):**',
        'summary_p': '🟡 **फॉस्फरस (P):**',
        'summary_k': '🔴 **पोटॅशियम (K):**',
        'reference_header': '📋 आदर्श श्रेणी संदर्भ',
        'ref_text': '<strong>इष्टतम वाढीच्या अटी:</strong><br>• **तापमान:** २०-३०°C (बहुतेक पिके)<br>• **आर्द्रता:** ४०-७०% (इष्टतम श्रेणी)<br>• **पर्जन्यमान:** ५०-२०० मिमी (पिकानुसार बदलते)<br>• **pH:** ६.०-७.५ (तटस्थ ते किंचित आम्ल)<br>• **NPK:** निरोगी वाढीसाठी संतुलित गुणोत्तर',
        'warning_temp': '🌡️ तापमान सामान्य वाढीच्या श्रेणीच्या (५-४५°C) बाहेर आहे',
        'warning_hum': '💧 आर्द्रतेची पातळी बहुतेक पिकांसाठी आव्हानात्मक असू शकते',
        'warning_ph': '⚗️ pH स्तर खूप जास्त आहे आणि पिकांचे पर्याय मर्यादित करू शकते',
        'warning_n': '🔵 खूप जास्त नायट्रोजन पातळीमुळे जास्त वाढ होऊ शकते',
        'warning_p': '🟡 जास्त फॉस्फरस पातळी इतर पोषक तत्वांच्या शोषणात अडथळा आणू शकते',
        'warning_k': '🔴 खूप जास्त पोटॅशियम पातळी मातीच्या संरचनेवर परिणाम करू शकते',
        'warnings_header': '⚠️ इनपुट चेतावणी:',
        'validation_header': '✅ प्रमाणीकरण स्थिती',
        'validation_text': 'सर्व इनपुट मूल्ये स्वीकार्य श्रेणींमध्ये आहेत! तुमच्या परिस्थिती पीक लागवडीसाठी खूप चांगल्या दिसत आहेत.',
        'predict_button': '🔮 सर्वोत्तम पिकाचा अंदाज घ्या',
        'loading_1': 'मातीच्या परिस्थितीचे विश्लेषण करत आहे...',
        'loading_2': 'पर्यावरण डेटावर प्रक्रिया करत आहे...',
        'loading_3': 'पीक डेटाबेसशी जुळणी करत आहे...',
        'loading_4': 'शिफारशींना अंतिम रूप देत आहे...',
        'result_header': '🎯 शिफारस केलेले पीक:',
        'result_confidence': '📊 आत्मविश्वासाचा स्कोअर:',
        'result_quality': '🌟 जुळणीची गुणवत्ता:',
        'quality_excellent': 'उत्कृष्ट',
        'quality_good': 'चांगले',
        'quality_fair': 'ठीक',
        'top_3_header': '📈 शीर्ष ३ पीक शिफारशी',
        'crop_season': 'हंगाम',
        'crop_water': 'पाण्याची गरज',
        'crop_match': 'जुळणी',
        'crop_suitability': 'योग्यता',
        'personalized_tips_header': '💡 वैयक्तिकृत शेती टिपा',
        'tips_climate_header': '🌡️ हवामानाचा विचार',
        'tips_temp_high': '<strong>🌡️ जास्त तापमानाची चेतावणी:</strong> उष्णता-प्रतिरोधक वाणांचा, शेड नेटचा आणि वारंवार सिंचनाचा विचार करा. जल कार्यक्षमतेसाठी ठिबक सिंचन बसवा.',
        'tips_temp_low': '<strong>❄️ थंड तापमान:</strong> थंड-हवामानातील पिकांसाठी आदर्श. रो कव्हर आणि ग्रीनहाऊस शेतीसारख्या दंव संरक्षण उपायांचा विचार करा.',
        'tips_temp_ok': '<strong>🌡️ इष्टतम तापमान:</strong> बहुतेक पीक वाणांसाठी योग्य परिस्थिती. नियमित पाणी देणे सुरू ठेवा आणि कीटकांचे निरीक्षण करा.',
        'tips_hum_high': '<strong>💧 जास्त आर्द्रतेची चेतावणी:</strong> बुरशीजन्य रोगांना प्रतिबंध करण्यासाठी योग्य रोपांमधील अंतर आणि वायुवीजन सुनिश्चित करा. बुरशीनाशक उपचारांचा विचार करा.',
        'tips_hum_low': '<strong>🏜️ कमी आर्द्रतेची चेतावणी:</strong> मातीची आर्द्रता टिकवण्यासाठी मल्चिंग आणि वारंवार हलके पाणी देणे विचारात घ्या. आर्द्रता टिकवून ठेवण्याचे तंत्रज्ञान वापरा.',
        'tips_hum_ok': '<strong>💧 चांगली आर्द्रता पातळी:</strong> निरोगी रोपांच्या वाढीसाठी अनुकूल परिस्थिती. योग्य रोपांच्या वाढीसाठी निरीक्षण करा.',
        'tips_soil_header': '🧪 माती व्यवस्थापन',
        'tips_ph_acidic': '<strong>⚗️ आम्लयुक्त माती:</strong> pH वाढवण्यासाठी चुना घालण्याचा विचार करा. ॲल्युमिनियमच्या विषारीपणाची चाचणी घ्या आणि मातीची रचना सुधारण्यासाठी सेंद्रिय पदार्थ घाला.',
        'tips_ph_alkaline': '<strong>⚗️ अल्कधर्मीय माती:</strong> pH कमी करण्यासाठी सल्फर किंवा सेंद्रिय पदार्थ घालण्याचा विचार करा. सूक्ष्म पोषक तत्वांच्या कमतरतेवर लक्ष ठेवा.',
        'tips_ph_ok': '<strong>⚗️ इष्टतम pH श्रेणी:</strong> पोषक तत्वांच्या उपलब्धतेसाठी योग्य परिस्थिती. नियमित सेंद्रिय बदलांसह मातीचे आरोग्य राखा.',
        'tips_n_low': '<strong>🔵 कमी नायट्रोजन:</strong> युरिया किंवा सेंद्रिय खतांसारख्या नायट्रोजन-समृद्ध खतांचा विचार करा. चांगल्या शोषणासाठी विभाजित डोसमध्ये वापरा.',
        'tips_n_high': '<strong>🔵 जास्त नायट्रोजन:</strong> जास्त वाढ होऊ शकते. काळजीपूर्वक निरीक्षण करा आणि आवश्यक असल्यास नायट्रोजन इनपुट कमी करा.',
        'tips_p_low': '<strong>🟡 कमी फॉस्फरस:</strong> डीएपी किंवा रॉक फॉस्फेटचा वापर करण्याचा विचार करा. मुळांच्या वाढीसाठी आणि फुलांसाठी आवश्यक.',
        'tips_k_low': '<strong>🔴 कमी पोटॅशियम:</strong> एमओपी (म्युरेट ऑफ पोटॅश) चा वापर करण्याचा विचार करा. रोगप्रतिकार आणि जल नियंत्रणासाठी महत्त्वपूर्ण.',
        'summary_box_header': '🌟 तुमच्या वैयक्तिकृत पीक शिफारशीचा सारांश',
        'summary_box_text': 'तुमच्या माती आणि पर्यावरणीय परिस्थितीच्या आमच्या एआय विश्लेषणावर आधारित, **{}** हे तुमच्या जमिनीसाठी **{:.1f}% आत्मविश्वासाच्या स्कोअर**सह सर्वात योग्य पीक आहे.',
        'summary_match_quality': '🎯 जुळणीची गुणवत्ता',
        'summary_growth_potential': '🌱 वाढीची क्षमता',
        'summary_econ_viability': '💰 आर्थिक व्यवहार्यता',
        'growth_high': 'जास्त',
        'growth_medium': 'मध्यम',
        'growth_moderate': 'मध्यम',
        'econ_prof': 'लाभदायक',
        'econ_good': 'चांगले',
    },
    'fertilizer_recommendation': {
        'main_title': '🧪 खत शिफारस प्रणाली',
        'subtitle': 'तुमच्या पिकाच्या आणि मातीच्या परिस्थितीनुसार योग्य खतांच्या शिफारशी मिळवा.',
        'section_info': '🌱 पीक आणि माती माहिती',
        'section_env': '🌡️ पर्यावरणीय परिस्थिती',
        'section_nutrients': '🧪 वर्तमान मातीतील पोषक तत्व',
        'crop_type_label': 'पिकाचा प्रकार',
        'soil_type_label': 'मातीचा प्रकार',
        'temp_label': 'तापमान (°C)',
        'hum_label': 'आर्द्रता (%)',
        'moisture_label': 'मातीतील आर्द्रता (%)',
        'nitrogen_label': 'नायट्रोजन सामग्री',
        'phosphorus_label': 'फॉस्फरस सामग्री',
        'potassium_label': 'पोटॅशियम सामग्री',
        'nutrient_status_header': '📊 पोषक तत्वांची स्थिती',
        'low': '🔴 कमी',
        'medium': '🟡 मध्यम',
        'high': '🟢 जास्त',
        'predict_button': '💡 खताची शिफारस मिळवा',
        'result_header': '🎯 शिफारस केलेले खत:',
        'result_confidence': '📊 आत्मविश्वास:',
        'result_info_pre': '',
        'result_info_in': ' मातीमध्ये:',
        'result_info_apply': '- **{}** खत वापरा',
        'result_info_tips': '- प्रमाण ठरवताना सध्याच्या पोषक तत्वांच्या पातळीचा विचार करा\n- योग्य वाढीच्या टप्प्यात वापरा\n- मातीतील आर्द्रता आणि हवामानाची स्थिती तपासा',
        'error_message': 'अंदाजामध्ये त्रुटी. कृपया आपले इनपुट तपासा.',
    },
    'disease_detection': {
        'main_title': '🔬 रोग ओळख',
        'subtitle': 'डीप लर्निंग सीएनएन मॉडेल वापरून पानांच्या फोटोमधून त्वरित रोगाची ओळख करा.',
        'upload_header': '📷 वनस्पतींच्या पानांचा फोटो अपलोड करा',
        'upload_guidelines_title': '📸 फोटो अपलोड मार्गदर्शक तत्त्वे:',
        'upload_guidelines_text': '✓ स्वच्छ, चांगले प्रकाशित पानांचे फोटो<br>✓ बाधित क्षेत्र किंवा लक्षणांवर लक्ष केंद्रित करा<br>✓ समर्थित फॉरमॅट: JPG, PNG, JPEG<br>✓ कमाल आकार: 10MB',
        'file_uploader_label': 'पानाचा फोटो निवडा...',
        'file_uploader_help': 'वनस्पतीच्या पानाचा एक स्वच्छ फोटो अपलोड करा',
        'uploaded_image_caption': 'अपलोड केलेला पानाचा फोटो',
        'analyze_button': '🔍 रोगांसाठी विश्लेषण करा',
        'loading_message': '🧠 एआय फोटोचे विश्लेषण करत आहे...',
        'analysis_complete': 'विश्लेषण पूर्ण झाले!',
        'result_header': '🎯 अंदाजित रोग:',
        'result_confidence': '📊 आत्मविश्वास:',
        'disease_warning': '❗ तुमच्या वनस्पतीला रोग झाला असावा. कृपया पुष्टीकरणासाठी व्यावसायिकांचा सल्ला घ्या.',
        'healthy_message': '✅ वनस्पती निरोगी दिसत आहे!',
    },
    'about_page': {
        'main_title': '👥 आमच्याबद्दल',
        'subtitle': 'स्मार्ट शेती क्रांतीमागील नाविन्यपूर्ण टीमला भेटा!',
        'mission_title': '🌟 आमचे ध्येय',
        'mission_text': 'डीपअॅग्रो अत्याधुनिक एआय आणि यंत्र शिक्षण तंत्रज्ञानाद्वारे पारंपारिक शेतीत परिवर्तन घडवण्यासाठी समर्पित आहे. चांगले पीक निवड, इष्टतम खताचा वापर आणि लवकर रोग ओळख यासाठी उपयुक्त अंतर्दृष्टी देऊन शेतकऱ्यांना सशक्त बनवणे हे आमचे ध्येय आहे.',
        'team_header': '👨‍💻 आमची विकास टीम',
        'team_desc': 'आयआयआयटी रायचूरमधील विद्यार्थ्यांचा एक उत्साही गट तंत्रज्ञानाने शेतीत क्रांती घडवण्यासाठी एकत्र काम करत आहे.',
        'tech_stack_header': '🛠️ तंत्रज्ञान स्टॅक',
        'ml_title': '🤖 यंत्र शिक्षण',
        'ml_text': '• रँडम फॉरेस्ट क्लासिफायर<br>• सायकीट-लर्न<br>• नम्पाई आणि पांडा<br>• फीचर इंजिनीअरिंग',
        'web_title': '🌐 वेब फ्रेमवर्क',
        'web_text': '• स्ट्रीमलिट<br>• पायथन बॅकएंड<br>• इंटरअॅक्टिव्ह यूआय/यूएक्स<br>• रिअल-टाइम प्रोसेसिंग',
        'data_title': '📊 डेटा आणि व्हिज्युअलायझेशन',
        'data_text': '• चार्टसाठी प्लॉटली<br>• फोटो प्रोसेसिंगसाठी पीआयएल<br>• कस्टम सीएसएस स्टाइलिंग<br>• रिस्पॉन्सिव्ह डिझाइन',
        'features_header': '✨ प्रमुख वैशिष्ट्ये',
        'smart_pred_header': '🎯 स्मार्ट अंदाज',
        'smart_pred_list': '- **पीक शिफारस:** माती आणि हवामानावर आधारित एआय-समर्थित पीक निवड\n- **खत अनुकूलन:** जास्तीत जास्त उत्पादनासाठी उपयुक्त खतांच्या शिफारशी\n- **रोग ओळख:** वनस्पतींच्या रोगांच्या ओळखीसाठी संगणकीय दृष्टी',
        'ux_header': '🔧 वापरकर्ता अनुभव',
        'ux_list': '- **इंटरअॅक्टिव्ह इंटरफेस:** वापरण्यास सोपे स्लाइडर आणि इनपुट फील्ड\n- **रिअल-टाइम विश्लेषण:** त्वरित अंदाज आणि शिफारशी\n- **शैक्षणिक सामग्री:** तपशीलवार स्पष्टीकरण आणि शेती टिपा',
        'institution_title': '🏫 संस्था',
        'institution_text': '<strong>इंडियन इन्स्टिट्यूट ऑफ इन्फॉर्मेशन टेक्नॉलॉजी, रायचूर</strong><br>शेती तंत्रज्ञान आणि टिकाऊ शेती उपायांमध्ये नाविन्य आणणे.',
        'acknowledgements_title': '🙏 आभार',
        'acknowledgements_text': 'या शेती एआय उपायाची निर्मिती करण्यासाठी त्यांचे समर्थन आणि मार्गदर्शनासाठी आमचे प्राध्यापक मार्गदर्शक डॉ. प्रियोद्यूती प्रधान आणि आयआयआयटी रायचूर समुदायाचे विशेष आभार.',
        'footer_title': '🌱 **डीपअॅग्रो**',
        'footer_slogan': 'एआय आणि एमएल सह शेतीला सशक्त बनवणे',
        'footer_credit': '❤️ टीम डीपअॅग्रो द्वारे निर्मित | आयआयआयटी रायचूर | २०२५'
    }
}
}
# Set page config
st.set_page_config(
    page_title=translations['en']['page_title'], # Default title
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main {
        background-color: #f8fdf8;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1b5e20, #2e7d32, #43a047);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(46, 125, 50, 0.2);
    }
    
    .logo {
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    
    .tagline {
        font-size: 1.4rem;
        opacity: 0.95;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #ffffff, #f1f8e9);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #4caf50;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease;
        color: #1b5e20;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-card h4 {
        color: #1b5e20;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .feature-card p {
        color: #2e7d32;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #e8f5e8);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 3px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(76, 175, 80, 0.2);
    }
    
    .metric-card h4 {
        color: #1b5e20 !important;
        margin: 0 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    css<br> /* Chatbot Styles */<br> .chat-message-container {<br> display: flex;<br> margin-bottom: 15px;<br> align-items: flex-end;<br> }<br> .user-message {<br> justify-content: flex-end;<br> text-align: right;<br> }<br> .message-content {<br> max-width: 80%;<br> padding: 12px 18px;<br> border-radius: 20px;<br> font-size: 1rem;<br> line-height: 1.5;<br> box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);<br> }<br> .user-message .message-content {<br> background-color: #e3f2fd; /* Light Blue */<br> color: #1a237e;<br> border-bottom-right-radius: 5px;<br> }<br> .ai-message .message-content {<br> background-color: #e8f5e8; /* Light Green */<br> color: #1b5e20;<br> border-bottom-left-radius: 5px;<br> }<br> .message-icon {<br> font-size: 2rem;<br> margin: 0 10px;<br> line-height: 1;<br> }<br>
    .metric-card p {
        color: #2e7d32 !important;
        margin: 0.5rem 0 0 0 !important;
        font-weight: 500 !important;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #ffffff, #c8e6c9);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.1);
        border: 2px solid #81c784;
        color: #1b5e20;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #ffffff, #f1f8e9);
        padding: 2.5rem;
        border-radius: 15px;
        border: 3px dashed #4caf50;
        text-align: center;
        margin: 1.5rem 0;
        color: #2e7d32;
    }
    
    .upload-section h4 {
        color: #1b5e20;
        margin-bottom: 1rem;
    }
    /* Bidirectional Diagnostic Scan: Line moves Top -> Bottom, then Bottom -> Top, repeating. */

@keyframes bidirectional-scan {
    /* Step 1: Start (Line is outside the top border) */
    0%   { background-position: 0 -100%; } 
    
    /* Step 2: Midpoint (Line reaches the bottom border) - This is the DOWNWARD scan */
    50%  { background-position: 0 100%; } 
    
    /* Step 3: Reset/Pause (Line is outside the bottom border) */
    50.01% { background-position: 0 100%; } /* Slight pause/reset point before reverse */

    /* Step 4: Finish (Line is back outside the top border) - This is the UPWARD scan */
    100% { background-position: 0 -100%; } 
}

@keyframes result-glow {
    /* Final subtle glow for the total duration */
    0%   { box-shadow: 0 0 0px #4CAF50; border-color: #3f51b5; }
    100% { box-shadow: 0 0 15px 5px #4CAF50; border-color: #4CAF50; }
}

.diagnosing-container {
    /* Main container styling and final glow */
    border: 3px solid #3f51b5; /* Initial Blue Border */
    border-radius: 10px;
    overflow: hidden;
    position: relative; 
    
    /* Apply the final result glow/border change over the full duration */
    animation: result-glow 4.0s forwards; /* Duration matched to total animation time */
}

.diagnosing-container::before {
    /* This pseudo-element creates the moving scan line */
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none; 
    
    /* The scan line definition (fixed horizontal light band) */
    background: linear-gradient(
        0deg, /* Angle remains 0deg for vertical movement */
        rgba(255, 255, 255, 0) 0%,      
        rgba(255, 255, 255, 0.5) 50%, 
        rgba(255, 255, 255, 0) 100%    
    );
    background-size: 100% 10%; /* Tall and thin gradient */
    background-repeat: no-repeat;
    
    /* Run the entire Bidirectional cycle (Down and Up) twice over 4.0 seconds */
    animation: bidirectional-scan 2.0s linear infinite; 
    animation-iteration-count: 2; /* Runs twice (2 seconds/cycle * 2 cycles = 4 seconds) */
}

.zooming-image {
    width: 100%;
    height: auto;
    display: block;
}
    .parameter-info {
        background: linear-gradient(135deg, #e8f5e8, #f1f8e9);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.9rem;
        color: #2e7d32;
    }
    
    .team-card {
        background: linear-gradient(135deg, #ffffff, #f1f8e9);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 2px solid #4caf50;
        transition: transform 0.3s ease;
        color: #1b5e20;
        min-height: 180px;
    }
    
    .team-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(76, 175, 80, 0.2);
    }
    
    .team-card h3 {
        color: #1b5e20;
        margin-bottom: 0.5rem;
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    .team-card p {
        color: #2e7d32;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0.3rem 0;
    }
    
    .team-card .roll-number {
        color: #4caf50;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32, #4caf50);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.4);
    }
    
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff, #e8f5e8);
        border: 1px solid rgba(76, 175, 80, 0.2);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="metric-container"] > div {
        color: #1b5e20;
    }
</style>
""", unsafe_allow_html=True)

# Sample datasets
@st.cache_data
def load_crop_data():
    """Create sample crop recommendation dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'N': np.random.randint(0, 140, n_samples),
        'P': np.random.randint(5, 145, n_samples),
        'K': np.random.randint(5, 205, n_samples),
        'temperature': np.random.uniform(8.8, 43.7, n_samples),
        'humidity': np.random.uniform(14.3, 99.9, n_samples),
        'ph': np.random.uniform(3.5, 9.9, n_samples),
        'rainfall': np.random.uniform(20.2, 298.6, n_samples)
    }
    
    crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
             'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
             'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
             'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    
    data['label'] = np.random.choice(crops, n_samples)
    
    return pd.DataFrame(data)


@st.cache_resource
def load_cnn_model_and_labels():
    """
    Loads the CNN model using the robust architecture rebuild + load_weights() method 
    to bypass the known Mixed Precision/MobileNetV2 loading error.
    """
    import streamlit as st # Already present, but needed for st.success/warning
    import pickle # Used for loading class_labels.pkl

    # --- TensorFlow and Keras Imports (Crucial for the Rebuild) ---
    import tensorflow as tf 
    from tensorflow import keras
    from keras.applications import MobileNetV2
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from keras.regularizers import l2

# Note: numpy and time are also required by the surrounding Streamlit application logic (e.g., show_crop_prediction) 
# and should already be at the top of your file:
# import numpy as np 
# import time
    
    # CRITICAL: Use the file name available in your Streamlit directory for weights
    model_path = 'best_cnn.keras'  
    labels_path = 'disease_labels.pkl' 
    TARGET_SIZE = (224, 224) 

    try:
        # --- 1. Load Class Labels and Get NUM_CLASSES ---
        with open(labels_path, 'rb') as f:
            class_labels = pickle.load(f)
            
        if isinstance(class_labels, dict):
            NUM_CLASSES = len(class_labels)
        else: # Handle list of class names if saved differently
            NUM_CLASSES = len(class_labels)
            class_labels = {i: label for i, label in enumerate(class_labels)}


        # --- 2. Rebuild the Model Architecture (Exactly as in Notebook Cell 8) ---
        
        # 2.1 Set up Mixed Precision Policy if GPU is available (CRUCIAL for loading float16 weights)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
             keras.mixed_precision.set_global_policy('mixed_float16')
        else:
            keras.mixed_precision.set_global_policy('float32')

        # 2.2 Recreate Base Model (Input shape (224, 224, 3))
        base_model = MobileNetV2(
            input_shape=(*TARGET_SIZE, 3),
            include_top=False,
            weights='imagenet' # Re-download weights for the base model
        )
        base_model.trainable = False 

        # 2.3 Recreate Custom Classification Head (Matching Notebook Cell 8)
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)), 
            Dropout(0.6), 
            # Final output MUST use dtype='float32'
            Dense(NUM_CLASSES, activation='softmax', dtype='float32') 
        ])

        # --- 3. Load Weights and Recompile ---
        
        # The problem is that model.load_weights(model_path) often fails 
        # when the model was saved with mixed precision. 
        # We must load the weights into the new structure.
        
        # NOTE: If your 'best_cnn.keras' file only contains weights (saved via model.save_weights),
        # this will work. If it contains the full model (saved via model.save), 
        # we will force Keras to load only the compatible weights.
        
        model.load_weights(model_path) 
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        st.success("✅ CNN model successfully rebuilt and weights loaded! 🔬")

        return model, class_labels

    except Exception as e:
        st.error(f"❌ Critical Error loading CNN model: {type(e).__name__} - {str(e)}")
        st.warning("⚠️ The specific error points to an internal conflict during Keras model reconstruction. The only workaround is to ensure the code exactly matches the training architecture.")
        
        # Fallback to prevent crash
        class_labels_fallback = {0: "Apple___healthy", 1: "Apple___scab"}
        return None, class_labels_fallback
@st.cache_data
def load_fertilizer_data():
    """Create sample fertilizer dataset"""
    np.random.seed(42)
    n_samples = 800
    
    data = {
        'Temparature': np.random.uniform(10, 45, n_samples),
        'Humidity': np.random.uniform(20, 95, n_samples),
        'Moisture': np.random.uniform(0, 100, n_samples),
        'Soil_Type': np.random.choice(['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'], n_samples),
        'Crop_Type': np.random.choice(['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 
                                     'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 
                                     'Ground Nuts'], n_samples),
        'Nitrogen': np.random.randint(0, 50, n_samples),
        'Potassium': np.random.randint(0, 50, n_samples),
        'Phosphorous': np.random.randint(0, 50, n_samples)
    }
    
    fertilizers = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea']
    data['Fertilizer_Name'] = np.random.choice(fertilizers, n_samples)
    
    return pd.DataFrame(data)

# Train models
@st.cache_resource
def train_crop_model():
    """Load the pre-trained XGBoost Pipeline and the separate LabelEncoder."""
    
    pipeline_path = 'xgb_pipeline.pkl'
    encoder_path = 'crop_label_encoder.pkl' # Assuming you saved the encoder here

    try:
        # Load the fitted Pipeline
        model_pipeline = joblib.load(pipeline_path)
        
        # Load the fitted LabelEncoder
        le = joblib.load(encoder_path)
        
        # st.success("Pre-trained Pipeline (Scaler + XGBoost) and LabelEncoder Loaded Successfully! 🌱")
        return model_pipeline, le

    except FileNotFoundError:
        st.error(f"Error: Required files ('{pipeline_path}' and/or '{encoder_path}') not found. Falling back to mock model.")
        
        # --- Fallback: Create and return a mock model and its encoder ---
        df = load_crop_data()
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        le_fallback = LabelEncoder()
        y_encoded = le_fallback.fit_transform(y)
        
        # Use LogisticRegression (or XGBoost directly) as a simple fallback model
        # NOTE: We keep the variable name 'model_fallback' to prevent caching confusion

        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        pipeline_fallback = Pipeline(steps=[
            ('scaler', StandardScaler()),  # Scaling is likely part of the real pipeline
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
        ])
        pipeline_fallback.fit(X, y_encoded)
        st.warning("Using mock XGBoost model for demo purposes.")
        return pipeline_fallback, le_fallback
    
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        st.warning("Please ensure both 'xgb_pipeline.pkl' and 'crop_label_encoder.pkl' are saved correctly.")
        # Return a simple fallback
        df = load_crop_data()
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        le_fallback = LabelEncoder()
        y_encoded = le_fallback.fit_transform(y)
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        pipeline_fallback = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, verbosity=0))
        ])
        pipeline_fallback.fit(X, y_encoded)
        
        return pipeline_fallback, le_fallback
    

@st.cache_resource
def train_fertilizer_model():
    """
    Loads the pre-trained Pipeline (randpipe.pkl) and creates a LabelEncoder 
    for fertilizer name decoding (as the encoder was not saved in the provided notebook).
    """
    pipeline_path = 'randpipe.pkl'
    
    # Define fallback classes for the LabelEncoder using the sample data
    df_data = load_fertilizer_data()
    
    # 1. Load the Pipeline (randpipe.pkl)
    try:
        with open(pipeline_path, 'rb') as f:
            model_pipeline = pickle.load(f)
        st.success("Pre-trained Fertilizer Pipeline Loaded Successfully! 🧪")

        # 2. Recreate LabelEncoder using the full set of labels from the sample data
        # This assumes the sample data contains all possible output classes.
        le_fertilizer = LabelEncoder()
        # We need to fit the encoder on the entire original label set to ensure correct decoding
        le_fertilizer.fit(df_data['Fertilizer_Name']) 
        
        # NOTE: The other two encoders (le_soil and le_crop) are technically not needed 
        # in the calling function because the pipeline handles the encoding. 
        # We return a placeholder for le_soil and le_crop as per the function signature.
        le_soil_placeholder = LabelEncoder().fit(df_data['Soil_Type'])
        le_crop_placeholder = LabelEncoder().fit(df_data['Crop_Type'])
        
        return model_pipeline, le_soil_placeholder, le_crop_placeholder, le_fertilizer

    except FileNotFoundError:
        st.error(f"Error: Required pipeline file ('{pipeline_path}') not found. Falling back to mock model training.")
        
        # --- Fallback: Train and return a mock model (original app logic) ---
        le_soil_fallback = LabelEncoder()
        le_crop_fallback = LabelEncoder()
        le_fert_fallback = LabelEncoder()
        
        df_data['Soil_Encoded'] = le_soil_fallback.fit_transform(df_data['Soil_Type'])
        df_data['Crop_Encoded'] = le_crop_fallback.fit_transform(df_data['Crop_Type'])
        y_encoded = le_fert_fallback.fit_transform(df_data['Fertilizer_Name'])
        
        X = df_data[['Temparature', 'Humidity', 'Moisture', 'Soil_Encoded', 
                     'Crop_Encoded', 'Nitrogen', 'Potassium', 'Phosphorous']]
        
        model_fallback = RandomForestClassifier(n_estimators=100, random_state=42)
        model_fallback.fit(X, y_encoded)
        
        st.warning("Using mock RandomForestClassifier for demo purposes.")
        return model_fallback, le_soil_fallback, le_crop_fallback, le_fert_fallback
    
    except Exception as e:
        st.error(f"Error loading model component '{pipeline_path}': {e}. Falling back to mock training.")
        
        # --- Fallback: Train and return a mock model (original app logic) ---
        le_soil_fallback = LabelEncoder()
        le_crop_fallback = LabelEncoder()
        le_fert_fallback = LabelEncoder()
        
        df_data['Soil_Encoded'] = le_soil_fallback.fit_transform(df_data['Soil_Type'])
        df_data['Crop_Encoded'] = le_crop_fallback.fit_transform(df_data['Crop_Type'])
        y_encoded = le_fert_fallback.fit_transform(df_data['Fertilizer_Name'])
        
        X = df_data[['Temparature', 'Humidity', 'Moisture', 'Soil_Encoded', 
                     'Crop_Encoded', 'Nitrogen', 'Potassium', 'Phosphorous']]
        
        model_fallback = RandomForestClassifier(n_estimators=100, random_state=42)
        model_fallback.fit(X, y_encoded)
        
        return model_fallback, le_soil_fallback, le_crop_fallback, le_fert_fallback

# Main App
def main():
    if 'lang' not in st.session_state:
        st.session_state.lang = 'en'

    # Sidebar language selector
    st.sidebar.markdown("**🌐 Select Language**", unsafe_allow_html=True)
    lang_choice = st.sidebar.selectbox(
        "",
        ('English', 'हिन्दी', 'தமிழ் (Tamil)', 'తెలుగు (Telugu)',
         'मराठी (Marathi)', 'ਪੰਜਾਬੀ (Punjabi)', 'ଓଡ଼ିଆ (Odia)', 'বাংলা (Bengali)'),
        key='lang_select'
    )

    # Map the language names to their corresponding language codes
    lang_map = {
        'English': 'en',
        'हिन्दी': 'hi',
        'தமிழ் (Tamil)': 'ta',
        'తెలుగు (Telugu)': 'tel',
        'मराठी (Marathi)': 'mr',
        'ਪੰਜਾਬੀ (Punjabi)': 'pa',
        'ଓଡ଼ିଆ (Odia)': 'or',
        'বাংলা (Bengali)': 'bn'
    }

    # Set the session state language based on the map
    st.session_state.lang = lang_map.get(lang_choice, 'en')

    # Get the correct language strings
    lang = st.session_state.lang
    t = translations[lang]

    # Header
    st.markdown(f"""
        <div class="main-header">
            <div class="logo">{t['home']['header_logo']}</div>
            <div class="tagline">{t['home']['header_tagline']}</div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar header
    st.sidebar.markdown(f"""
        <div style="text-align: center; padding: 1rem; 
                    background: linear-gradient(135deg, #2E7D32, #4CAF50); 
                    color: white; border-radius: 10px; margin-bottom: 1rem;">
            <h2>{t['sidebar_title']}</h2>
        </div>
    """, unsafe_allow_html=True)

    # Navigation
    page = st.sidebar.selectbox(
        t['sidebar_title'],
        [t['nav_home'], t['nav_crop'], t['nav_fertilizer'], t['nav_chat'],  
        t['nav_disease'], t['nav_about']],
        key='page_select'
    )

    # Page routing
    if page == t['nav_home']:
        show_home_page()
    elif page == t['nav_crop']:
        show_crop_prediction()
    elif page == t['nav_fertilizer']:
        show_fertilizer_recommendation()
    elif page == t['nav_disease']:
        show_disease_detection()
    elif page == t['nav_about']:
        show_about_page()
    elif page == t['nav_chat']:
        show_chatbot_page()


def show_home_page():
    t = translations[st.session_state.lang]
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"## {t['home']['welcome_header']}")
        st.markdown(f"**{t['home']['welcome_text']}**")
        
        st.markdown(f"""
        <div class="feature-card">
            <h4>{t['home']['card_crop_title']}</h4>
            <p>{t['home']['card_crop_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <h4>{t['home']['card_fert_title']}</h4>
            <p>{t['home']['card_fert_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <h4>{t['home']['card_disease_title']}</h4>
            <p>{t['home']['card_disease_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #ffffff, #f1f8e9); border-radius: 20px; margin: 1rem 0; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08); border: 1px solid rgba(76, 175, 80, 0.2);">
            <h3 style="color: #1b5e20; margin-bottom: 1.5rem; font-size: 1.8rem; font-weight: 700;">{t['home']['metrics_header']}</h3>
            <div class="metric-card">
                <h4>22+</h4>
                <p>{t['home']['metric_crops']}</p>
            </div>
            <div class="metric-card">
                <h4>7+</h4>
                <p>{t['home']['metric_fertilizers']}</p>
            </div>
            <div class="metric-card">
                <h4>95%+</h4>
                <p>{t['home']['metric_accuracy']}</p>
            </div>
            <div class="metric-card">
                <h4>AI/ML</h4>
                <p>{t['home']['metric_power']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional features section
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem 0;">
        <h2 style="color: #1b5e20; font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">{t['home']['why_choose_title']}</h2>
        <p style="color: #2e7d32; font-size: 1.2rem; max-width: 800px; margin: 0 auto; line-height: 1.8;">
            {t['home']['why_choose_desc']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Benefits grid
    benefit_col1, benefit_col2, benefit_col3 = st.columns(3)
    
    with benefit_col1:
        st.markdown(f"""
        <div class="feature-card" style="text-align: center; min-height: 200px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
            <h4>{t['home']['benefit_precision_title']}</h4>
            <p>{t['home']['benefit_precision_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with benefit_col2:
        st.markdown(f"""
        <div class="feature-card" style="text-align: center; min-height: 200px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌱</div>
            <h4>{t['home']['benefit_sustain_title']}</h4>
            <p>{t['home']['benefit_sustain_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with benefit_col3:
        st.markdown(f"""
        <div class="feature-card" style="text-align: center; min-height: 200px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
            <h4>{t['home']['benefit_realtime_title']}</h4>
            <p>{t['home']['benefit_realtime_desc']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_crop_prediction():
    t = translations[st.session_state.lang]
    lang = st.session_state.lang 
    # Custom CSS for enhanced styling and animations
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: slideInDown 1s ease-out;
    }
    
    @keyframes slideInDown {
        from {
            transform: translateY(-100px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes fadeInUp {
        from {
            transform: translateY(50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes slideInLeft {
        from {
            transform: translateX(-100px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes bounceIn {
        0%, 20%, 40%, 60%, 80% {
            transform: translateY(0);
        }
        10% {
            transform: translateY(-10px);
        }
        30% {
            transform: translateY(-5px);
        }
        50% {
            transform: translateY(-3px);
        }
        70% {
            transform: translateY(-1px);
        }
    }
    
    .subtitle {
        font-family: 'Poppins', sans-serif;
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out 0.3s both;
    }
    
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #e3f2fd, #f1f8e9);
        border-radius: 10px;
        border-left: 5px solid #667eea;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .parameter-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .parameter-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .parameter-info {
        background: linear-gradient(135deg, #e8f5e8, #f0f8ff);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 4px solid #4CAF50;
        font-size: 0.9rem;
        color: #2c5530;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .input-label {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #2c3e50;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
        display: block;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.3);
        animation: bounceIn 1s ease-out;
        font-family: 'Poppins', sans-serif;
    }
    
    .current-values-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        animation: fadeInUp 1s ease-out;
    }
    
    .crop-recommendation {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .crop-recommendation:hover {
        border-color: #4CAF50;
        transform: translateX(10px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
                
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        font-family: 'Poppins', sans-serif;
        animation: pulse 3s infinite;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        animation: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title with animation
    st.markdown(f'<h1 class="main-title">{t["crop_prediction"]["main_title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{t["crop_prediction"]["subtitle"]}</p>', unsafe_allow_html=True)
    
    # Information section with enhanced styling
    with st.expander(t["crop_prediction"]["expander_header"], expanded=False):
        st.markdown(f"""
        <div class="parameter-info">
        {t["crop_prediction"]["expander_info_text"]}
        
        {t["crop_prediction"]["how_it_works"]}
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f'<div class="section-header">{t["crop_prediction"]["env_factors_header"]}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="parameter-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="input-label">{t["crop_prediction"]["temp_label"]}</span>', unsafe_allow_html=True)
        temp_col1, temp_col2 = st.columns([2, 1])
        with temp_col1:
            temperature = st.slider("", 0.0, 50.0, 25.0, step=0.1, key="temp_slider", label_visibility="collapsed")
        with temp_col2:
            temperature = st.number_input("", min_value=0.0, max_value=50.0, value=temperature, step=0.1, key="temp_input", label_visibility="collapsed")
        st.markdown(f"""
        <div class="parameter-info">
        {t["crop_prediction"]["temp_info"]}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="parameter-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="input-label">{t["crop_prediction"]["hum_label"]}</span>', unsafe_allow_html=True)
        hum_col1, hum_col2 = st.columns([2, 1])
        with hum_col1:
            humidity = st.slider("", 0.0, 100.0, 50.0, step=0.1, key="hum_slider", label_visibility="collapsed")
        with hum_col2:
            humidity = st.number_input("", min_value=0.0, max_value=100.0, value=humidity, step=0.1, key="hum_input", label_visibility="collapsed")
        st.markdown(f"""
        <div class="parameter-info">
        {t["crop_prediction"]["hum_info"]}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="parameter-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="input-label">{t["crop_prediction"]["rain_label"]}</span>', unsafe_allow_html=True)
        rain_col1, rain_col2 = st.columns([2, 1])
        with rain_col1:
            rainfall = st.slider("", 0.0, 300.0, 100.0, step=0.1, key="rain_slider", label_visibility="collapsed")
        with rain_col2:
            rainfall = st.number_input("", min_value=0.0, max_value=300.0, value=rainfall, step=0.1, key="rain_input", label_visibility="collapsed")
        st.markdown(f"""
        <div class="parameter-info">
        {t["crop_prediction"]["rain_info"]}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="parameter-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="input-label">{t["crop_prediction"]["ph_label"]}</span>', unsafe_allow_html=True)
        ph_col1, ph_col2 = st.columns([2, 1])
        with ph_col1:
            ph = st.slider("", 0.0, 14.0, 7.0, step=0.1, key="ph_slider", label_visibility="collapsed")
        with ph_col2:
            ph = st.number_input("", min_value=0.0, max_value=14.0, value=ph, step=0.1, key="ph_input", label_visibility="collapsed")
        st.markdown(f"""
        <div class="parameter-info">
        {t["crop_prediction"]["ph_info"]}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f'<div class="section-header">{t["crop_prediction"]["nutrients_header"]}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="parameter-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="input-label">{t["crop_prediction"]["n_label"]}</span>', unsafe_allow_html=True)
        n_col1, n_col2 = st.columns([2, 1])
        with n_col1:
            nitrogen = st.slider("", 0, 150, 50, key="n_slider", label_visibility="collapsed")
        with n_col2:
            nitrogen = st.number_input("", min_value=0, max_value=150, value=nitrogen, step=1, key="n_input", label_visibility="collapsed")
        st.markdown(f"""
        <div class="parameter-info">
        {t["crop_prediction"]["n_info"]}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="parameter-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="input-label">{t["crop_prediction"]["p_label"]}</span>', unsafe_allow_html=True)
        p_col1, p_col2 = st.columns([2, 1])
        with p_col1:
            phosphorus = st.slider("", 0, 150, 50, key="p_slider", label_visibility="collapsed")
        with p_col2:
            phosphorus = st.number_input("", min_value=0, max_value=150, value=phosphorus, step=1, key="p_input", label_visibility="collapsed")
        st.markdown(f"""
        <div class="parameter-info">
        {t["crop_prediction"]["p_info"]}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="parameter-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="input-label">{t["crop_prediction"]["k_label"]}</span>', unsafe_allow_html=True)
        k_col1, k_col2 = st.columns([2, 1])
        with k_col1:
            potassium = st.slider("", 0, 200, 50, key="k_slider", label_visibility="collapsed")
        with k_col2:
            potassium = st.number_input("", min_value=0, max_value=200, value=potassium, step=1, key="k_input", label_visibility="collapsed")
        st.markdown(f"""
        <div class="parameter-info">
        {t["crop_prediction"]["k_info"]}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="current-values-card">
        <h3 style="margin-top: 0; color: white;">{t["crop_prediction"]["summary_header"]}</h3>
        <div style="font-size: 1rem; line-height: 1.6;">
        {t["crop_prediction"]["summary_temp"]} {temperature:.1f}°C<br>
        {t["crop_prediction"]["summary_hum"]} {humidity:.1f}%<br>
        {t["crop_prediction"]["summary_rain"]} {rainfall:.1f}mm<br>
        {t["crop_prediction"]["summary_ph"]} {ph:.1f}<br>
        {t["crop_prediction"]["summary_n"]} {nitrogen}<br>
        {t["crop_prediction"]["summary_p"]} {phosphorus}<br>
        {t["crop_prediction"]["summary_k"]} {potassium}
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="parameter-info" style="margin-top: 1.5rem;">
        <h4 style="margin-top: 0; color: #2c5530;">{t["crop_prediction"]["reference_header"]}</h4>
        {t["crop_prediction"]["ref_text"]}
        </div>
        """, unsafe_allow_html=True)
    
    warnings = []
    if temperature < 5 or temperature > 45:
        warnings.append(t['crop_prediction']['warning_temp'])
    if humidity < 20 or humidity > 95:
        warnings.append(t['crop_prediction']['warning_hum'])
    if ph < 4 or ph > 10:
        warnings.append(t['crop_prediction']['warning_ph'])
    if nitrogen > 140:
        warnings.append(t['crop_prediction']['warning_n'])
    if phosphorus > 140:
        warnings.append(t['crop_prediction']['warning_p'])
    if potassium > 180:
        warnings.append(t['crop_prediction']['warning_k'])
    
    if warnings:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fff3cd, #ffeaa7); padding: 1.5rem; text:black; border-radius: 10px; border-left: 4px solid #856404; margin: 1rem 0;">
        <h4 style="color: #856404; margin-top: 0;">{t['crop_prediction']['warnings_header']}</h4>
        {'<br>'.join([f'• {w}' for w in warnings])}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #d4edda, #c3e6cb); padding: 1.5rem; test:black; border-radius: 10px; border-left: 4px solid #155724; margin: 1rem 0;">
        <h4 style="color: #155724; margin-top: 0;">{t['crop_prediction']['validation_header']}</h4>
        {t['crop_prediction']['validation_text']}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_button = st.button(t['crop_prediction']['predict_button'], type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner(''):
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            for i in range(100):
                if i < 30:
                    progress_text.markdown(f'<div style="text-align: center; font-family: Poppins;"><span class="loading-spinner"></span>{t["crop_prediction"]["loading_1"]} {i+1}%</div>', unsafe_allow_html=True)
                elif i < 60:
                    progress_text.markdown(f'<div style="text-align: center; font-family: Poppins;"><span class="loading-spinner"></span>{t["crop_prediction"]["loading_2"]} {i+1}%</div>', unsafe_allow_html=True)
                elif i < 90:
                    progress_text.markdown(f'<div style="text-align: center; font-family: Poppins;"><span class="loading-spinner"></span>{t["crop_prediction"]["loading_3"]} {i+1}%</div>', unsafe_allow_html=True)
                else:
                    progress_text.markdown(f'<div style="text-align: center; font-family: Poppins;"><span class="loading-spinner"></span>{t["crop_prediction"]["loading_4"]} {i+1}%</div>', unsafe_allow_html=True)
                
                progress_bar.progress(i + 1)
                time.sleep(0.03)
            
            progress_text.empty()
            progress_bar.empty()
        

        
        
        model_pipeline, le = train_crop_model()
        
        input_values = [[nitrogen, phosphorus, potassium, 
                     temperature, humidity, ph, rainfall]]
        

        column_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_df = pd.DataFrame(input_values, columns=column_names)
        input_df = input_df.astype('float64')

        
        prediction_encoded = int(model_pipeline.predict(input_df)[0])
        probabilities = model_pipeline.predict_proba(input_df)[0]
        
        prediction = le.inverse_transform([prediction_encoded])[0]
        confidence = probabilities[prediction_encoded] * 100
        
        match_quality = t['crop_prediction']['quality_excellent'] if confidence > 80 else t['crop_prediction']['quality_good'] if confidence > 60 else t['crop_prediction']['quality_fair']
        
        st.markdown("---")
        st.markdown(f"""
        <div class="prediction-result">
            {t['crop_prediction']['result_header']} <strong>{prediction.upper()}</strong><br>
            {t['crop_prediction']['result_confidence']} <strong>{confidence:.1f}%</strong><br>
            {t['crop_prediction']['result_quality']} <strong>{match_quality}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        crop_classes = le.classes_
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        
        st.markdown(f'<div class="section-header">{t["crop_prediction"]["top_3_header"]}</div>', unsafe_allow_html=True)
        
        crop_info = {
            'rice': {'icon': '🌾', 'desc': 'Water-intensive crop, high humidity preferred', 'season': 'Kharif', 'water': 'High', 'color': '#8BC34A', 'desc_hi': 'पानी की अधिक आवश्यकता वाली फसल, उच्च आर्द्रता पसंद करती है।', 'season_hi': 'खरीफ', 'water_hi': 'उच्च'},
            'maize': {'icon': '🌽', 'desc': 'Moderate water needs, warm climate', 'season': 'Kharif/Rabi', 'water': 'Medium', 'color': '#FFC107', 'desc_hi': 'मध्यम पानी की आवश्यकता, गर्म जलवायु।', 'season_hi': 'खरीफ/रबी', 'water_hi': 'मध्यम'},
            'wheat': {'icon': '🌾', 'desc': 'Cool climate crop, moderate water', 'season': 'Rabi', 'water': 'Medium', 'color': '#FF9800', 'desc_hi': 'ठंडी जलवायु की फसल, मध्यम पानी।', 'season_hi': 'रबी', 'water_hi': 'मध्यम'},
            'cotton': {'icon': '🌿', 'desc': 'High temperature, moderate rainfall', 'season': 'Kharif', 'water': 'Medium-High', 'color': '#4CAF50', 'desc_hi': 'उच्च तापमान, मध्यम वर्षा।', 'season_hi': 'खरीफ', 'water_hi': 'मध्यम-उच्च'},
            'sugarcane': {'icon': '🎋', 'desc': 'High water and temperature needs', 'season': 'Year-round', 'water': 'Very High', 'color': '#2196F3', 'desc_hi': 'उच्च पानी और तापमान की आवश्यकता।', 'season_hi': 'साल भर', 'water_hi': 'बहुत उच्च'},
            'banana': {'icon': '🍌', 'desc': 'High humidity and warm climate', 'season': 'Year-round', 'water': 'High', 'color': '#FFEB3B', 'desc_hi': 'उच्च आर्द्रता और गर्म जलवायु।', 'season_hi': 'साल भर', 'water_hi': 'उच्च'},
            'potato': {'icon': '🥔', 'desc': 'Cool climate, well-drained soil', 'season': 'Rabi', 'water': 'Medium', 'color': '#795548', 'desc_hi': 'ठंडी जलवायु, अच्छी तरह से जल निकासी वाली मिट्टी।', 'season_hi': 'रबी', 'water_hi': 'मध्यम'},
            'chickpea': {'icon': '🟤', 'desc': 'Drought-tolerant, cool season', 'season': 'Rabi', 'water': 'Low', 'color': '#9C27B0', 'desc_hi': 'सूखा-सहिष्णु, ठंडा मौसम।', 'season_hi': 'रबी', 'water_hi': 'कम'},
            'kidneybeans': {'icon': '🫘', 'desc': 'Moderate climate, well-drained soil', 'season': 'Kharif', 'water': 'Medium', 'color': '#E91E63', 'desc_hi': 'मध्यम जलवायु, अच्छी तरह से जल निकासी वाली मिट्टी।', 'season_hi': 'खरीफ', 'water_hi': 'मध्यम'},
            'lentil': {'icon': '🟫', 'desc': 'Cool season, low water requirement', 'season': 'Rabi', 'water': 'Low', 'color': '#607D8B', 'desc_hi': 'ठंडा मौसम, कम पानी की आवश्यकता।', 'season_hi': 'रबी', 'water_hi': 'कम'},
            'apple': {'icon': '🍎', 'desc': 'Cool climate, high altitude preferred', 'season': 'Perennial', 'water': 'Medium', 'color': '#F44336', 'desc_hi': 'ठंडी जलवायु, उच्च ऊंचाई पसंद करती है।', 'season_hi': 'बारहमासी', 'water_hi': 'मध्यम'},
            'mango': {'icon': '🥭', 'desc': 'Tropical climate, moderate water', 'season': 'Perennial', 'water': 'Medium', 'color': '#FF5722', 'desc_hi': 'उष्णकटिबंधीय जलवायु, मध्यम पानी।', 'season_hi': 'बारहमासी', 'water_hi': 'मध्यम'},
            'grapes': {'icon': '🍇', 'desc': 'Mediterranean climate preferred', 'season': 'Perennial', 'water': 'Medium', 'color': '#9C27B0', 'desc_hi': 'भूमध्यसागरीय जलवायु पसंद की जाती है।', 'season_hi': 'बारहमासी', 'water_hi': 'मध्यम'},
            'watermelon': {'icon': '🍉', 'desc': 'Hot climate, sandy soil preferred', 'season': 'Summer', 'water': 'High', 'color': '#4CAF50', 'desc_hi': 'गर्म जलवायु, रेतीली मिट्टी पसंद करती है।', 'season_hi': 'गर्मी', 'water_hi': 'उच्च'},
            'orange': {'icon': '🍊', 'desc': 'Subtropical climate, well-drained soil', 'season': 'Perennial', 'water': 'Medium', 'color': '#FF9800', 'desc_hi': 'उप-उष्णकटिबंधीय जलवायु, अच्छी तरह से जल निकासी वाली मिट्टी।', 'season_hi': 'बारहमासी', 'water_hi': 'मध्यम'},
            'coconut': {'icon': '🥥', 'desc': 'Tropical coastal climate', 'season': 'Perennial', 'water': 'High', 'color': '#795548', 'desc_hi': 'उष्णकटिबंधीय तटीय जलवायु।', 'season_hi': 'बारहमासी', 'water_hi': 'उच्च'},
            'coffee': {'icon': '☕', 'desc': 'High altitude, moderate temperature', 'season': 'Perennial', 'water': 'High', 'color': '#3E2723', 'desc_hi': 'उच्च ऊंचाई, मध्यम तापमान।', 'season_hi': 'बारहमासी', 'water_hi': 'उच्च'}
        }
        
        for i, idx in enumerate(top_3_indices):
            crop = crop_classes[idx]
            prob = probabilities[idx] * 100
            
            info = crop_info.get(crop, {'icon': '🌱', 'desc': 'Suitable for your soil and climate conditions', 'season': 'Variable', 'water': 'Medium', 'color': '#4CAF50', 'desc_hi': 'आपकी मिट्टी और जलवायु परिस्थितियों के लिए उपयुक्त', 'season_hi': 'परिवर्तनीय', 'water_hi': 'मध्यम'})
            
            with st.expander(f"**#{i+1} {info['icon']} {crop.upper()} - {prob:.1f}% Confidence**"):
                st.markdown(f"""
                <div class="crop-recommendation">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <div style="font-size: 3rem; margin-right: 1rem;">{info['icon']}</div>
                        <div>
                            <h3 style="margin: 0; color: {info['color']};">{crop.upper()}</h3>
                            <p style="margin: 0.5rem 0; color: #666; font-style: italic;">{info.get(f'desc_{lang}', info['desc'])}</p>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                            <strong>🗓️ {t['crop_prediction']['crop_season']}:</strong><br>{info.get(f'season_{lang}', info['season'])}
                        </div>
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                            <strong>💧 {t['crop_prediction']['crop_water']}:</strong><br>{info.get(f'water_{lang}', info['water'])}
                        </div>
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                            <strong>📊 {t['crop_prediction']['crop_match']}:</strong><br>{prob:.1f}%
                        </div>
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                            <strong>🎯 {t['crop_prediction']['crop_suitability']}:</strong><br>{match_quality}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown(f'<div class="section-header">{t["crop_prediction"]["personalized_tips_header"]}</div>', unsafe_allow_html=True)
        
        tip_col1, tip_col2 = st.columns(2)
        
        with tip_col1:
            st.markdown(f"#### {t['crop_prediction']['tips_climate_header']}")
            if temperature > 30:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #ff9800;">
                {t['crop_prediction']['tips_temp_high']}
                </div>
                """, unsafe_allow_html=True)
            elif temperature < 15:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #2196f3;">
                {t['crop_prediction']['tips_temp_low']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #4caf50;">
                {t['crop_prediction']['tips_temp_ok']}
                </div>
                """, unsafe_allow_html=True)
            
            if humidity > 70:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #ff9800;">
                {t['crop_prediction']['tips_hum_high']}
                </div>
                """, unsafe_allow_html=True)
            elif humidity < 30:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #ff9800;">
                {t['crop_prediction']['tips_hum_low']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #4caf50;">
                {t['crop_prediction']['tips_hum_ok']}
                </div>
                """, unsafe_allow_html=True)
        
        with tip_col2:
            st.markdown(f"#### {t['crop_prediction']['tips_soil_header']}")
            if ph < 6:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #f44336;">
                {t['crop_prediction']['tips_ph_acidic']}
                </div>
                """, unsafe_allow_html=True)
            elif ph > 8:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #f44336;">
                {t['crop_prediction']['tips_ph_alkaline']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #4caf50;">
                {t['crop_prediction']['tips_ph_ok']}
                </div>
                """, unsafe_allow_html=True)
            
            if nitrogen < 30:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #2196f3;">
                {t['crop_prediction']['tips_n_low']}
                </div>
                """, unsafe_allow_html=True)
            elif nitrogen > 100:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #ff9800;">
                {t['crop_prediction']['tips_n_high']}
                </div>
                """, unsafe_allow_html=True)
            
            if phosphorus < 20:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #2196f3;">
                {t['crop_prediction']['tips_p_low']}
                </div>
                """, unsafe_allow_html=True)
            
            if potassium < 30:
                st.markdown(f"""
                <div class="parameter-info" style="border-left-color: #2196f3;">
                {t['crop_prediction']['tips_k_low']}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        growth_potential = t['crop_prediction']['growth_high'] if confidence > 70 else t['crop_prediction']['growth_medium'] if confidence > 50 else t['crop_prediction']['growth_moderate']
        economic_viability = t['crop_prediction']['econ_prof'] if prediction.lower() in ['rice', 'wheat', 'cotton'] else t['crop_prediction']['econ_good']

        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 20px; margin: 2rem 0; animation: fadeInUp 1s ease-out;">
            <h3 style="color: white; margin-top: 0;">{t['crop_prediction']['summary_box_header']}</h3>
            <p style="font-size: 1.1rem; margin-bottom: 1.5rem;">
                {t['crop_prediction']['summary_box_text'].format(prediction.upper(), confidence)}
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 1.5rem;">
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; min-width: 150px;">
                    <strong>{t['crop_prediction']['summary_match_quality']}</strong><br>
                    {match_quality}
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; min-width: 150px;">
                    <strong>{t['crop_prediction']['summary_growth_potential']}</strong><br>
                    {growth_potential}
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; min-width: 150px;">
                    <strong>{t['crop_prediction']['summary_econ_viability']}</strong><br>
                    {economic_viability}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
def show_fertilizer_recommendation():
    """Display fertilizer recommendation page - MATCHING TRAINING PIPELINE"""
    t = translations[st.session_state.lang]

    st.markdown(f"# 🧪 {t['fertilizer_recommendation']['main_title']}")
    st.markdown(f"**{t['fertilizer_recommendation']['subtitle']}**")
    
    col1, col2 = st.columns([1, 1])
    
    # --- Input Fields ---
    with col1:
        st.markdown(f"### {t['fertilizer_recommendation']['section_info']}")
        
        crop_type = st.selectbox(
            t['fertilizer_recommendation']['crop_type_label'], 
            ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 
             'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts'], 
            key="crop_type"
        )
        soil_type = st.selectbox(
            t['fertilizer_recommendation']['soil_type_label'], 
            ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'], 
            key="soil_type"
        )
        
        st.markdown(f"### {t['fertilizer_recommendation']['section_env']}") 
        temp = st.slider(t['fertilizer_recommendation']['temp_label'], 10.0, 45.0, 25.0, key="fert_temp")
        humidity_fert = st.slider(t['fertilizer_recommendation']['hum_label'], 20.0, 95.0, 50.0, key="fert_humidity")
        moisture = st.slider(t['fertilizer_recommendation']['moisture_label'], 0.0, 100.0, 50.0, key="fert_moisture")
        
    with col2:
        st.markdown(f"### {t['fertilizer_recommendation']['section_nutrients']}")
        nitrogen_fert = st.slider(t['fertilizer_recommendation']['nitrogen_label'], 0, 50, 25, key="fert_nitrogen")
        phosphorus_fert = st.slider(t['fertilizer_recommendation']['phosphorus_label'], 0, 50, 25, key="fert_phosphorus")
        potassium_fert = st.slider(t['fertilizer_recommendation']['potassium_label'], 0, 50, 25, key="fert_potassium")
        
        # Nutrient Status Metrics
        st.markdown(f"### {t['fertilizer_recommendation']['nutrient_status_header']}")
        col2_1, col2_2, col2_3 = st.columns(3)
        
        def get_nutrient_status(value):
            if value < 15:
                return t['fertilizer_recommendation']['low']
            elif value < 35:
                return t['fertilizer_recommendation']['medium']
            else:
                return t['fertilizer_recommendation']['high']
        
        with col2_1:
            status = get_nutrient_status(nitrogen_fert)
            st.metric(t['fertilizer_recommendation']['nitrogen_label'], f"{nitrogen_fert}", status.split(' ')[-1]) 
        with col2_2:
            status = get_nutrient_status(phosphorus_fert)
            st.metric(t['fertilizer_recommendation']['phosphorus_label'], f"{phosphorus_fert}", status.split(' ')[-1])
        with col2_3:
            status = get_nutrient_status(potassium_fert)
            st.metric(t['fertilizer_recommendation']['potassium_label'], f"{potassium_fert}", status.split(' ')[-1])
            
    # Prediction Button
    if st.button(t['fertilizer_recommendation']['predict_button'], type="primary", key="get_fert_rec"):
        
        try:
            # Load the pipeline (which contains the preprocessor)
            pipeline_path = 'randpipe.pkl'
            with open(pipeline_path, 'rb') as f:
                randpipe = pickle.load(f)
            
            # Load the fertilizer label encoder
            encoder_path = 'fert_label_encoder.pkl'
            with open(encoder_path, 'rb') as f:
                le_fertilizer = pickle.load(f)
            
            # ✅ Create input DataFrame with EXACT column names and order from training
            # This is CRITICAL - must match training data exactly
            input_df = pd.DataFrame({
                'Temparature': [float(temp)],           # Note: Typo "Temparature" is intentional
                'Humidity': [float(humidity_fert)],
                'Moisture': [float(moisture)],
                'Soil Type': [soil_type],               # OneHotEncoder will handle this
                'Crop Type': [crop_type],               # OneHotEncoder will handle this
                'Nitrogen': [int(nitrogen_fert)],
                'Potassium': [int(potassium_fert)],
                'Phosphorous': [int(phosphorus_fert)]   # Note: American spelling
            })
            
            # ✅ The pipeline preprocessor (with OneHotEncoder) will handle categorical encoding
            # Make prediction - let the pipeline's preprocessor do the encoding
            prediction_encoded = randpipe.predict(input_df)[0]
            probabilities = randpipe.predict_proba(input_df)[0]
            
            # Decode the prediction using the fertilizer label encoder
            prediction = le_fertilizer.inverse_transform([int(prediction_encoded)])[0]
            confidence = probabilities[int(prediction_encoded)] * 100
            
            # Display Results
            st.markdown(f"""
            <div class="prediction-result">
                🎯 {t['fertilizer_recommendation']['result_header']} <strong>{prediction}</strong><br>
                📊 {t['fertilizer_recommendation']['result_confidence']} <strong>{confidence:.1f}%</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Display usage tips
            st.success(f"""
**✅ Fertilizer Recommendation for {crop_type}**

**Recommended Fertilizer:** {prediction}

**Applied to:** {soil_type} soil

**Tips:**
- Consider current nutrient levels when determining quantity
- Apply during the appropriate growth stage
- Monitor soil moisture and weather conditions
- Retest soil after 2-3 months for optimal results
            """)
            
            # Show confidence level
            if confidence > 80:
                st.info(f"🎯 **High Confidence** ({confidence:.1f}%) - This recommendation is well-matched to your conditions")
            elif confidence > 60:
                st.info(f"📊 **Good Confidence** ({confidence:.1f}%) - This recommendation should work well")
            else:
                st.warning(f"⚠️ **Moderate Confidence** ({confidence:.1f}%) - Consider consulting with an agronomist")
            
        except FileNotFoundError as e:
            st.error(f"❌ Model file not found!")
            st.warning("""
            Please ensure these files are in the app directory:
            - `randpipe.pkl` (trained pipeline with preprocessor)
            - `fert_label_encoder.pkl` (fertilizer encoder)
            """)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {type(e).__name__}")
            st.write(f"**Details:** {str(e)}")
            
            # Debug info
            with st.expander("Debug Information"):
                st.write("Input DataFrame:")
                st.write(input_df)
                st.write(f"Input columns: {list(input_df.columns)}")
                import traceback
                st.write(traceback.format_exc())


# C:\Users\user\OneDrive\Desktop\new\ferti.py

def predict_disease_from_image(image):
    """Enhanced image preprocessing and prediction"""
    cnn_model, class_labels = load_cnn_model_and_labels()
    t = translations.get(st.session_state.get('lang', 'en'), translations['en'])
    
    if cnn_model is None:
        st.warning("⚠️ Using fallback prediction mode")
        return "Healthy", 95.0
    
    try:
        import tensorflow as tf
        
        # TARGET SIZE - Match your training size
        target_size = (224, 224)  # Adjust if your model uses different size
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        processed_image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to array
        img_array = np.array(processed_image)
        
        # CRITICAL: Match training preprocessing
        # Option 1: If trained with ImageDataGenerator with rescale=1./255
        img_array = img_array.astype('float32') / 255.0
        
        # Option 2: If trained with tf.keras.applications preprocessing
        # from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        # img_array = preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Ensure correct dtype (important for mixed precision)
        img_array = tf.cast(img_array, tf.float32)
        
        # Make prediction
        predictions = cnn_model.predict(img_array, verbose=0)
        
        # Get results
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)
        
        # Get disease name
        predicted_disease_en = class_labels.get(predicted_index, "Unknown Disease")
        
        # Clean disease name
        if "___" in predicted_disease_en:
            plant_name, disease_name = predicted_disease_en.split("___")
            predicted_disease_en = disease_name.replace("_", " ").title()
        
        # Get translation
        disease_names = t.get('disease_detection', {}).get('disease_names', {})
        translated_disease = disease_names.get(predicted_disease_en, predicted_disease_en)
        
        return translated_disease, confidence
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return "Error", 0.0
    

    2
def mock_gemini_api_call(user_prompt, system_prompt, model_name="gemini-2.5-flash-preview-09-2025"):
    """
    Mocks the API call to the Gemini generateContent endpoint.
    In a real environment, replace this with actual HTTP logic using libraries like 'requests' or 'httpx'.
    """
    
    # 1. Configuration (Set API Key and Endpoint)
    # The API key is assumed to be handled by the execution environment or retrieved securely.
    api_key = "AIzaSyAr9UV0NfkoEkuoo8I74rXABuWIaDmP9V8" # Leave empty as instructed.
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    # 2. Payload Construction
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "tools": [{"google_search": {}}],  # Enable Google Search Grounding for current information
        "systemInstruction": {
            "parts": [{
                "text": system_prompt
            }]
        },
    }

    # --- Simulated API Response ---
    # For demonstration, we simulate the structure of a successful API response.
    time.sleep(2) # Simulate network latency

    if "rice irrigation" in user_prompt.lower():
        mock_text = "Rice requires significant water, usually flooded fields (paddy system) from planting until about two weeks before harvest. The water depth should generally be maintained at 5-10 cm to suppress weeds and regulate temperature. Use the System of Rice Intensification (SRI) method for better yield and water efficiency, which involves intermittent wetting and drying."
        sources = [
            {"uri": "https://source.example/rice-farming", "title": "Rice Farming Techniques"},
            {"uri": "https://source.example/sri-method", "title": "SRI Method Explained"}
        ]
    elif "soil ph" in user_prompt.lower():
        mock_text = "Soil pH measures acidity or alkalinity. Most crops prefer a neutral range (6.0 to 7.5). If your soil is too acidic (below 6.0), you can add agricultural lime to raise the pH. If it's too alkaline (above 7.5), sulfur or gypsum can be added to lower it."
        sources = [{"uri": "https://source.example/soil-ph", "title": "Soil pH Management"}]
    else:
        # Fallback response simulating general LLM output
        mock_text = f"I am ready to help you research {user_prompt.lower()}. Generally, ensure you follow local agricultural guidelines for the best results."
        sources = []


    # 3. Process Response
    
    # Format citations if they exist
    citation_text = ""
    if sources:
        citation_text = "\n\n**Sources:**"
        for i, source in enumerate(sources):
            citation_text += f'\n[{i+1}] [{source["title"]}]({source["uri"]})'
    
    return mock_text + citation_text

# --- Gemini API Call Wrapper ---

def get_ai_response(prompt):
    """
    Function to get AI response from Gemini API with Google Search grounding
    """
    t = translations.get(st.session_state.get('lang', 'en'), translations['en'])
    
    # Get API key from Streamlit secrets or environment variable
    # For production, use: st.secrets["GEMINI_API_KEY"]
    api_key = "AIzaSyAr9UV0NfkoEkuoo8I74rXABuWIaDmP9V8"  # Replace with secure method
    
    if not api_key:
        return t['chatbot_page'].get('error', "API key not configured. Please contact support.")
    
    model_name = "gemini-2.0-flash-exp"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    # System prompt for agricultural assistant
    system_prompt = (
    "You are DeepChat, an AI assistant for the DeepAgro smart agriculture platform. "
    "Your PRIMARY expertise is agriculture, farming, crop science, soil health, "
    "pest management, irrigation, and yield optimization. "
    "\n\n**CORE DIRECTIVES:**"
    "\n\n1. **Detail and Depth:** When users ask agriculture-related questions, provide "
    "**detailed, comprehensive, and practical advice**. Answers should be thorough, "
    "well-structured, and written in simple, actionable language for farmers."
    "\n\n2. **Language Flexibility:** You are capable of communicating and providing "
    "detailed answers in **any language the user requests**. If the user's initial "
    "message is in a specific language, respond in that language unless otherwise instructed."
    "\n\n3. **Source Integration (Knowledge Requirement):** To ensure accuracy and "
    "provide detailed context, you are equipped with and **must use a search tool (Google)** "
    "to find and integrate the latest, most reliable, and detailed information into "
    "your agricultural advice. Always provide a synthesized, practical answer, not just a list of sources."
    "\n\n4. **Scope Management:** You can answer general questions, but always try "
    "to relate answers back to agriculture when relevant. If a question is completely "
    "unrelated to farming, provide a brief, helpful answer but gently remind users "
    "that you specialize in agricultural assistance."
    "\n\nBe friendly, professional, and concise."
)
    
    # Construct payload
    payload = {
        "contents": [{
            "parts": [{"text": f"{system_prompt}\n\nUser Question: {prompt}"}]
        }],
        "tools": [{"google_search": {}}],  # Enable Google Search for current information
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }
    
    try:
        # Make API request
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            
            # Extract text content
            if "content" in candidate and "parts" in candidate["content"]:
                response_text = candidate["content"]["parts"][0].get("text", "")
                
                # Extract grounding metadata (sources) if available
                citation_text = ""
                if "groundingMetadata" in candidate:
                    grounding = candidate["groundingMetadata"]
                    if "groundingChunks" in grounding and grounding["groundingChunks"]:
                        citation_text = "\n\n**Sources:**"
                        for i, chunk in enumerate(grounding["groundingChunks"][:3]):  # Limit to 3 sources
                            if "web" in chunk:
                                web_info = chunk["web"]
                                title = web_info.get("title", "Source")
                                uri = web_info.get("uri", "#")
                                citation_text += f'\n[{i+1}] [{title}]({uri})'
                
                return response_text + citation_text
            
        # Fallback if no valid response
        return t['chatbot_page'].get('error', "I couldn't generate a response. Please try rephrasing your question.")
        
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        error_msg = t['chatbot_page'].get('error', "Sorry, I encountered an error.")
        return f"{error_msg} (Network Error)"
    except json.JSONDecodeError:
        return "Invalid response from server. Please try again."
    except Exception as e:
        error_msg = t['chatbot_page'].get('error', "Sorry, I encountered an error.")
        return f"{error_msg} ({type(e).__name__})"


def show_chatbot_page():
    """Display chatbot interface"""
    t = translations[st.session_state.lang]

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": t['chatbot_page']['initial_message']}
        ]

    st.markdown(f'<h1 class="main-title">{t["chatbot_page"]["main_title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{t["chatbot_page"]["subtitle"]}</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input(t['chatbot_page']['user_placeholder']):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner(t['chatbot_page']['loading']):
                response = get_ai_response(prompt)
                st.markdown(response)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})
def show_disease_detection():
    t = translations[st.session_state.lang]
    
    st.markdown(f"# 🔬 {t['disease_detection']['main_title']}")
    st.markdown(f"{t['disease_detection']['subtitle']}")
    
    st.markdown(f"""
    <div class="upload-section">
        <h4>{t['disease_detection']['upload_guidelines_title']}</h4>
        {t['disease_detection']['upload_guidelines_text']}
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        t['disease_detection']['file_uploader_label'],
        type=['jpg', 'jpeg', 'png'],
        help=t['disease_detection']['file_uploader_help']
    )
    
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file)
            
            # Display original image
            st.image(image, caption=t['disease_detection']['uploaded_image_caption'], 
                    use_container_width=True)
            
            # Analyze button
            if st.button(t['disease_detection']['analyze_button'], 
                        type="primary", use_container_width=True):
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("📷 Processing image...")
                    elif i < 60:
                        status_text.text("🧠 Analyzing with AI...")
                    else:
                        status_text.text("🔍 Identifying disease...")
                
                progress_bar.empty()
                status_text.empty()
                
                # Get prediction
                predicted_disease, confidence = predict_disease_from_image(image)
                
                # Display results with enhanced styling
                if predicted_disease.lower() != "error":
                    st.success(t['disease_detection']['analysis_complete'])
                    
                    # Color-coded result based on confidence
                    color = "#4CAF50" if confidence > 80 else "#FF9800" if confidence > 60 else "#F44336"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}, {color}dd); 
                                padding: 2rem; border-radius: 15px; color: white; 
                                text-align: center; margin: 1rem 0;">
                        <h2 style="color: white; margin: 0;">
                            🎯 {t['disease_detection']['result_header']}
                        </h2>
                        <h1 style="color: white; margin: 1rem 0; font-size: 2.5rem;">
                            {predicted_disease}
                        </h1>
                        <p style="color: white; font-size: 1.3rem; margin: 0;">
                            📊 {t['disease_detection']['result_confidence']} {confidence:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show appropriate message
                    if "healthy" in predicted_disease.lower():
                        st.balloons()
                        st.success(t['disease_detection']['healthy_message'])
                    else:
                        st.warning(t['disease_detection']['disease_warning'])
                        
                        # Add recommendations section
                        with st.expander("💡 Recommended Actions"):
                            st.markdown("""
                            - Isolate affected plants
                            - Remove infected leaves
                            - Apply appropriate fungicide/pesticide
                            - Improve air circulation
                            - Monitor other plants
                            - Consult agricultural expert for severe cases
                            """)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("Debug Info"):
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("📤 Please upload a leaf image to start analysis.")

def show_about_page():
    """Display about page with team information"""
    t = translations[st.session_state.lang]

    st.markdown(f"# {t['about_page']['main_title']}")
    st.markdown(f"**{t['about_page']['subtitle']}**")
    
    st.markdown(f"""
    <div class="feature-card">
        <h4>{t['about_page']['mission_title']}</h4>
        <p>{t['about_page']['mission_text']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0;">
        <h2 style="color: #1b5e20; font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">{t['about_page']['team_header']}</h2>
        <p style="color: #2e7d32; font-size: 1.2rem; max-width: 800px; margin: 0 auto; line-height: 1.8;">
            {t['about_page']['team_desc']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    team_members = [
        {"name": "Aditya Upendra Gupta", "roll": "AD24B1003", "branch": "Artificial Intelligence (AI) and Data Science (DS)"},
        {"name": "Aaditya Awasthi", "roll": "CS24B1001", "branch": "Computer Science and Engineering"},
        {"name": "Aditya Raj", "roll": "CS24B1004", "branch": "Computer Science and Engineering"},
        {"name": "Sudhavalli Murali", "roll": "CS24B1057", "branch": "Computer Science and Engineering"},
        {"name": "Kinshu Keshri", "roll": "AD24B1034", "branch": "Artificial Intelligence (AI) and Data Science (DS)"},
        {"name": "Rishita", "roll": "CS24B1021", "branch": "Computer Science and Engineering"}
    ]
    
    cols = st.columns(3)
    for i, member in enumerate(team_members):
        col_index = i % 3
        with cols[col_index]:
            st.markdown(f"""
            <div class="team-card">
                <div style="font-size: 3rem; margin-bottom: 1rem;">👨‍🎓</div>
                <h3>{member['name']}</h3>
                <p class="roll-number">{member['roll']}</p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8;">{member['branch']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(f"## {t['about_page']['tech_stack_header']}")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown(f"""
        <div class="feature-card">
            <h4>{t['about_page']['ml_title']}</h4>
            <p>{t['about_page']['ml_text']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col2:
        st.markdown(f"""
        <div class="feature-card">
            <h4>{t['about_page']['web_title']}</h4>
            <p>{t['about_page']['web_text']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col3:
        st.markdown(f"""
        <div class="feature-card">
            <h4>{t['about_page']['data_title']}</h4>
            <p>{t['about_page']['data_text']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"## {t['about_page']['features_header']}")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown(f"""
        ### {t['about_page']['smart_pred_header']}
        {t['about_page']['smart_pred_list']}
        """)
    
    with feature_col2:
        st.markdown(f"""
        ### {t['about_page']['ux_header']}
        {t['about_page']['ux_list']}
        """)
    
    st.markdown("---")
    
    contact_col1, contact_col2 = st.columns(2)
    
    with contact_col1:
        st.markdown(f"""
        <div class="feature-card">
            <h4>{t['about_page']['institution_title']}</h4>
            <p>{t['about_page']['institution_text']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with contact_col2:
        st.markdown(f"""
        <div class="feature-card">
            <h4>{t['about_page']['acknowledgements_title']}</h4>
            <p>{t['about_page']['acknowledgements_text']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #e8f5e8, #f1f8e9); border-radius: 15px; margin: 2rem 0;">
        <p style="margin: 0; color: #1b5e20; font-size: 1.1rem; font-weight: 600;">
            {t['about_page']['footer_title']} - {t['about_page']['footer_slogan']}
        </p>
        <p style="margin: 0.5rem 0 0 0; color: #2e7d32; font-size: 0.95rem;">
            {t['about_page']['footer_credit']}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Call the main function to run the app
if __name__ == "__main__":
    main() 