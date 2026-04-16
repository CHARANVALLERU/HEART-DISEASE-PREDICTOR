"""
Heart Disease Prediction - Streamlit Web Application
=====================================================
Interactive web app for predicting heart disease risk using trained ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 { color: white; font-size: 2.5rem; margin-bottom: 0.5rem; }
    .main-header p { color: rgba(255,255,255,0.9); font-size: 1.1rem; }
    
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .prediction-positive {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    
    .prediction-negative {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .info-box {
        background: #f0f4ff;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .feature-info {
        background: #fff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# LOAD MODEL & SCALER
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    """Load pre-trained model and scaler."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'models', 'best_model.pkl')
    scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None


def train_quick_model():
    """Train a quick model if no saved model exists."""
    from data_preprocessing import load_data, preprocess_data
    from sklearn.linear_model import LogisticRegression
    
    df = load_data()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(
        df, save_scaler=True, models_dir=models_dir
    )
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'best_model.pkl'))
    
    return model, scaler


# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>AI-powered health risk assessment using Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# SIDEBAR - INPUT FORM
# ──────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏥 Patient Information")
st.sidebar.markdown("---")

# Age
age = st.sidebar.slider("🎂 Age (years)", min_value=20, max_value=80, value=45, step=1)

# Sex
sex = st.sidebar.selectbox("👤 Sex", options=["Male", "Female"])
sex_val = 1 if sex == "Male" else 0

# Chest Pain Type
cp_options = {
    "Typical Angina (0)": 0,
    "Atypical Angina (1)": 1,
    "Non-Anginal Pain (2)": 2,
    "Asymptomatic (3)": 3
}
cp = st.sidebar.selectbox("💔 Chest Pain Type", options=list(cp_options.keys()))
cp_val = cp_options[cp]

# Resting Blood Pressure
trestbps = st.sidebar.slider("🩺 Resting Blood Pressure (mm Hg)", 
                              min_value=80, max_value=200, value=120, step=1)

# Cholesterol
chol = st.sidebar.slider("🧪 Serum Cholesterol (mg/dl)", 
                          min_value=100, max_value=600, value=200, step=1)

# Fasting Blood Sugar
fbs_option = st.sidebar.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl?", 
                                   options=["No", "Yes"])
fbs_val = 1 if fbs_option == "Yes" else 0

# Resting ECG
restecg_options = {
    "Normal (0)": 0,
    "ST-T Wave Abnormality (1)": 1,
    "Left Ventricular Hypertrophy (2)": 2
}
restecg = st.sidebar.selectbox("📊 Resting ECG Results", options=list(restecg_options.keys()))
restecg_val = restecg_options[restecg]

# Max Heart Rate
thalach = st.sidebar.slider("💓 Maximum Heart Rate Achieved", 
                             min_value=60, max_value=220, value=150, step=1)

# Exercise Induced Angina
exang_option = st.sidebar.selectbox("🏃 Exercise Induced Angina?", options=["No", "Yes"])
exang_val = 1 if exang_option == "Yes" else 0

# Oldpeak
oldpeak = st.sidebar.slider("📉 ST Depression (Oldpeak)", 
                             min_value=0.0, max_value=7.0, value=1.0, step=0.1)

# Slope
slope_options = {
    "Upsloping (0)": 0,
    "Flat (1)": 1,
    "Downsloping (2)": 2
}
slope = st.sidebar.selectbox("📐 Slope of Peak Exercise ST Segment", 
                              options=list(slope_options.keys()))
slope_val = slope_options[slope]

# CA
ca = st.sidebar.selectbox("🔬 Number of Major Vessels (0-3)", options=[0, 1, 2, 3])

# Thal
thal_options = {
    "Normal (0)": 0,
    "Fixed Defect (1)": 1,
    "Reversible Defect (2)": 2,
    "Thalassemia Defect (3)": 3
}
thal = st.sidebar.selectbox("🧬 Thalassemia", options=list(thal_options.keys()))
thal_val = thal_options[thal]

st.sidebar.markdown("---")


# ──────────────────────────────────────────────────────────────
# MAIN CONTENT - TABS
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Data Insights", "📋 About", "📖 Feature Guide"])


# ──────────────────────────────────────────────────────────────
# TAB 1: PREDICTION
# ──────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Patient Summary")
        
        patient_data = {
            "Parameter": ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
                          "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate",
                          "Exercise Angina", "ST Depression", "Slope", "Major Vessels", "Thalassemia"],
            "Value": [str(age), sex, cp.split("(")[0].strip(), f"{trestbps} mm Hg", 
                      f"{chol} mg/dl", fbs_option, restecg.split("(")[0].strip(),
                      f"{thalach} bpm", exang_option, str(oldpeak), 
                      slope.split("(")[0].strip(), str(ca), thal.split("(")[0].strip()]
        }
        st.dataframe(pd.DataFrame(patient_data), width='stretch', hide_index=True)
    
    with col2:
        st.markdown("### 🔮 Prediction Result")
        
        # Predict button
        if st.button("🫀 Predict Heart Disease Risk", width='stretch'):
            # Load or train model
            model, scaler = load_model_and_scaler()
            if model is None:
                with st.spinner("🔧 Training model for first time..."):
                    model, scaler = train_quick_model()
            
            # Prepare input
            input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, 
                                    restecg_val, thalach, exang_val, oldpeak, 
                                    slope_val, ca, thal_val]])
            
            # Scale
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_scaled)[0]
                prob_no_disease = probability[0] * 100
                prob_disease = probability[1] * 100
            else:
                prob_disease = 100 if prediction == 1 else 0
                prob_no_disease = 100 - prob_disease
            
            # Display result
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-card prediction-positive">
                    <h2>⚠️ HIGH RISK DETECTED</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{prob_disease:.1f}%</h1>
                    <p style="font-size: 1.2rem;">Probability of Heart Disease</p>
                    <p style="margin-top: 1rem;">⚕️ Please consult a cardiologist immediately.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card prediction-negative">
                    <h2>✅ LOW RISK</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{prob_no_disease:.1f}%</h1>
                    <p style="font-size: 1.2rem;">Probability of No Heart Disease</p>
                    <p style="margin-top: 1rem;">🎉 Your heart appears to be healthy!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_disease,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Heart Disease Risk Score", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#e74c3c" if prediction == 1 else "#2ecc71"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'steps': [
                        {'range': [0, 30], 'color': '#d5f5e3'},
                        {'range': [30, 60], 'color': '#fdebd0'},
                        {'range': [60, 100], 'color': '#fadbd8'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, width='stretch')
            
            # Health tips
            st.markdown("---")
            st.markdown("### 💡 Health Recommendations")
            if prediction == 1:
                st.error("⚠️ **Important:** This is a screening tool, NOT a diagnosis. Please consult a healthcare professional.")
                st.markdown("""
                - 🏥 **Schedule a cardiology appointment** as soon as possible
                - 🩺 **Get a comprehensive heart check-up** including ECG, Echo, and stress test
                - 🥗 **Adopt a heart-healthy diet** - low sodium, low saturated fats
                - 🏃 **Regular moderate exercise** as recommended by your doctor
                - 🚭 **Quit smoking** and limit alcohol consumption
                - 😊 **Manage stress** through meditation or relaxation techniques
                """)
            else:
                st.success("✅ **Good news!** Your risk appears low, but regular check-ups are still important.")
                st.markdown("""
                - ✅ **Continue regular health check-ups** annually
                - 🥗 **Maintain a balanced diet** rich in fruits and vegetables
                - 🏃 **Stay physically active** - aim for 150 min/week
                - 😴 **Get adequate sleep** - 7-9 hours per night
                - 💧 **Stay hydrated** and manage stress levels
                """)
        else:
            st.markdown("""
            <div class="info-box">
                <h4>👈 Enter patient parameters in the sidebar</h4>
                <p>Fill in all the health parameters and click the <strong>Predict</strong> button to get the risk assessment.</p>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TAB 2: DATA INSIGHTS
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Dataset Overview & Insights")
    
    from data_preprocessing import load_data
    df = load_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", df.shape[0])
    with col2:
        st.metric("Heart Disease Cases", int(df['target'].sum()))
    with col3:
        st.metric("Healthy Cases", int((df['target'] == 0).sum()))
    with col4:
        st.metric("Disease Rate", f"{df['target'].mean()*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Target distribution
        fig = px.pie(df, names=df['target'].map({0: 'No Disease', 1: 'Heart Disease'}),
                     title="Heart Disease Distribution",
                     color_discrete_sequence=['#2ecc71', '#e74c3c'],
                     hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Age distribution
        fig = px.histogram(df, x='age', color=df['target'].map({0: 'No Disease', 1: 'Heart Disease'}),
                          title="Age Distribution by Disease Status",
                          color_discrete_sequence=['#2ecc71', '#e74c3c'],
                          barmode='overlay', opacity=0.7)
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        corr = df.corr()
        fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                       title="Feature Correlation Matrix",
                       color_continuous_scale='RdYlBu_r')
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Feature box plots
        feature = st.selectbox("Select Feature to Explore", 
                               ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
        fig = px.box(df, x=df['target'].map({0: 'No Disease', 1: 'Heart Disease'}),
                    y=feature, color=df['target'].map({0: 'No Disease', 1: 'Heart Disease'}),
                    title=f"{feature.upper()} by Disease Status",
                    color_discrete_sequence=['#2ecc71', '#e74c3c'])
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    # Raw data
    with st.expander("📄 View Raw Dataset"):
        st.dataframe(df, width='stretch', height=400)
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")


# ──────────────────────────────────────────────────────────────
# TAB 3: ABOUT
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    ### 📖 About This Project
    
    **Heart Disease Prediction System** uses machine learning to predict the likelihood 
    of heart disease based on clinical parameters. This tool is designed as a screening 
    aid and should NOT replace professional medical diagnosis.
    
    ---
    
    ### 🧠 Machine Learning Models Used
    
    | Model | Description |
    |-------|-------------|
    | **Logistic Regression** | Linear model for binary classification |
    | **Random Forest** | Ensemble of decision trees |
    | **SVM** | Support Vector Machine with RBF kernel |
    | **KNN** | K-Nearest Neighbors classifier |
    | **Decision Tree** | Tree-based classification |
    | **Gradient Boosting** | Sequential ensemble method |
    
    ---
    
    ### 📊 Dataset Information
    
    The **UCI Heart Disease Dataset** (Cleveland) contains 303 patient records 
    with 14 clinical attributes including age, sex, blood pressure, cholesterol, 
    and various cardiac test results.
    
    **Source:** UCI Machine Learning Repository
    
    **Reference:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). 
    Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
    
    ---
    
    ### ⚠️ Disclaimer
    
    This application is for **educational and demonstration purposes only**. 
    It should not be used as a substitute for professional medical advice, 
    diagnosis, or treatment. Always seek the advice of a qualified healthcare 
    provider with any questions you may have regarding a medical condition.
    """)


# ──────────────────────────────────────────────────────────────
# TAB 4: FEATURE GUIDE
# ──────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 📖 Feature Guide - Understanding the Input Parameters")
    st.markdown("---")
    
    features_info = [
        ("🎂 Age", "The age of the patient in years. Heart disease risk generally increases with age."),
        ("👤 Sex", "Biological sex of the patient. Males typically have higher risk at younger ages."),
        ("💔 Chest Pain Type (cp)", """
        - **Typical Angina (0):** Chest pain related to decreased blood supply to the heart
        - **Atypical Angina (1):** Chest pain not related to heart
        - **Non-Anginal Pain (2):** Not angina-related pain
        - **Asymptomatic (3):** No chest pain
        """),
        ("🩺 Resting Blood Pressure (trestbps)", "Blood pressure (mm Hg) measured at rest. Normal is typically below 120/80 mm Hg."),
        ("🧪 Serum Cholesterol (chol)", "Total cholesterol in mg/dl. Desirable level is less than 200 mg/dl."),
        ("🍬 Fasting Blood Sugar (fbs)", "Whether fasting blood sugar is greater than 120 mg/dl. Elevated levels may indicate diabetes."),
        ("📊 Resting ECG (restecg)", """
        - **Normal (0):** Normal ECG reading
        - **ST-T Wave Abnormality (1):** May indicate non-specific changes
        - **Left Ventricular Hypertrophy (2):** Enlarged heart chamber
        """),
        ("💓 Max Heart Rate (thalach)", "Maximum heart rate achieved during exercise. Generally decreases with age."),
        ("🏃 Exercise Induced Angina (exang)", "Whether exercise causes chest pain (angina). This is a significant indicator."),
        ("📉 ST Depression - Oldpeak", "ST segment depression induced by exercise relative to rest. Higher values may indicate heart issues."),
        ("📐 Slope", """
        - **Upsloping (0):** Better heart rate response
        - **Flat (1):** Minimal change - potential concern
        - **Downsloping (2):** Signs of unhealthy heart
        """),
        ("🔬 Major Vessels (ca)", "Number of major blood vessels (0-3) colored by fluoroscopy. More vessels colored may indicate heart disease."),
        ("🧬 Thalassemia (thal)", """
        - **Normal (0):** Normal blood flow
        - **Fixed Defect (1):** No blood flow in some part
        - **Reversible Defect (2):** Blood flow observed but abnormal
        - **Defect (3):** Blood disorder present
        """),
    ]
    
    for title, description in features_info:
        with st.expander(title):
            st.markdown(description)


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>❤️ Heart Disease Prediction System | Built with Python & Streamlit</p>
    <p style="font-size: 0.8rem;">Powered by Machine Learning | UCI Heart Disease Dataset</p>
</div>
""", unsafe_allow_html=True)
