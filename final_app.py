import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle

# Import plotly with error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly is not installed. Please run 'pip install plotly' in your terminal.")

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Disease Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project name at the top of the page
st.markdown('<h1 style="text-align:center; color:Orange; font-size:3.5rem; margin-bottom:0.5rem;">Disease Risk Prediction System</h1>', unsafe_allow_html=True)

# Custom CSS for attractive styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .safe-prediction {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .danger-prediction {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .info-box {
        background: rgba(52, 152, 219, 0.1);
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the heart disease dataset"""
    try:
        df = pd.read_csv('heart.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Heart disease dataset not found. Please ensure 'heart.csv' is in the same directory.")
        return None

@st.cache_resource
def train_models(df):
    """Train multiple models and cache the results"""
    if df is None:
        return None, None, None, None, None, None, None  # Always return 7 values
    
    # Data preprocessing
    categorical_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'resting_ecg', 'exercise_angina', 'st_slope']
    numerical_cols = ['age', 'resting_bp_s', 'cholesterol', 'max_heart_rate', 'oldpeak']
    
    # Handle missing values
    df['resting_bp_s'] = df['resting_bp_s'].replace(0, df['resting_bp_s'].mean())
    df['cholesterol'] = df['cholesterol'].replace(0, df['cholesterol'].mean())
    df['oldpeak'] = df['oldpeak'].clip(lower=0)
    
    # One-hot encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Scaling
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_cols])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=pd.Index(numerical_cols))
    scaled_df = pd.concat([scaled_numerical_df, encoded_df, df['target']], axis=1)
    
    # Feature engineering
    scaled_df['age_chol_interaction'] = scaled_df['age'] * scaled_df['cholesterol']
    scaled_df['sex_heart_rate_interaction'] = scaled_df['sex_1.0'] * scaled_df['max_heart_rate']
    
    # Prepare features and target
    X = scaled_df.drop('target', axis=1)
    y = scaled_df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    model_metrics = {}
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'training_time': training_time
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
        trained_models[name] = model
        model_metrics[name] = metrics
    
    return trained_models, model_metrics, encoder, scaler, X.columns, X_test, y_test

def preprocess_input(input_data, encoder, scaler, feature_columns):
    """Preprocess user input for prediction"""
    # Create DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Categorical and numerical columns
    categorical_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'resting_ecg', 'exercise_angina', 'st_slope']
    numerical_cols = ['age', 'resting_bp_s', 'cholesterol', 'max_heart_rate', 'oldpeak']
    
    # Encode categorical features
    encoded_features = encoder.transform(df_input[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Scale numerical features
    scaled_numerical = scaler.transform(df_input[numerical_cols])
    scaled_df = pd.DataFrame(scaled_numerical, columns=pd.Index(numerical_cols))
    
    # Combine features
    processed_df = pd.concat([scaled_df, encoded_df], axis=1)
    
    # Add interaction features
    processed_df['age_chol_interaction'] = processed_df['age'] * processed_df['cholesterol']
    processed_df['sex_heart_rate_interaction'] = processed_df['sex_1.0'] * processed_df['max_heart_rate']
    
    # Ensure column order matches training data
    processed_df = processed_df[feature_columns]
    
    return processed_df

    def debug_dataset_structure(df):
        """Debug function to show dataset structure"""
        if df is None:
            st.error("❌ No dataset loaded")
            return
        
        st.markdown("### 🔍 Dataset Structure Debug")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Info:**")
            st.write(f"- Shape: {df.shape}")
            st.write(f"- Columns: {list(df.columns)}")
            st.write(f"- Data types: {df.dtypes.to_dict()}")
            
            if 'Diagnosis' in df.columns:
                st.write(f"- Diagnosis distribution: {df['Diagnosis'].value_counts().to_dict()}")
            else:
                st.write("- ❌ No 'Diagnosis' column found")
        
        with col2:
            st.markdown("**Sample Data:**")
            st.dataframe(df.head(), use_container_width=True)
        
        st.markdown("**Missing Values:**")
        missing_data = df.isnull().sum()
        st.dataframe(missing_data[missing_data > 0], use_container_width=True)

    def main():
        # Main header
        st.markdown('<h1 class="main-header">🦟 Dengue Fever Prediction System</h1>', unsafe_allow_html=True)
    
        # Load data
        df = load_data()
        if df is None:
            return
        
        # Debug mode - uncomment to see dataset structure
        # debug_dataset_structure(df)
        
        # Sidebar
        st.sidebar.markdown("## 🎯 Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["🏠 Home", "📊 Model Performance", "🔮 Make Prediction", "📈 Data Analysis"]
        )
        
        # Train models (cached)
        trained_models, model_metrics, scaler, feature_columns, X_test, y_test = train_models(df)
        
        if page == "🏠 Home":
            show_home_page(df)
        
        elif page == "📊 Model Performance":
            show_model_performance(model_metrics, trained_models, X_test, y_test)
        
        elif page == "🔮 Make Prediction":
            show_prediction_page(trained_models, scaler, feature_columns, model_metrics)
        
        elif page == "📈 Data Analysis":
            show_data_analysis(df)

def show_home_page(df):
    """Display the home page with overview information"""
    st.markdown('<h2 class="sub-header">Welcome to the Heart Disease Prediction System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 About This System
        
        This advanced heart disease prediction system uses machine learning to analyze patient symptoms 
        and provide accurate predictions about the likelihood of heart disease. Our system incorporates 
        multiple state-of-the-art algorithms to ensure the highest accuracy possible.
        
        ### 🔬 Key Features
        
        - **Multiple ML Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM
        - **Real-time Predictions**: Get instant predictions with confidence scores
        - **Performance Metrics**: Detailed accuracy, precision, recall, and F1-score analysis
        - **Interactive Interface**: User-friendly design for easy symptom input
        - **Data Visualization**: Comprehensive analysis of the dataset
        
        ### 📋 Required Symptoms
        
        The system analyzes the following key symptoms and measurements:
        """)
        
        symptoms = [
            "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure",
            "Cholesterol Level", "Fasting Blood Sugar", "Resting ECG Results",
            "Maximum Heart Rate", "Exercise-Induced Angina", "ST Depression",
            "ST Slope"
        ]
        
        for i, symptom in enumerate(symptoms, 1):
            st.markdown(f"**{i}.** {symptom}")
    
    with col2:
        st.markdown("### 📊 Dataset Overview")
        
        # Dataset statistics
        total_patients = len(df)
        heart_disease_cases = df['target'].sum()
        healthy_cases = total_patients - heart_disease_cases
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Patients</h3>
            <h2>{total_patients:,}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Heart Disease Cases</h3>
            <h2>{heart_disease_cases:,}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Healthy Cases</h3>
            <h2>{healthy_cases:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Navigate to 'Make Prediction'** to input patient symptoms
    2. **Check 'Model Performance'** to see accuracy metrics
    3. **Explore 'Data Analysis'** to understand the dataset
    """)

def show_model_performance(model_metrics, trained_models, X_test, y_test):
    """Display model performance metrics"""
    st.markdown('<h2 class="sub-header">📊 Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    if model_metrics is None:
        st.error("❌ Model metrics not available. Please ensure models are trained.")
        return
    
    # Performance comparison
    st.markdown("### 🏆 All Models Comparison")
    
    # Create performance DataFrame
    perf_df = pd.DataFrame(model_metrics).T
    perf_df = perf_df.round(4)
    # Update accuracy values for highest and lowest (for extra table only)
    if 'accuracy' in perf_df.columns:
        max_acc = perf_df['accuracy'].max()
        min_acc = perf_df['accuracy'].min()
        max_idx = perf_df['accuracy'].idxmax()
        min_idx = perf_df['accuracy'].idxmin()
    # Display main metrics table (all models, no labels) with all rows visible (not scrollable)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(perf_df, use_container_width=True)
    with col2:
        best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; padding: 32px 16px; color: white; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.1);'>
            <div style='font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;'>🏆 Best Model</div>
            <div style='font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;'>{best_model[0]}</div>
            <div style='font-size: 1.2rem; margin-top: 1rem;'>Accuracy: {best_model[1]['accuracy']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    # Extra table for highest and lowest accuracy models
    st.markdown("## 🥇Model Accuracy Comparison (Highest & Lowest Accuracy)")
    extra_df = perf_df.loc[[max_idx, min_idx]].copy()
    if str(max_idx) == str(min_idx):
        # Only one model, show only one row
        extra_df = perf_df.loc[[max_idx]].copy()
    if 'accuracy' in extra_df.columns:
        if max_idx in extra_df.index:
            extra_df.loc[max_idx, 'accuracy'] = f"Highest: {max_acc:.2%}"
        if min_idx in extra_df.index:
            extra_df.loc[min_idx, 'accuracy'] = f"Lowest: {min_acc:.2%}"
    st.dataframe(extra_df, use_container_width=True)
    # Performance visualization
    st.markdown("### 📈 Performance Metrics Visualization")
    # Create subplots for different metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    models = list(model_metrics.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for i, metric in enumerate(metrics):
        row = (i // 2) + 1
        col = (i % 2) + 1
        values = [model_metrics[model][metric] for model in models]
        fig.add_trace(
            go.Bar(x=models, y=values, name=metric.title()),
            row=row, col=col
        )
    fig.update_layout(height=600, showlegend=False, title_text="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)
    # Training time comparison
    st.markdown("### ⏱️ Training Time Comparison")
    training_times = [model_metrics[model]['training_time'] for model in models]
    fig_time = px.bar(
        x=models, 
        y=training_times,
        title="Model Training Time (seconds)",
        labels={'x': 'Model', 'y': 'Training Time (seconds)'}
    )
    fig_time.update_traces(marker_color='lightcoral')
    st.plotly_chart(fig_time, use_container_width=True)
    # Cross-validation scores
    st.markdown("### 🔄 Cross-Validation Results")
    if st.button("Run Cross-Validation Analysis"):
        with st.spinner("Running cross-validation..."):
            cv_results = {}
            for name, model in trained_models.items():
                cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
                cv_results[name] = {
                    'mean_cv_score': cv_scores.mean(),
                    'std_cv_score': cv_scores.std(),
                    'cv_scores': cv_scores
                }
            # Display CV results
            cv_df = pd.DataFrame({
                'Model': list(cv_results.keys()),
                'Mean CV Score': [cv_results[model]['mean_cv_score'] for model in cv_results.keys()],
                'Std CV Score': [cv_results[model]['std_cv_score'] for model in cv_results.keys()]
            }).round(4)
            st.dataframe(cv_df, use_container_width=True)

def show_prediction_page(trained_models, encoder, scaler, feature_columns, model_metrics):
    """Display the prediction interface"""
    st.markdown('<h2 class="sub-header">🔮 Heart Disease Prediction</h2>', unsafe_allow_html=True)
    
    if trained_models is None:
        st.error("❌ Models not available. Please ensure models are trained.")
        return
    
    # Best model for prediction
    best_model_name = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = trained_models[best_model_name]
    
    st.markdown(f"""
    <div class="info-box">
        <strong>ℹ️ Using Model:</strong> {best_model_name} (Accuracy: {model_metrics[best_model_name]['accuracy']:.2%})
    </div>
    """, unsafe_allow_html=True)
    
    # Input form
    st.markdown("### 📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Personal Information")
        age = st.slider("Age", min_value=20, max_value=100, value=50, help="Patient's age in years")
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Patient's biological sex")
        
        st.markdown("#### Vital Signs")
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120, help="Systolic blood pressure at rest")
        cholesterol = st.slider("Cholesterol Level (mg/dl)", min_value=100, max_value=600, value=250, help="Serum cholesterol level")
        max_heart_rate = st.slider("Maximum Heart Rate", min_value=60, max_value=202, value=150, help="Maximum heart rate achieved during exercise")
    
    with col2:
        st.markdown("#### Medical Tests")
        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical Angina",
                2: "Atypical Angina", 
                3: "Non-anginal Pain",
                4: "Asymptomatic"
            }[x],
            help="Type of chest pain experienced"
        )
        
        fasting_blood_sugar = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether fasting blood sugar is above 120 mg/dl"
        )
        
        resting_ecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }[x],
            help="Results of resting electrocardiogram"
        )
        
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether angina is induced by exercise"
        )
        
        oldpeak = st.slider("ST Depression", min_value=0.0, max_value=6.0, value=0.0, step=0.1, help="ST depression induced by exercise relative to rest")
        
        st_slope = st.selectbox(
            "ST Slope",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Upsloping",
                2: "Flat",
                3: "Downsloping"
            }[x],
            help="Slope of the peak exercise ST segment"
        )
    
    # Prediction button
    if st.button("🔮 Predict Heart Disease", type="primary"):
        # Prepare input data
        input_data = {
            'age': age,
            'sex': sex,
            'chest_pain_type': chest_pain_type,
            'resting_bp_s': resting_bp,
            'cholesterol': cholesterol,
            'fasting_blood_sugar': fasting_blood_sugar,
            'resting_ecg': resting_ecg,
            'max_heart_rate': max_heart_rate,
            'exercise_angina': exercise_angina,
            'oldpeak': oldpeak,
            'st_slope': st_slope
        }
        
        # Preprocess input
        processed_input = preprocess_input(input_data, encoder, scaler, feature_columns)
        
        # Make prediction with timing
        start_time = time.time()
        prediction = best_model.predict(processed_input)[0]
        prediction_proba = best_model.predict_proba(processed_input)[0][1] if hasattr(best_model, 'predict_proba') else None
        prediction_time = time.time() - start_time
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="prediction-card danger-prediction">
                    <h2>⚠️ HIGH RISK</h2>
                    <h3>Heart Disease Detected</h3>
                    <p>Please consult a healthcare professional immediately.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-card safe-prediction">
                    <h2>✅ LOW RISK</h2>
                    <h3>No Heart Disease Detected</h3>
                    <p>Continue maintaining a healthy lifestyle.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if prediction_proba is not None:
                confidence = prediction_proba if prediction == 1 else (1 - prediction_proba)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Confidence Level</h3>
                    <h2>{confidence:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Prediction Time</h3>
                <h2>{prediction_time:.3f}s</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed probability
        if prediction_proba is not None:
            st.markdown("### 📊 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction_proba * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probability of Heart Disease"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk factors analysis
                st.markdown("#### 🚨 Risk Factors Analysis")
                
                risk_factors = []
                if age > 65:
                    risk_factors.append("Advanced age (>65 years)")
                if resting_bp > 140:
                    risk_factors.append("High blood pressure (>140 mm Hg)")
                if cholesterol > 300:
                    risk_factors.append("High cholesterol (>300 mg/dl)")
                if max_heart_rate < 100:
                    risk_factors.append("Low maximum heart rate (<100)")
                if exercise_angina == 1:
                    risk_factors.append("Exercise-induced angina")
                if oldpeak > 2:
                    risk_factors.append("Significant ST depression (>2)")
                
                if risk_factors:
                    st.markdown("**Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.markdown(f"• {factor}")
                else:
                    st.markdown("✅ No significant risk factors identified")

def show_data_analysis(df):
    """Display data analysis and visualizations"""
    st.markdown('<h2 class="sub-header">📈 Data Analysis & Insights</h2>', unsafe_allow_html=True)
    # Basic statistics
    st.markdown("### 📊 Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Heart Disease Cases", df['target'].sum())
    with col3:
        st.metric("Healthy Cases", len(df) - df['target'].sum())
    with col4:
        st.metric("Disease Rate", f"{df['target'].mean():.1%}")
    # Distribution plots
    st.markdown("### 📈 Feature Distributions")
    fig_age = px.histogram(
        df, x='age', color='target',
        title="Age Distribution by Heart Disease Status",
        labels={'age': 'Age', 'target': 'Heart Disease'},
        color_discrete_map={0: 'blue', 1: 'red'}
    )
    st.plotly_chart(fig_age, use_container_width=True)
    fig_scatter = px.scatter(
        df, x='resting_bp_s', y='cholesterol', color='target',
        title="Blood Pressure vs Cholesterol by Heart Disease Status",
        labels={'resting_bp_s': 'Resting Blood Pressure', 'cholesterol': 'Cholesterol'},
        color_discrete_map={0: 'blue', 1: 'red'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("### 🔗 Feature Correlations")
    corr_matrix = df.corr()
    fig_heatmap = px.imshow(
        corr_matrix,
        title="Feature Correlation Heatmap",
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("### 📊 Disease Prevalence by Categories")
    col1, col2 = st.columns(2)
    with col1:
        sex_dist = df.groupby(['sex', 'target']).size().unstack(fill_value=0)
        fig_sex = px.bar(
            x=['Female', 'Male'],
            y=[sex_dist[0], sex_dist[1]],
            title="Heart Disease by Sex",
            labels={'x': 'Sex', 'y': 'Count'}
        )
        st.plotly_chart(fig_sex, use_container_width=True)
    with col2:
        chest_pain_dist = df.groupby(['chest_pain_type', 'target']).size().unstack(fill_value=0)
        y_vals = [chest_pain_dist.loc[i].sum() if i in chest_pain_dist.index else 0 for i in [1,2,3,4]]
        fig_chest = px.bar(
            x=['Typical Angina', 'Atypical Angina', 'Non-anginal', 'Asymptomatic'],
            y=y_vals,
            title="Heart Disease by Chest Pain Type",
            labels={'x': 'Chest Pain Type', 'y': 'Count'}
        )
        st.plotly_chart(fig_chest, use_container_width=True)

def diabetes_main():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import time
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import xgboost as xgb
    import lightgbm as lgb
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import warnings
    warnings.filterwarnings('ignore')

    # Page config
    #st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="wide")

    # Custom CSS for attractive styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #16a085;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2c3e50;
            margin-bottom: 1rem;
            font-weight: bold;
        }
        .metric-card {
            background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .prediction-card {
            background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
            padding: 2rem;
            border-radius: 15px;
            color: #2c3e50;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .safe-prediction {
            background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
            color: white;
        }
        .danger-prediction {
            background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
            color: #2c3e50;
        }
        .stButton > button {
            background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .info-box {
            background: rgba(22, 160, 133, 0.1);
            border-left: 4px solid #16a085;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        df = pd.read_csv('diabetes_prediction_dataset.csv')
        return df

    @st.cache_resource
    def preprocess_and_train(df):
        # Data cleaning
        df = df[(df['age'] > 0) & (df['age'] <= 120)]
        df = df[(df['bmi'] >= 10)]
        for col in ['bmi', 'blood_glucose_level']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        # Feature engineering
        df['bmi_age_interaction'] = df['bmi'] * df['age']
        df['glucose_HbA1c_interaction'] = df['blood_glucose_level'] * df['HbA1c_level']
        # One-hot encoding
        categorical_cols = ['gender', 'smoking_history']
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
        df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
        # Scaling
        scaler = MinMaxScaler()
        numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'bmi_age_interaction', 'glucose_HbA1c_interaction']
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        # Split
        X = df.drop('diabetes', axis=1)
        y = df['diabetes']
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        # Models
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'LightGBM': lgb.LGBMClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        param_dists = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5, 6],
                'subsample': [0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5, 6],
                'subsample': [0.8, 1.0]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5, 6],
                'subsample': [0.8, 1.0]
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100]
            }
        }
        best_models = {}
        metrics = {}
        for name, model in models.items():
            # Fast: use default params, skip RandomizedSearchCV
            model.fit(X_val, y_val)
            best = model
            best_models[name] = best
            y_pred = best.predict(X_test)
            start_pred = time.time()
            _ = best.predict(X_test)
            pred_time = time.time() - start_pred
            y_prob = best.predict_proba(X_test)[:, 1] if hasattr(best, 'predict_proba') else None
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
            metrics[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc, 'training_time': getattr(best, 'training_time_', 0), 'prediction_time': pred_time}
        # Select best model
        best_model_name = max(metrics, key=lambda k: metrics[k]['accuracy'])
        best_model = best_models[best_model_name]
        return best_models, metrics, best_model_name, best_model, encoder, scaler, X.columns, X_test, y_test

    def preprocess_input_diabetes(input_data, encoder, scaler, feature_columns):
        df_input = pd.DataFrame([input_data])
        categorical_cols = ['gender', 'smoking_history']
        numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        # Feature engineering
        df_input['bmi_age_interaction'] = df_input['bmi'] * df_input['age']
        df_input['glucose_HbA1c_interaction'] = df_input['blood_glucose_level'] * df_input['HbA1c_level']
        # Encode
        encoded = encoder.transform(df_input[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
        df_input = pd.concat([df_input.drop(categorical_cols, axis=1), encoded_df], axis=1)
        # Scale
        scale_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'bmi_age_interaction', 'glucose_HbA1c_interaction']
        df_input[scale_cols] = scaler.transform(df_input[scale_cols])
        # Ensure order
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)
        return df_input

    def main():
        st.markdown('<h1 class="main-header">🩺 Diabetes Prediction System</h1>', unsafe_allow_html=True)
        df = load_data()
        best_models, metrics, best_model_name, best_model, encoder, scaler, feature_columns, X_test, y_test = preprocess_and_train(df)
        st.sidebar.markdown("## Navigation")
        page = st.sidebar.selectbox("Choose a page:", ["🏠 Home", "📊 Model Performance", "🔮 Make Prediction", "📈 Data Analysis"])
        if page == "🏠 Home":
            show_home_page_diabetes(df)
        elif page == "📊 Model Performance":
            show_model_performance_diabetes(metrics, best_models, X_test, y_test)
        elif page == "🔮 Make Prediction":
            show_prediction_page_diabetes(best_model, encoder, scaler, feature_columns, metrics, best_model_name)
        elif page == "📈 Data Analysis":
            show_data_analysis_diabetes(df)

    def show_home_page_diabetes(df):
        st.markdown('<h2 class="sub-header">Welcome to the Diabetes Prediction System</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### 🎯About This System
            This advanced diabetes prediction system uses machine learning to analyze patient data and provide accurate predictions about the likelihood of diabetes. Multiple state-of-the-art algorithms are used to ensure the highest accuracy possible.
            ### 🔬Key Features
            - Multiple ML Models: Random Forest, XGBoost, LightGBM, Gradient Boosting, Logistic Regression
            - Real-time Predictions: Get instant predictions with confidence scores
            - Performance Metrics: Detailed accuracy, precision, recall, and F1-score analysis
            - Interactive Interface: User-friendly design for easy data input
            - Data Visualization: Comprehensive analysis of the dataset
            ### 📋 Required Symptoms 
            The system analyzes the following features:
            """)
            features = [
                "Age", "Gender", "BMI", "HbA1c Level", "Blood Glucose Level", "Smoking History"
            ]
            for i, feat in enumerate(features, 1):
                st.markdown(f"**{i}.** {feat}")
        with col2:
            st.markdown("### 📊 Dataset Overview")
            total_patients = len(df)
            diabetes_cases = df['diabetes'].sum()
            healthy_cases = total_patients - diabetes_cases
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Patients</h3>
                <h2>{total_patients:,}</h2>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Diabetes Cases</h3>
                <h2>{diabetes_cases:,}</h2>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Healthy Cases</h3>
                <h2>{healthy_cases:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🚀Getting Started")
        st.markdown("""
        1. Navigate to 'Make Prediction' to input patient data
        2. Check 'Model Performance' to see accuracy metrics
        3. Explore 'Data Analysis' to understand the dataset
        """)

    def show_model_performance_diabetes(metrics, best_models, X_test, y_test):
        st.markdown('<h2 class="sub-header"> 📊 Model Performance Analysis</h2>', unsafe_allow_html=True)
        st.markdown('## 🏆 All Models Comparison')
        perf_df = pd.DataFrame(metrics).T.round(4)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(perf_df, use_container_width=True)
        with col2:
            best_model = max(metrics.items(), key=lambda x: x[1]['accuracy'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>Best Model</h3>
                <h2>{best_model[0]}</h2>
                <p>Accuracy: {best_model[1]['accuracy']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        # Extra table for highest and lowest accuracy models
        st.markdown('## 🥇Model Accuracy Comparison (Highest & Lowest Accuracy)')
        if 'accuracy' in perf_df.columns:
            max_acc = perf_df['accuracy'].max()
            min_acc = perf_df['accuracy'].min()
            max_idx = perf_df['accuracy'].idxmax()
            min_idx = perf_df['accuracy'].idxmin()
            extra_df = perf_df.loc[[max_idx, min_idx]].copy()
            if str(max_idx) == str(min_idx):
                extra_df = perf_df.loc[[max_idx]].copy()
            if max_idx in extra_df.index:
                extra_df.loc[max_idx, 'accuracy'] = f"Highest: {max_acc:.2%}"
            if min_idx in extra_df.index:
                extra_df.loc[min_idx, 'accuracy'] = f"Lowest: {min_acc:.2%}"
            st.dataframe(extra_df, use_container_width=True)
        st.markdown("### 📊 Performance Metrics Visualization")
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score'), specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]])
        models = list(metrics.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        for i, metric in enumerate(metric_names):
            row = (i // 2) + 1
            col = (i % 2) + 1
            values = [metrics[model][metric] for model in models]
            fig.add_trace(go.Bar(x=models, y=values, name=metric.title()), row=row, col=col)
        fig.update_layout(height=600, showlegend=False, title_text="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
        # Prediction time comparison
        st.markdown("## ⏱️ Prediction Time Comparison")
        prediction_times = [metrics[model]['prediction_time'] for model in models if 'prediction_time' in metrics[model]]
        fig_pred_time = px.bar(
            x=models,
            y=prediction_times,
            title="Model Prediction Time (seconds)",
            labels={'x': 'Model', 'y': 'Prediction Time (seconds)'}
        )
        fig_pred_time.update_traces(marker_color='mediumpurple')
        st.plotly_chart(fig_pred_time, use_container_width=True)
        # Cross-validation scores
        st.markdown("## 🌀 Cross-Validation Results")
        if st.button("Run Cross-Validation Analysis", key="cv_diabetes", help="Run 5-fold cross-validation on all models"):
            with st.spinner("Running cross-validation..."):
                import sklearn.model_selection
                cv_results = {}
                for name, model in best_models.items():
                    try:
                        cv_scores = sklearn.model_selection.cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
                    except Exception:
                        continue
                    cv_results[name] = {
                        'mean_cv_score': cv_scores.mean(),
                        'std_cv_score': cv_scores.std(),
                        'cv_scores': cv_scores
                    }
                if cv_results:
                    cv_df = pd.DataFrame({
                        'Model': list(cv_results.keys()),
                        'Mean CV Score': [cv_results[model]['mean_cv_score'] for model in cv_results.keys()],
                        'Std CV Score': [cv_results[model]['std_cv_score'] for model in cv_results.keys()]
                    }).round(4)
                    st.dataframe(cv_df, use_container_width=True)

    def show_prediction_page_diabetes(best_model, encoder, scaler, feature_columns, metrics, best_model_name):
        st.markdown('<h2 class="sub-header">Diabetes Prediction</h2>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
            <strong>Using Model:</strong> {best_model_name} (Accuracy: {metrics[best_model_name]['accuracy']:.2%})
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Personal Information")
            age = st.slider("Age", min_value=1, max_value=120, value=40)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            bmi = st.slider("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            smoking_history = st.selectbox("Smoking History", options=["never", "current", "former", "not current", "ever", "No Info"])
        with col2:
            st.markdown("#### Medical Information")
            hba1c = st.slider("HbA1c Level", min_value=3.0, max_value=15.0, value=6.0, step=0.1)
            glucose = st.slider("Blood Glucose Level", min_value=50.0, max_value=300.0, value=120.0, step=0.1)
        if st.button("Predict Diabetes", type="primary"):
            input_data = {
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'HbA1c_level': hba1c,
                'blood_glucose_level': glucose,
                'smoking_history': smoking_history
            }
            processed_input = preprocess_input_diabetes(input_data, encoder, scaler, feature_columns)
            start_time = time.time()
            prediction = best_model.predict(processed_input)[0]
            prediction_proba = best_model.predict_proba(processed_input)[0][1] if hasattr(best_model, 'predict_proba') else None
            prediction_time = time.time() - start_time
            st.markdown("---")
            st.markdown("### Prediction Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div class="prediction-card danger-prediction">
                        <h2>⚠️ HIGH RISK</h2>
                        <h3>Diabetes Detected</h3>
                        <p>Please consult a healthcare professional.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-card safe-prediction">
                        <h2>✅ LOW RISK</h2>
                        <h3>No Diabetes Detected</h3>
                        <p>Continue maintaining a healthy lifestyle.</p>
                    </div>
                    """, unsafe_allow_html=True)
            with col2:
                if prediction_proba is not None:
                    confidence = prediction_proba if prediction == 1 else (1 - prediction_proba)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Confidence Level</h3>
                        <h2>{confidence:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Prediction Time</h3>
                    <h2>{prediction_time:.3f}s</h2>
                </div>
                """, unsafe_allow_html=True)
            if prediction_proba is not None:
                st.markdown("### 📊 Detailed Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = prediction_proba * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Probability of Diabetes"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#f7971e"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("#### 🚨 Risk Factors Analysis")
                    st.markdown("**Identified Risk Factors:**")
                    risk_factors = []
                    if bmi > 30:
                        risk_factors.append("High BMI (>30)")
                    if hba1c > 6.5:
                        risk_factors.append("High HbA1c (>6.5)")
                    if glucose > 140:
                        risk_factors.append("High blood glucose (>140 mg/dl)")
                    if smoking_history in ["current", "ever"]:
                        risk_factors.append("History of smoking")
                    if age > 60:
                        risk_factors.append("Older age (>60 years)")
                    if not risk_factors:
                        st.markdown("✅ No significant risk factors identified")
                    else:
                        for factor in risk_factors:
                            st.markdown(f"• {factor}")

    def show_data_analysis_diabetes(df):
        st.markdown('<h2 class="sub-header">Data Analysis & Insights</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            st.metric("Diabetes Cases", df['diabetes'].sum())
        with col3:
            st.metric("Healthy Cases", len(df) - df['diabetes'].sum())
        with col4:
            st.metric("Disease Rate", f"{df['diabetes'].mean():.1%}")
        st.markdown("### Feature Distributions")
        fig_age = px.histogram(df, x='age', color='diabetes', title="Age Distribution by Diabetes Status", labels={'age': 'Age', 'diabetes': 'Diabetes'}, color_discrete_map={0: 'blue', 1: 'red'})
        st.plotly_chart(fig_age, use_container_width=True)
        fig_bmi = px.histogram(df, x='bmi', color='diabetes', title="BMI Distribution by Diabetes Status", labels={'bmi': 'BMI', 'diabetes': 'Diabetes'}, color_discrete_map={0: 'blue', 1: 'red'})
        st.plotly_chart(fig_bmi, use_container_width=True)
        st.markdown("### Feature Correlations")
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        fig_heatmap = px.imshow(corr_matrix, title="Feature Correlation Heatmap", color_continuous_scale='RdBu', aspect="auto")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown("### Disease Prevalence by Categories")
        col1, col2 = st.columns(2)
        with col1:
            if 'gender' in df.columns:
                gender_dist = df.groupby(['gender', 'diabetes']).size().unstack(fill_value=0)
                fig_gender = px.bar(x=gender_dist.index, y=[gender_dist[0], gender_dist[1]], title="Diabetes by Gender", labels={'x': 'Gender', 'y': 'Count'})
                st.plotly_chart(fig_gender, use_container_width=True)
        with col2:
            if 'smoking_history' in df.columns:
                smoke_dist = df.groupby(['smoking_history', 'diabetes']).size().unstack(fill_value=0)
                fig_smoke = px.bar(x=smoke_dist.index, y=[smoke_dist[0], smoke_dist[1]], title="Diabetes by Smoking History", labels={'x': 'Smoking History', 'y': 'Count'})
                st.plotly_chart(fig_smoke, use_container_width=True)

    if __name__ == "__main__":
        main()

def dengue_main():
    """Dengue Fever Prediction Module"""
    import streamlit as st
    import pandas as pd
    import numpy as np
    import time
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    import xgboost as xgb
    import lightgbm as lgb
    import warnings
    warnings.filterwarnings('ignore')

    # Custom CSS for dengue styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #e74c3c;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2c3e50;
            margin-bottom: 1rem;
            font-weight: bold;
        }
        .metric-card {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .prediction-card {
            background: linear-gradient(135deg, #ff9ff3 0%, #f368e0 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .safe-prediction {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        .danger-prediction {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }
        .stButton > button {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .info-box {
            background: rgba(255, 107, 107, 0.1);
            border-left: 4px solid #ff6b6b;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        """Load and cache the dengue dataset"""
        try:
            # First, let's check if the file exists and show basic info
            import os
            file_path = 'Dengue_25000.csv'
            
            if not os.path.exists(file_path):
                st.warning(f"⚠️ File not found: {file_path}")
                return create_synthetic_dengue_data()
            
            # Try to load the actual dengue dataset
            df = pd.read_csv(file_path)
            
            # Check if Diagnosis column exists, if not create it based on symptoms
            if 'Diagnosis' not in df.columns:
                # Create diagnosis based on key indicators
                # High probability of dengue if multiple symptoms present
                dengue_score = (
                    df['Fever'] * 3 +
                    df['Headache'] * 2 +
                    df['Retro_Orbital_Pain'] * 3 +
                    df['Muscle_Joint_Pain'] * 2 +
                    df['Skin_Rash'] * 1 +
                    df['Bleeding'] * 4 +
                    df['Vomiting_Nausea'] * 1 +
                    df['Abdominal_Pain'] * 2 +
                    (df['Platelet_Count'] < 150000).astype(int) * 3 +
                    (df['WBC_Count'] < 4000).astype(int) * 2 +
                    df['NS1_Antigen_Test'] * 5 +
                    df['IgM_IgG_Test'] * 4
                )
                
                # Normalize score and create binary target
                dengue_score = (dengue_score - dengue_score.min()) / (dengue_score.max() - dengue_score.min())
                df['Diagnosis'] = (dengue_score > 0.5).astype(int)
            
            # Ensure all required columns exist
            required_columns = [
                'Fever', 'Fever_Duration_Days', 'Headache', 'Retro_Orbital_Pain',
                'Muscle_Joint_Pain', 'Skin_Rash', 'Bleeding', 'Vomiting_Nausea',
                'Abdominal_Pain', 'Platelet_Count', 'WBC_Count', 'NS1_Antigen_Test', 'IgM_IgG_Test'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.warning(f"⚠️ Missing columns in dataset: {missing_columns}")
                return create_synthetic_dengue_data()
            
            return df
            
        except FileNotFoundError:
            st.warning("⚠️ Dengue dataset not found. Using synthetic data for demonstration.")
            return create_synthetic_dengue_data()
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return create_synthetic_dengue_data()

    def create_synthetic_dengue_data():
        """Create synthetic dengue fever data for demonstration"""
        np.random.seed(42)
        n_samples = 25000
        
        # Generate synthetic data based on dengue fever characteristics
        data = {
            'Fever': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),  # 90% have fever
            'Fever_Duration_Days': np.random.randint(-5, 15, n_samples),
            'Headache': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'Retro_Orbital_Pain': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'Muscle_Joint_Pain': np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),
            'Skin_Rash': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'Bleeding': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'Vomiting_Nausea': np.random.choice([0, 1], n_samples, p=[0.25, 0.75]),
            'Abdominal_Pain': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'Platelet_Count': np.random.randint(50000, 300000, n_samples),
            'WBC_Count': np.random.randint(2000, 12000, n_samples),
            'NS1_Antigen_Test': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'IgM_IgG_Test': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on dengue symptoms
        # High probability of dengue if multiple symptoms present
        dengue_score = (
            df['Fever'] * 3 +
            df['Headache'] * 2 +
            df['Retro_Orbital_Pain'] * 3 +
            df['Muscle_Joint_Pain'] * 2 +
            df['Skin_Rash'] * 1 +
            df['Bleeding'] * 4 +
            df['Vomiting_Nausea'] * 1 +
            df['Abdominal_Pain'] * 2 +
            (df['Platelet_Count'] < 150000).astype(int) * 3 +
            (df['WBC_Count'] < 4000).astype(int) * 2 +
            df['NS1_Antigen_Test'] * 5 +
            df['IgM_IgG_Test'] * 4
        )
        
        # Normalize score and create binary target
        dengue_score = (dengue_score - dengue_score.min()) / (dengue_score.max() - dengue_score.min())
        df['Diagnosis'] = (dengue_score > 0.5).astype(int)
        
        return df

    @st.cache_resource
    def train_models(df):
        """Train multiple models and cache the results"""
        if df is None or df.empty:
            st.error("❌ No data available for training models.")
            return None, None, None, None, None, None
        
        try:
            # Data preprocessing
            # All features are numerical in dengue dataset
            feature_cols = ['Fever', 'Fever_Duration_Days', 'Headache', 'Retro_Orbital_Pain', 
                           'Muscle_Joint_Pain', 'Skin_Rash', 'Bleeding', 'Vomiting_Nausea', 
                           'Abdominal_Pain', 'Platelet_Count', 'WBC_Count', 'NS1_Antigen_Test', 'IgM_IgG_Test']
            
            # Check if all required columns exist
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                return None, None, None, None, None, None
            
            # Check if Diagnosis column exists
            if 'Diagnosis' not in df.columns:
                st.error("❌ Diagnosis column not found in dataset.")
                return None, None, None, None, None, None
            
            # Handle missing values
            df_clean = df.copy()
            df_clean = df_clean.fillna(df_clean.mean())
            
            # Feature engineering for dengue
            df_clean['Platelet_Low'] = (df_clean['Platelet_Count'] < 150000).astype(int)
            df_clean['WBC_Low'] = (df_clean['WBC_Count'] < 4000).astype(int)
            df_clean['Multiple_Symptoms'] = (df_clean['Fever'] + df_clean['Headache'] + df_clean['Retro_Orbital_Pain'] + 
                                            df_clean['Muscle_Joint_Pain'] + df_clean['Skin_Rash'] + df_clean['Bleeding'] + 
                                            df_clean['Vomiting_Nausea'] + df_clean['Abdominal_Pain'])
            df_clean['Test_Positive'] = (df_clean['NS1_Antigen_Test'] + df_clean['IgM_IgG_Test'])
            
            # Updated feature columns
            feature_cols = ['Fever', 'Fever_Duration_Days', 'Headache', 'Retro_Orbital_Pain', 
                           'Muscle_Joint_Pain', 'Skin_Rash', 'Bleeding', 'Vomiting_Nausea', 
                           'Abdominal_Pain', 'Platelet_Count', 'WBC_Count', 'NS1_Antigen_Test', 
                           'IgM_IgG_Test', 'Platelet_Low', 'WBC_Low', 'Multiple_Symptoms', 'Test_Positive']
            
            # Ensure all engineered features exist
            for col in feature_cols:
                if col not in df_clean.columns:
                    st.error(f"❌ Missing engineered feature: {col}")
                    return None, None, None, None, None, None
            
            # Scaling
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_clean[feature_cols])
            scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
            scaled_df['Diagnosis'] = df_clean['Diagnosis']
            
            # Prepare features and target
            X = scaled_df.drop('Diagnosis', axis=1)
            y = scaled_df['Diagnosis']
            
            # Check for class balance
            class_counts = y.value_counts()
            if len(class_counts) < 2:
                st.error("❌ Dataset contains only one class. Cannot train classification models.")
                return None, None, None, None, None, None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Train models
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
                'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42)
            }
            
            trained_models = {}
            model_metrics = {}
            
            with st.spinner("🔄 Training models... This may take a few moments."):
                for name, model in models.items():
                    try:
                        start_time = time.time()
                        model.fit(X_train, y_train)
                        training_time = time.time() - start_time
                        
                        y_pred = model.predict(X_test)
                        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                        
                        metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, zero_division=0),
                            'recall': recall_score(y_test, y_pred, zero_division=0),
                            'f1_score': f1_score(y_test, y_pred, zero_division=0),
                            'training_time': training_time
                        }
                        
                        if y_prob is not None:
                            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
                        
                        trained_models[name] = model
                        model_metrics[name] = metrics
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error training {name}: {str(e)}")
                        continue
            
            if not trained_models:
                st.error("❌ No models were successfully trained.")
                return None, None, None, None, None, None
            
            return trained_models, model_metrics, scaler, feature_cols, X_test, y_test
            
        except Exception as e:
            st.error(f"❌ Error in model training: {str(e)}")
            return None, None, None, None, None, None

    def preprocess_input(input_data, scaler, feature_columns):
        """Preprocess user input for prediction"""
        # Create DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Feature engineering
        df_input['Platelet_Low'] = (df_input['Platelet_Count'] < 150000).astype(int)
        df_input['WBC_Low'] = (df_input['WBC_Count'] < 4000).astype(int)
        df_input['Multiple_Symptoms'] = (df_input['Fever'] + df_input['Headache'] + df_input['Retro_Orbital_Pain'] + 
                                        df_input['Muscle_Joint_Pain'] + df_input['Skin_Rash'] + df_input['Bleeding'] + 
                                        df_input['Vomiting_Nausea'] + df_input['Abdominal_Pain'])
        df_input['Test_Positive'] = (df_input['NS1_Antigen_Test'] + df_input['IgM_IgG_Test'])
        
        # Scale features
        processed_df = pd.DataFrame(scaler.transform(df_input[feature_columns]), columns=feature_columns)
        
        return processed_df

    def main():
        # Main header
        st.markdown('<h1 class="main-header">🦟 Dengue Fever Prediction System</h1>', unsafe_allow_html=True)
        
        # Load data
        df = load_data()
        if df is None:
            return
        
        # Debug mode - uncomment to see dataset structure
        # debug_dataset_structure(df)
        
        # Sidebar
        st.sidebar.markdown("## 🎯 Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["🏠 Home", "📊 Model Performance", "🔮 Make Prediction", "📈 Data Analysis"]
        )
        
        # Train models (cached)
        trained_models, model_metrics, scaler, feature_columns, X_test, y_test = train_models(df)
        
        if page == "🏠 Home":
            show_home_page(df)
        
        elif page == "📊 Model Performance":
            show_model_performance(model_metrics, trained_models, X_test, y_test)
        
        elif page == "🔮 Make Prediction":
            show_prediction_page(trained_models, scaler, feature_columns, model_metrics)
        
        elif page == "📈 Data Analysis":
            show_data_analysis(df)

    def show_home_page(df):
        """Display the home page with overview information"""
        st.markdown('<h2 class="sub-header">Welcome to the Dengue Fever Prediction System</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 About This System
            
            This advanced dengue fever prediction system uses machine learning to analyze patient symptoms 
            and provide accurate predictions about the likelihood of dengue fever. Our system incorporates 
            multiple state-of-the-art algorithms to ensure the highest accuracy possible.
            
            ### 🔬 Key Features
            
            - **Multiple ML Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM
            - **Real-time Predictions**: Get instant predictions with confidence scores
            - **Performance Metrics**: Detailed accuracy, precision, recall, and F1-score analysis
            - **Interactive Interface**: User-friendly design for easy symptom input
            - **Data Visualization**: Comprehensive analysis of the dataset
            
            ### 🚨 Critical Dengue Symptoms
            
            The system analyzes the following key symptoms and measurements:
            """)
            
            # Critical symptoms
            critical_symptoms = [
                "High Fever (≥38°C)",
                "Severe Headache",
                "Retro-orbital Pain (Eye Pain)",
                "Muscle and Joint Pain",
                "Skin Rash",
                "Bleeding Manifestations",
                "Vomiting and Nausea",
                "Abdominal Pain"
            ]
            
            for i, symptom in enumerate(critical_symptoms, 1):
                st.markdown(f"**{i}.** {symptom}")
            
            st.markdown("### 🧪 Laboratory Tests")
            
            lab_tests = [
                "Platelet Count (Normal: 150,000-450,000/μL)",
                "White Blood Cell Count (Normal: 4,000-11,000/μL)",
                "NS1 Antigen Test",
                "IgM/IgG Antibody Test"
            ]
            
            for i, test in enumerate(lab_tests, 1):
                st.markdown(f"**{i}.** {test}")
        
        with col2:
            st.markdown("### 📊 Dataset Overview")
            
            # Dataset statistics
            total_patients = len(df)
            dengue_cases = df['Diagnosis'].sum()
            healthy_cases = total_patients - dengue_cases
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Patients</h3>
                <h2>{total_patients:,}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Dengue Cases</h3>
                <h2>{dengue_cases:,}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Healthy Cases</h3>
                <h2>{healthy_cases:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. **Navigate to 'Make Prediction'** to input patient symptoms
        2. **Check 'Model Performance'** to see accuracy metrics
        3. **Explore 'Data Analysis'** to understand the dataset
        """)
        
        # Dengue warning
        st.markdown("""
        <div class="info-box">
            <strong>⚠️ Important:</strong> This system is for educational and screening purposes only. 
            Always consult a healthcare professional for proper diagnosis and treatment of dengue fever.
        </div>
        """, unsafe_allow_html=True)

    def show_model_performance(model_metrics, trained_models, X_test, y_test):
        """Display comprehensive model performance metrics and visualizations"""
        st.markdown('<h2 class="sub-header">📊 Comprehensive Model Performance Analysis</h2>', unsafe_allow_html=True)
        
        if model_metrics is None:
            st.error("❌ Model metrics not available. Please ensure models are trained.")
            return
        
        # Performance comparison
        st.markdown("### 🏆 All Models Comparison")
        
        # Create performance DataFrame
        perf_df = pd.DataFrame(model_metrics).T
        perf_df = perf_df.round(4)
        
        # Rename columns to bold format
        perf_df.columns = [f"{col}" for col in perf_df.columns]
        perf_df.index = [f"{idx}" for idx in perf_df.index]
        
        # Display main metrics table
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(perf_df, use_container_width=True)
        with col2:
            best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); border-radius: 16px; padding: 32px 16px; color: white; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.1);'>
                <div style='font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;'>🏆 Best Model</div>
                <div style='font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;'><strong>{best_model[0]}</strong></div>
                <div style='font-size: 1.2rem; margin-top: 1rem;'>Accuracy: {best_model[1]['accuracy']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if PLOTLY_AVAILABLE:
            # 1. Model Accuracy Comparison (Highest & Lowest)
            st.markdown("### 🏆 Model Accuracy Comparison (Highest & Lowest Accuracy)")
            
            # Find highest and lowest accuracy models
            sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            highest_model = sorted_models[0]
            lowest_model = sorted_models[-1]
            
            # Create comparison table
            comparison_data = []
            for model_name, metrics in [highest_model, lowest_model]:
                comparison_data.append({
                    'Model': f"{model_name}",
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1 Score': f"{metrics['f1_score']:.4f}",
                    'Training Time': f"{metrics['training_time']:.4f}",
                    'ROC AUC': f"{metrics.get('roc_auc', 'N/A')}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display the comparison table
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(comparison_df, use_container_width=True)
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%); border-radius: 16px; padding: 32px 16px; color: #333; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.1);'>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🥇</div>
                    <div style='font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;'>Best Model</div>
                    <div style='font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;'><strong>{highest_model[0]}</strong></div>
                    <div style='font-size: 1rem;'>Accuracy: {highest_model[1]['accuracy']:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # 2. Performance Metrics Visualization
            st.markdown("### 📈 Performance Metrics Comparison")
            
            # Create subplots for different metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('<b>Accuracy</b>', '<b>Precision</b>', '<b>Recall</b>', '<b>F1 Score</b>'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            models = list(model_metrics.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            colors = ['#ff6b6b', '#4facfe', '#feca57', '#48dbfb']
            
            for i, metric in enumerate(metrics):
                row = (i // 2) + 1
                col = (i % 2) + 1
                values = [model_metrics[model][metric] for model in models]
                fig.add_trace(
                    go.Bar(x=models, y=values, name=f"<b>{metric.title()}</b>", marker_color=colors[i]),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, showlegend=False, title_text="<b>Model Performance Comparison</b>")
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. ROC Curves
            st.markdown("### 📊 ROC Curves Analysis")
            
            fig_roc = go.Figure()
            
            for model_name, model in trained_models.items():
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate ROC curve
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    auc_score = roc_auc_score(y_test, y_prob)
                    
                    fig_roc.add_trace(
                        go.Scatter(
                            x=fpr, y=tpr,
                            name=f'<b>{model_name}</b> (AUC = {auc_score:.3f})',
                            mode='lines',
                            line=dict(width=2)
                        )
                    )
            
            # Add diagonal line
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name='<b>Random Classifier</b>',
                    mode='lines',
                    line=dict(dash='dash', color='gray')
                )
            )
            
            fig_roc.update_layout(
                title="<b>ROC Curves for All Models</b>",
                xaxis_title="<b>False Positive Rate</b>",
                yaxis_title="<b>True Positive Rate</b>",
                height=500
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # 3. Confusion Matrices
            st.markdown("### 🎯 Confusion Matrices")
            
            # Select top 3 models for confusion matrices
            top_models = sorted(model_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
            
            cols = st.columns(3)
            for i, (model_name, metrics) in enumerate(top_models):
                with cols[i]:
                    model = trained_models[model_name]
                    y_pred = model.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Create confusion matrix heatmap
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title=f"<b>{model_name}</b>\nAccuracy: {metrics['accuracy']:.3f}",
                        color_continuous_scale='Reds'
                    )
                    fig_cm.update_layout(
                        xaxis_title="<b>Predicted</b>",
                        yaxis_title="<b>Actual</b>",
                        height=300
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
            
            # 4. Feature Importance (for tree-based models)
            st.markdown("### 🔍 Feature Importance Analysis")
            
            # Get feature importance from tree-based models
            importance_data = []
            for model_name, model in trained_models.items():
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, importance in enumerate(importances):
                        importance_data.append([model_name, f'Feature_{i+1}', importance])
            
            if importance_data:
                importance_df = pd.DataFrame(importance_data, columns=['Model', 'Feature', 'Importance'])
                
                # Show top features for each model
                fig_importance = px.bar(
                    importance_df.groupby('Model')['Importance'].sum().reset_index(),
                    x='Model', y='Importance',
                    title="<b>Overall Feature Importance by Model</b>",
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # 5. Training Time Comparison
            st.markdown("### ⏱️ Training Time Comparison")
            st.markdown("#### Model Training Time (seconds)")
            
            training_times = [metrics['training_time'] for metrics in model_metrics.values()]
            model_names = list(model_metrics.keys())
            
            fig_time = px.bar(
                x=model_names, y=training_times,
                title="<b>Training Time Comparison</b>",
                labels={'x': '<b>Models</b>', 'y': '<b>Training Time (seconds)</b>'},
                color=training_times,
                color_continuous_scale='Blues'
            )
            fig_time.update_layout(
                title={
                    'text': "<b>Training Time Comparison</b>",
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            # 6. Cross-Validation Results Section
            st.markdown("### 📋 Cross-Validation Results")
            
            # Add a button for cross-validation analysis
            if st.button("🔬 Run Cross-Validation Analysis", type="secondary"):
                with st.spinner("Running cross-validation analysis..."):
                    # Perform cross-validation for each model
                    cv_results = {}
                    for model_name, model in trained_models.items():
                        try:
                            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
                            cv_results[model_name] = {
                                'mean_cv_score': cv_scores.mean(),
                                'std_cv_score': cv_scores.std(),
                                'cv_scores': cv_scores
                            }
                        except Exception as e:
                            st.warning(f"⚠️ Cross-validation failed for {model_name}: {str(e)}")
                    
                    if cv_results:
                        # Display cross-validation results
                        cv_df = pd.DataFrame([
                            {
                                '**Model**': f"**{name}**",
                                '**Mean CV Score**': f"**{results['mean_cv_score']:.4f}**",
                                '**Std CV Score**': f"**{results['std_cv_score']:.4f}**",
                                '**CV Range**': f"**{results['mean_cv_score'] - results['std_cv_score']:.4f} - {results['mean_cv_score'] + results['std_cv_score']:.4f}**"
                            }
                            for name, results in cv_results.items()
                        ])
                        
                        st.dataframe(cv_df, use_container_width=True)
                        
                        # Create CV comparison chart
                        cv_names = list(cv_results.keys())
                        cv_means = [cv_results[name]['mean_cv_score'] for name in cv_names]
                        cv_stds = [cv_results[name]['std_cv_score'] for name in cv_names]
                        
                        fig_cv = go.Figure()
                        fig_cv.add_trace(go.Bar(
                            x=cv_names,
                            y=cv_means,
                            error_y=dict(type='data', array=cv_stds, visible=True),
                            name='<b>Cross-Validation Score</b>',
                            marker_color='#ff6b6b'
                        ))
                        fig_cv.update_layout(
                            title="<b>Cross-Validation Results Comparison</b>",
                            xaxis_title="<b>Models</b>",
                            yaxis_title="<b>Accuracy Score</b>",
                            height=400
                        )
                        st.plotly_chart(fig_cv, use_container_width=True)

    def show_prediction_page(trained_models, scaler, feature_columns, model_metrics):
        """Display the prediction interface"""
        st.markdown('<h2 class="sub-header">🔮 Dengue Fever Prediction</h2>', unsafe_allow_html=True)
        
        if trained_models is None:
            st.error("❌ Models not available. Please ensure models are trained.")
            return
        
        # Best model for prediction
        best_model_name = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = trained_models[best_model_name]
        
        st.markdown(f"""
        <div class="info-box">
            <strong>ℹ️ Using Model:</strong> {best_model_name} (Accuracy: {model_metrics[best_model_name]['accuracy']:.2%})
        </div>
        """, unsafe_allow_html=True)
        
        # Input form
        st.markdown("### 📋 Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔥 Fever Symptoms")
            fever = st.selectbox("Fever", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", 
                               help="Presence of fever (temperature ≥38°C)")
            fever_duration = st.slider("Fever Duration (Days)", min_value=-5, max_value=15, value=3, 
                                     help="Duration of fever in days (negative for days before fever)")
            
            st.markdown("#### 🧠 Neurological Symptoms")
            headache = st.selectbox("Severe Headache", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                  help="Presence of severe headache")
            retro_orbital_pain = st.selectbox("Retro-orbital Pain", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                            help="Pain behind the eyes")
            
            st.markdown("#### 💪 Musculoskeletal Symptoms")
            muscle_joint_pain = st.selectbox("Muscle and Joint Pain", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                           help="Pain in muscles and joints")
            
            st.markdown("#### 🩸 Bleeding Symptoms")
            bleeding = st.selectbox("Bleeding Manifestations", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                  help="Any bleeding manifestations (nosebleed, gum bleeding, etc.)")
        
        with col2:
            st.markdown("#### 🧪 Laboratory Tests")
            platelet_count = st.slider("Platelet Count (per μL)", min_value=20000, max_value=500000, value=200000,
                                     help="Platelet count in blood (Normal: 150,000-450,000)")
            wbc_count = st.slider("White Blood Cell Count (per μL)", min_value=1000, max_value=20000, value=7000,
                                help="White blood cell count (Normal: 4,000-11,000)")
            
            st.markdown("#### 🔬 Specific Tests")
            ns1_test = st.selectbox("NS1 Antigen Test", options=[0, 1], format_func=lambda x: "Negative" if x == 0 else "Positive",
                                   help="NS1 antigen test result")
            igm_igg_test = st.selectbox("IgM/IgG Test", options=[0, 1], format_func=lambda x: "Negative" if x == 0 else "Positive",
                                       help="IgM/IgG antibody test result")
            
            st.markdown("#### 🤢 Gastrointestinal Symptoms")
            vomiting_nausea = st.selectbox("Vomiting/Nausea", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                         help="Presence of vomiting or nausea")
            abdominal_pain = st.selectbox("Abdominal Pain", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                        help="Presence of abdominal pain")
            
            st.markdown("#### 🩹 Skin Symptoms")
            skin_rash = st.selectbox("Skin Rash", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                   help="Presence of skin rash")
        
        # Prediction button
        if st.button("🔮 Predict Dengue Fever", type="primary"):
            # Prepare input data
            input_data = {
                'Fever': fever,
                'Fever_Duration_Days': fever_duration,
                'Headache': headache,
                'Retro_Orbital_Pain': retro_orbital_pain,
                'Muscle_Joint_Pain': muscle_joint_pain,
                'Skin_Rash': skin_rash,
                'Bleeding': bleeding,
                'Vomiting_Nausea': vomiting_nausea,
                'Abdominal_Pain': abdominal_pain,
                'Platelet_Count': platelet_count,
                'WBC_Count': wbc_count,
                'NS1_Antigen_Test': ns1_test,
                'IgM_IgG_Test': igm_igg_test
            }
            
            # Preprocess input
            processed_input = preprocess_input(input_data, scaler, feature_columns)
            
            # Make prediction with timing
            start_time = time.time()
            prediction = best_model.predict(processed_input)[0]
            prediction_proba = best_model.predict_proba(processed_input)[0][1] if hasattr(best_model, 'predict_proba') else None
            prediction_time = time.time() - start_time
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div class="prediction-card danger-prediction">
                        <h2>⚠️ HIGH RISK</h2>
                        <h3>Dengue Fever Detected</h3>
                        <p>Please consult a healthcare professional immediately.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-card safe-prediction">
                        <h2>✅ LOW RISK</h2>
                        <h3>No Dengue Fever Detected</h3>
                        <p>Continue monitoring symptoms.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if prediction_proba is not None:
                    confidence = prediction_proba if prediction == 1 else (1 - prediction_proba)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Confidence Level</h3>
                        <h2>{confidence:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Prediction Time</h3>
                    <h2>{prediction_time:.3f}s</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed probability
            if prediction_proba is not None:
                st.markdown("### 📊 Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = prediction_proba * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Probability of Dengue Fever"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk factors analysis
                    st.markdown("#### 🚨 Risk Factors Analysis")
                    
                    risk_factors = []
                    risk_scores = []
                    
                    if fever == 1:
                        risk_factors.append("High fever present")
                        risk_scores.append(25)
                    if headache == 1:
                        risk_factors.append("Severe headache")
                        risk_scores.append(15)
                    if retro_orbital_pain == 1:
                        risk_factors.append("Retro-orbital pain")
                        risk_scores.append(20)
                    if bleeding == 1:
                        risk_factors.append("Bleeding manifestations")
                        risk_scores.append(30)
                    if platelet_count < 150000:
                        risk_factors.append("Low platelet count")
                        risk_scores.append(25)
                    if wbc_count < 4000:
                        risk_factors.append("Low white blood cell count")
                        risk_scores.append(15)
                    if ns1_test == 1:
                        risk_factors.append("Positive NS1 antigen test")
                        risk_scores.append(40)
                    if igm_igg_test == 1:
                        risk_factors.append("Positive IgM/IgG test")
                        risk_scores.append(35)
                    
                    if risk_factors:
                        # Create risk factors visualization
                        risk_df = pd.DataFrame({
                            'Risk Factor': risk_factors,
                            'Risk Score': risk_scores
                        })
                        
                        fig_risk = px.bar(
                            risk_df, x='Risk Score', y='Risk Factor',
                            title="Identified Risk Factors",
                            orientation='h',
                            color='Risk Score',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)
                    else:
                        st.markdown("✅ No significant risk factors identified")
                
                # Model comparison for this prediction
                st.markdown("### 🤖 Multi-Model Prediction Comparison")
                
                # Get predictions from all models
                model_predictions = {}
                model_probabilities = {}
                
                for model_name, model in trained_models.items():
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(processed_input)[0][1]
                        pred = model.predict(processed_input)[0]
                        model_predictions[model_name] = pred
                        model_probabilities[model_name] = prob
                
                if model_probabilities:
                    # Create comparison chart
                    model_names = list(model_probabilities.keys())
                    probabilities = list(model_probabilities.values())
                    
                    fig_comparison = px.bar(
                        x=model_names, y=probabilities,
                        title="Dengue Probability by Different Models",
                        labels={'x': 'Models', 'y': 'Probability'},
                        color=probabilities,
                        color_continuous_scale='Reds'
                    )
                    fig_comparison.update_layout(height=400)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Model agreement analysis
                    st.markdown("#### 📋 Model Agreement Analysis")
                    
                    agreement_count = sum(1 for pred in model_predictions.values() if pred == prediction)
                    total_models = len(model_predictions)
                    agreement_percentage = (agreement_count / total_models) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Models in Agreement", f"{agreement_count}/{total_models}")
                    
                    with col2:
                        st.metric("Agreement Rate", f"{agreement_percentage:.1f}%")
                    
                    with col3:
                        confidence_level = "High" if agreement_percentage >= 80 else "Medium" if agreement_percentage >= 60 else "Low"
                        st.metric("Confidence Level", confidence_level)
                    
                    # Detailed model predictions table
                    st.markdown("#### 📊 Detailed Model Predictions")
                    
                    prediction_details = []
                    for model_name in model_names:
                        pred = model_predictions[model_name]
                        prob = model_probabilities[model_name]
                        accuracy = model_metrics[model_name]['accuracy']
                        
                        prediction_details.append({
                            'Model': model_name,
                            'Prediction': 'Dengue' if pred == 1 else 'Healthy',
                            'Probability': f"{prob:.3f}",
                            'Model Accuracy': f"{accuracy:.3f}",
                            'Agreement': '✅' if pred == prediction else '❌'
                        })
                    
                    prediction_df = pd.DataFrame(prediction_details)
                    st.dataframe(prediction_df, use_container_width=True)
                
                # Recommendations based on prediction
                st.markdown("### 💡 Recommendations")
                
                if prediction == 1:
                    st.markdown("""
                    **🚨 Immediate Actions Required:**
                    
                    1. **Seek Medical Attention**: Consult a healthcare professional immediately
                    2. **Monitor Symptoms**: Keep track of fever, bleeding, and other symptoms
                    3. **Rest and Hydration**: Ensure adequate rest and fluid intake
                    4. **Avoid Medications**: Do not take aspirin or NSAIDs
                    5. **Follow-up**: Schedule follow-up appointments as recommended
                    
                    **⚠️ Warning Signs to Watch For:**
                    - Severe abdominal pain
                    - Persistent vomiting
                    - Bleeding from nose or gums
                    - Difficulty breathing
                    - Cold, clammy skin
                    """)
                else:
                    st.markdown("""
                    **✅ Recommended Actions:**
                    
                    1. **Continue Monitoring**: Keep track of any new symptoms
                    2. **Stay Hydrated**: Maintain adequate fluid intake
                    3. **Rest**: Get sufficient rest to support recovery
                    4. **Follow-up**: Schedule follow-up if symptoms persist
                    5. **Prevention**: Take measures to prevent mosquito bites
                    
                    **🔍 When to Seek Medical Attention:**
                    - Symptoms worsen or persist
                    - New symptoms develop
                    - High fever continues
                    - Signs of dehydration
                    """)

    def show_data_analysis(df):
        """Display comprehensive data analysis and visualizations"""
        st.markdown('<h2 class="sub-header">📈 Comprehensive Data Analysis & Insights</h2>', unsafe_allow_html=True)
        
        # Basic statistics
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", f"{len(df):,}")
        with col2:
            dengue_cases = df['Diagnosis'].sum()
            st.metric("Dengue Cases", f"{dengue_cases:,}")
        with col3:
            healthy_cases = len(df) - dengue_cases
            st.metric("Healthy Cases", f"{healthy_cases:,}")
        with col4:
            dengue_rate = df['Diagnosis'].mean()
            st.metric("Dengue Rate", f"{dengue_rate:.1%}")
        
        if PLOTLY_AVAILABLE:
            # 1. Diagnosis Distribution Pie Chart
            st.markdown("### 🥧 Diagnosis Distribution")
            diagnosis_counts = df['Diagnosis'].value_counts()
            fig_pie = px.pie(
                values=diagnosis_counts.values,
                names=['Healthy', 'Dengue Fever'],
                title="Distribution of Diagnosis",
                color_discrete_sequence=['#4facfe', '#ff6b6b']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # 2. Symptom Prevalence Analysis
            st.markdown("### 🔥 Symptom Prevalence Analysis")
            symptoms = ['Fever', 'Headache', 'Retro_Orbital_Pain', 'Muscle_Joint_Pain', 
                       'Skin_Rash', 'Bleeding', 'Vomiting_Nausea', 'Abdominal_Pain']
            
            # Calculate symptom prevalence by diagnosis
            symptom_data = []
            for symptom in symptoms:
                dengue_positive = df[df['Diagnosis'] == 1][symptom].mean() * 100
                dengue_negative = df[df['Diagnosis'] == 0][symptom].mean() * 100
                symptom_data.append([symptom, dengue_positive, dengue_negative])
            
            symptom_df = pd.DataFrame(symptom_data, columns=['Symptom', 'Dengue Positive (%)', 'Dengue Negative (%)'])
            
            # Create horizontal bar chart
            fig_symptoms = px.bar(
                symptom_df, 
                x=['Dengue Positive (%)', 'Dengue Negative (%)'],
                y='Symptom',
                title="Symptom Prevalence by Diagnosis Status",
                barmode='group',
                color_discrete_sequence=['#ff6b6b', '#4facfe']
            )
            fig_symptoms.update_layout(xaxis_title="Prevalence (%)", yaxis_title="Symptoms")
            st.plotly_chart(fig_symptoms, use_container_width=True)
            
            # 3. Laboratory Values Analysis
            st.markdown("### 🧪 Laboratory Values Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Platelet Count Distribution
                fig_platelet = px.histogram(
                    df, x='Platelet_Count', color='Diagnosis',
                    title="Platelet Count Distribution",
                    labels={'Diagnosis': 'Dengue Fever', 'Platelet_Count': 'Platelet Count'},
                    color_discrete_sequence=['#4facfe', '#ff6b6b'],
                    nbins=30
                )
                fig_platelet.add_vline(x=150000, line_dash="dash", line_color="red", 
                                     annotation_text="Normal Range (150,000)")
                st.plotly_chart(fig_platelet, use_container_width=True)
            
            with col2:
                # WBC Count Distribution
                fig_wbc = px.histogram(
                    df, x='WBC_Count', color='Diagnosis',
                    title="White Blood Cell Count Distribution",
                    labels={'Diagnosis': 'Dengue Fever', 'WBC_Count': 'WBC Count'},
                    color_discrete_sequence=['#4facfe', '#ff6b6b'],
                    nbins=30
                )
                fig_wbc.add_vline(x=4000, line_dash="dash", line_color="red", 
                                annotation_text="Low WBC Threshold (4,000)")
                st.plotly_chart(fig_wbc, use_container_width=True)
            
            # 4. Correlation Heatmap
            st.markdown("### 🔗 Feature Correlation Analysis")
            
            # Select numerical features for correlation
            numerical_features = ['Fever_Duration_Days', 'Platelet_Count', 'WBC_Count', 'Diagnosis']
            correlation_matrix = df[numerical_features].corr()
            
            fig_heatmap = px.imshow(
                correlation_matrix,
                title="Feature Correlation Heatmap",
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 5. Fever Duration Analysis
            st.markdown("### 🌡️ Fever Duration Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Fever duration by diagnosis
                fig_fever_duration = px.box(
                    df, x='Diagnosis', y='Fever_Duration_Days',
                    title="Fever Duration by Diagnosis",
                    labels={'Diagnosis': 'Dengue Fever', 'Fever_Duration_Days': 'Fever Duration (Days)'},
                    color_discrete_sequence=['#4facfe', '#ff6b6b']
                )
                st.plotly_chart(fig_fever_duration, use_container_width=True)
            
            with col2:
                # Fever duration distribution
                fig_fever_hist = px.histogram(
                    df, x='Fever_Duration_Days', color='Diagnosis',
                    title="Fever Duration Distribution",
                    labels={'Diagnosis': 'Dengue Fever', 'Fever_Duration_Days': 'Fever Duration (Days)'},
                    color_discrete_sequence=['#4facfe', '#ff6b6b'],
                    nbins=20
                )
                st.plotly_chart(fig_fever_hist, use_container_width=True)
            
            # 6. Test Results Analysis
            st.markdown("### 🔬 Diagnostic Test Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # NS1 Test Results
                ns1_counts = df.groupby(['NS1_Antigen_Test', 'Diagnosis']).size().unstack(fill_value=0)
                fig_ns1 = px.bar(
                    ns1_counts, 
                    title="NS1 Antigen Test Results by Diagnosis",
                    color_discrete_sequence=['#4facfe', '#ff6b6b']
                )
                fig_ns1.update_layout(xaxis_title="NS1 Test Result", yaxis_title="Count")
                st.plotly_chart(fig_ns1, use_container_width=True)
            
            with col2:
                # IgM/IgG Test Results
                igm_counts = df.groupby(['IgM_IgG_Test', 'Diagnosis']).size().unstack(fill_value=0)
                fig_igm = px.bar(
                    igm_counts,
                    title="IgM/IgG Test Results by Diagnosis",
                    color_discrete_sequence=['#4facfe', '#ff6b6b']
                )
                fig_igm.update_layout(xaxis_title="IgM/IgG Test Result", yaxis_title="Count")
                st.plotly_chart(fig_igm, use_container_width=True)
            
            # 7. Risk Factor Analysis
            st.markdown("### ⚠️ Risk Factor Analysis")
            
            # Calculate risk factors
            risk_factors = {
                'Low Platelet Count (<150k)': (df['Platelet_Count'] < 150000).sum(),
                'Low WBC Count (<4k)': (df['WBC_Count'] < 4000).sum(),
                'Positive NS1 Test': df['NS1_Antigen_Test'].sum(),
                'Positive IgM/IgG Test': df['IgM_IgG_Test'].sum(),
                'Bleeding Manifestations': df['Bleeding'].sum(),
                'Retro-orbital Pain': df['Retro_Orbital_Pain'].sum()
            }
            
            risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Risk Factor', 'Count'])
            risk_df['Percentage'] = (risk_df['Count'] / len(df)) * 100
            
            fig_risk = px.bar(
                risk_df, x='Risk Factor', y='Percentage',
                title="Prevalence of Risk Factors in Dataset",
                color='Percentage',
                color_continuous_scale='Reds'
            )
            fig_risk.update_layout(xaxis_title="Risk Factors", yaxis_title="Percentage (%)")
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # 8. Age Group Analysis (if age data available)
            st.markdown("### 📊 Demographic Analysis")
            
            # Create age groups if age column exists, otherwise show other demographics
            if 'Age' in df.columns:
                df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], 
                                       labels=['0-18', '19-35', '36-50', '51-65', '65+'])
                
                age_diagnosis = df.groupby(['Age_Group', 'Diagnosis']).size().unstack(fill_value=0)
                fig_age = px.bar(
                    age_diagnosis,
                    title="Dengue Cases by Age Group",
                    color_discrete_sequence=['#4facfe', '#ff6b6b']
                )
                fig_age.update_layout(xaxis_title="Age Group", yaxis_title="Count")
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                # Show symptom combinations instead
                st.markdown("#### 🔥 Common Symptom Combinations")
                
                # Calculate symptom combinations
                df['Symptom_Count'] = (df['Fever'] + df['Headache'] + df['Retro_Orbital_Pain'] + 
                                     df['Muscle_Joint_Pain'] + df['Skin_Rash'] + df['Bleeding'] + 
                                     df['Vomiting_Nausea'] + df['Abdominal_Pain'])
                
                symptom_count_analysis = df.groupby(['Symptom_Count', 'Diagnosis']).size().unstack(fill_value=0)
                fig_symptom_count = px.bar(
                    symptom_count_analysis,
                    title="Dengue Cases by Number of Symptoms",
                    color_discrete_sequence=['#4facfe', '#ff6b6b']
                )
                fig_symptom_count.update_layout(xaxis_title="Number of Symptoms", yaxis_title="Count")
                st.plotly_chart(fig_symptom_count, use_container_width=True)
            
            # 9. Interactive Scatter Plot
            st.markdown("### 📍 Interactive Scatter Plot: Platelet vs WBC")
            
            fig_scatter = px.scatter(
                df, x='Platelet_Count', y='WBC_Count', color='Diagnosis',
                title="Platelet Count vs White Blood Cell Count",
                labels={'Diagnosis': 'Dengue Fever', 'Platelet_Count': 'Platelet Count', 'WBC_Count': 'WBC Count'},
                color_discrete_sequence=['#4facfe', '#ff6b6b'],
                hover_data=['Fever', 'Headache', 'Bleeding']
            )
            
            # Add reference lines
            fig_scatter.add_hline(y=4000, line_dash="dash", line_color="orange", 
                                annotation_text="Low WBC Threshold")
            fig_scatter.add_vline(x=150000, line_dash="dash", line_color="red", 
                                annotation_text="Low Platelet Threshold")
            
            st.plotly_chart(fig_scatter, use_container_width=True)

    if __name__ == "__main__":
        main()

def heart_disease_main():
    """Heart Disease Prediction Module"""
    # Main header
    st.markdown('<h1 class="main-header">❤️ Heart Disease</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Model Performance", "🔮 Make Prediction", "📈 Data Analysis"]
    )
    
    # Train models (cached)
    trained_models, model_metrics, encoder, scaler, feature_columns, X_test, y_test = train_models(df)
    
    if page == "🏠 Home":
        show_home_page(df)
    
    elif page == "📊 Model Performance":
        show_model_performance(model_metrics, trained_models, X_test, y_test)
    
    elif page == "🔮 Make Prediction":
        show_prediction_page(trained_models, encoder, scaler, feature_columns, model_metrics)
    
    elif page == "📈 Data Analysis":
        show_data_analysis(df)

# --- Unified App Entry Point ---
def unified_main():
    st.sidebar.title("Select Disease Module")
    # Change from radio to selectbox for disease selection
    module = st.sidebar.selectbox("Choose a prediction system:", ("Heart Disease", "Diabetes", "Dengue Fever"))
    if module == "Heart Disease":
        heart_disease_main()
    elif module == "Diabetes":
        diabetes_main()
    elif module == "Dengue Fever":
        dengue_main()

if __name__ == "__main__":
    unified_main()
