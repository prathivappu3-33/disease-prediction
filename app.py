import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
from config import GROQ_API_KEY
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="🏥 Disease Predictor", layout="wide")
st.title("🏥 AI Disease Prediction System")
st.markdown("### Select your symptoms and get an AI-powered health assessment")

# Disease-Symptom database
DISEASE_DATA = {
    "Common Cold": ["runny_nose", "sneezing", "sore_throat", "cough", "fatigue", "mild_fever"],
    "Flu": ["high_fever", "body_aches", "fatigue", "cough", "headache", "chills", "sore_throat"],
    "COVID-19": ["fever", "dry_cough", "fatigue", "loss_of_taste", "loss_of_smell", "breathlessness", "body_aches"],
    "Dengue": ["high_fever", "severe_headache", "joint_pain", "rash", "nausea", "vomiting", "eye_pain"],
    "Typhoid": ["high_fever", "stomach_pain", "headache", "fatigue", "constipation", "rash", "nausea"],
    "Malaria": ["high_fever", "chills", "sweating", "headache", "nausea", "vomiting", "body_aches"],
    "Pneumonia": ["high_fever", "cough", "breathlessness", "chest_pain", "fatigue", "chills"],
    "Diabetes": ["frequent_urination", "excessive_thirst", "fatigue", "blurred_vision", "slow_healing", "weight_loss"],
    "Hypertension": ["headache", "dizziness", "blurred_vision", "chest_pain", "fatigue", "nausea"],
    "Gastroenteritis": ["nausea", "vomiting", "diarrhea", "stomach_pain", "fever", "fatigue"],
    "Migraine": ["severe_headache", "nausea", "vomiting", "blurred_vision", "sensitivity_to_light"],
    "Asthma": ["breathlessness", "cough", "wheezing", "chest_tightness", "fatigue"],
    "Anemia": ["fatigue", "dizziness", "pale_skin", "breathlessness", "cold_hands", "headache"],
    "UTI": ["frequent_urination", "burning_urination", "pelvic_pain", "fever", "nausea"],
    "Appendicitis": ["stomach_pain", "nausea", "vomiting", "high_fever", "loss_of_appetite"],
}

ALL_SYMPTOMS = sorted(set(s for symptoms in DISEASE_DATA.values() for s in symptoms))
SYMPTOM_LABELS = {s: s.replace("_", " ").title() for s in ALL_SYMPTOMS}

def build_model():
    X, y = [], []
    for disease, symptoms in DISEASE_DATA.items():
        for _ in range(10):
            row = [1 if s in symptoms and np.random.random() > 0.2 else 0 for s in ALL_SYMPTOMS]
            X.append(row)
            y.append(disease)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

@st.cache_resource
def get_model():
    return build_model()

model = get_model()

def predict_disease(selected_symptoms):
    row = [1 if s in selected_symptoms else 0 for s in ALL_SYMPTOMS]
    proba = model.predict_proba([row])[0]
    classes = model.classes_
    top_indices = np.argsort(proba)[::-1][:3]
    return [(classes[i], proba[i]) for i in top_indices]

def get_ai_advice(disease, symptoms):
    prompt = f"""A patient has these symptoms: {', '.join(symptoms)}.
The AI model predicts the most likely condition is: {disease}.

Please provide:
1. 📋 Brief explanation of {disease}
2. ⚠️ Warning signs to watch for
3. 💊 General precautions and home remedies
4. 🏥 When to see a doctor immediately

Keep it concise, clear and helpful. Add a disclaimer that this is not medical advice."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Sidebar
st.sidebar.header("⚙️ About")
st.sidebar.info("This tool uses a Random Forest ML model trained on disease-symptom patterns + Groq AI for health advice.")
st.sidebar.warning("⚠️ This is NOT a substitute for professional medical advice.")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Rohith V** | [🔗 LinkedIn](https://www.linkedin.com/in/rohithv0507/)")

# Symptom selection
st.markdown("## 🤒 Select Your Symptoms")
cols = st.columns(3)
selected = []
for i, symptom in enumerate(ALL_SYMPTOMS):
    with cols[i % 3]:
        if st.checkbox(SYMPTOM_LABELS[symptom], key=symptom):
            selected.append(symptom)

st.markdown(f"**Selected symptoms:** {len(selected)}")

if st.button("🔍 Predict Disease", use_container_width=True, disabled=len(selected) == 0):
    if len(selected) < 2:
        st.warning("⚠️ Please select at least 2 symptoms for better accuracy.")
    else:
        with st.spinner("🤖 Analyzing symptoms..."):
            predictions = predict_disease(selected)

        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        col1, col2, col3 = st.columns(3)
        for i, (disease, prob) in enumerate(predictions):
            with [col1, col2, col3][i]:
                if i == 0:
                    st.error(f"### 🥇 {disease}")
                elif i == 1:
                    st.warning(f"### 🥈 {disease}")
                else:
                    st.info(f"### 🥉 {disease}")
                st.progress(prob)
                st.write(f"Confidence: **{prob*100:.1f}%**")

        top_disease = predictions[0][0]
        st.markdown("---")
        st.markdown(f"## 🧠 AI Health Advice for: **{top_disease}**")
        with st.spinner("Getting AI health advice..."):
            advice = get_ai_advice(top_disease, selected)
            st.markdown(advice)

elif len(selected) == 0:
    st.info("👆 Select your symptoms above and click Predict Disease!")
