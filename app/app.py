# ============================================
# IPL Win Probability Predictor
# Streamlit Web Application
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import time
import os

# ── Page config (MUST be first Streamlit command) ──
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

with st.sidebar:

    st.title("🏏 IPL Predictor")

    st.markdown("""
### About

Predicts chasing team's winning probability using an XGBoost model trained on IPL historical matches.

### Model

Accuracy: **77.95%**

Dataset:

IPL 2008–2019
""")

# ── Custom styling ──
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0a0a0a;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a27 50%, #1a472a 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
        border: 2px solid #4CAF50;
    }
    
    /* Team card styling */
    .team-card {
        background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
        margin-bottom: 15px;
    }
    
    /* Metric card */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #0f3460;
    }
    
    /* Win probability display */
    .win-prob-high {
        color: #4CAF50;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
    }
    
    .win-prob-medium {
        color: #FF9800;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
    }
    
    .win-prob-low {
        color: #F44336;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
    }
    
    /* Factor cards */
    .factor-good {
        background: rgba(76, 175, 80, 0.15);
        border-left: 4px solid #4CAF50;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 5px 0;
        color: #ffffff;
    }
    
    .factor-warning {
        background: rgba(255, 152, 0, 0.15);
        border-left: 4px solid #FF9800;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 5px 0;
        color: #ffffff;
    }
    
    .factor-bad {
        background: rgba(244, 67, 54, 0.15);
        border-left: 4px solid #F44336;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 5px 0;
        color: #ffffff;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1a472a, #4CAF50);
        color: white;
        border: none;
        padding: 15px 40px;
        font-size: 18px;
        border-radius: 10px;
        width: 100%;
        font-weight: bold;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4CAF50, #1a472a);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ── Load model and supporting data ──

from pathlib import Path
@st.cache_resource
def load_model():
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "models" / "win_predictor_model.pkl"

    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_columns():
    BASE_DIR = Path(__file__).resolve().parent.parent
    cols_path = BASE_DIR / "models" / "model_columns.pkl"

    with open(cols_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_match_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "model_ready_data.csv"

    return pd.read_csv(data_path)

model = load_model()
model_columns = load_columns()
match_data = load_match_data()

# Get all unique teams from data
all_teams = sorted(match_data['batting_team'].unique().tolist())

# ── App Header ──
st.markdown("""
<div class="main-header">
    <h1 style="color: #ffffff; font-size: 2.5em; margin: 0;">
        🏏 IPL Win Probability Predictor
    </h1>
    <p style="color: #a8d5a2; font-size: 1.1em; margin-top: 10px;">
        Powered by XGBoost • Trained on 15 Years of IPL Data (2008-2019) • 77.95% Accuracy
    </p>
</div>
""", unsafe_allow_html=True)


with st.expander("ℹ️ About this Model"):
    st.markdown("""
### Model Information

- **Algorithm:** XGBoost Classifier
- **Training Data:** IPL Matches (2008–2019)
- **Validation:** 5-Fold Group Cross Validation
- **Accuracy:** 77.95%
- **Prediction Target:** Win probability of the chasing team

### Features Used

- Current Score
- Wickets Fallen
- Balls Remaining
- Runs Remaining
- Current Run Rate
- Required Run Rate
- Batting Team
- Bowling Team
""")


# ── Input Section ──
st.markdown("### 🎯 Match Situation Input")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    batting_team = st.selectbox(
        "🏏 Batting Team (Chasing)",
        all_teams,
        index=None,
         placeholder="Select Batting Team"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    bowling_team = st.selectbox(
        "🎳 Bowling Team (Defending)",
        all_teams,
        index=None,
        placeholder="Select Bowling Team"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Match situation inputs
st.markdown("### 📊 Current Match Status")
col3, col4, col5, col6 = st.columns(4)

with col3:
    target = st.number_input(
        "🎯 Target (Runs to Win)",
        min_value=1, max_value=300,
        value=180, step=1
    )

with col4:
    current_score = st.number_input(
        "📈 Current Score",
        min_value=0, max_value=target-1,
        value=100, step=1
    )

with col5:
    overs_completed = st.number_input(
        "⏱️ Overs Completed",
        min_value=0.0, max_value=19.5,
        value=12.0, step=0.1,
        format="%.1f"
    )

with col6:
    wickets = st.number_input(
        "🚨 Wickets Fallen",
        min_value=0, max_value=10,
        value=3, step=1
    )


# ── Prediction Logic ──

def calculate_features(batting_team, bowling_team, target, 
                        current_score, overs_completed, wickets):
    """Convert match situation inputs into model features"""
    
    # Calculate derived features
    balls_part = round((overs_completed % 1) * 10)
    if balls_part > 5:
        st.error("Invalid overs. Balls can only be 0-5.")
        st.stop()
    balls_bowled = int(overs_completed) * 6 + balls_part
    balls_remaining = max(120 - balls_bowled, 1)
    runs_remaining = max(target - current_score, 0)
    
    # Run rates
    if balls_bowled > 0:
        current_run_rate = (current_score * 6) / balls_bowled
    else:
        current_run_rate = 0.0
    
    required_run_rate = (runs_remaining * 6) / balls_remaining
    
    # Build feature dictionary
    features = {
        'current_score': current_score,
        'wicket_fallen': wickets,
        'balls_remaining': balls_remaining,
        'runs_remaining': runs_remaining,
        'current_run_rate': round(current_run_rate, 4),
        'required_run_rate': round(required_run_rate, 4)
    }
    
    # Create DataFrame with all model columns
    input_df = pd.DataFrame([features])
    
    # Add all team columns as 0
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Set correct team columns to 1
    batting_col = f'batting_team_{batting_team}'
    bowling_col = f'bowling_team_{bowling_team}'
    
    if batting_col in input_df.columns:
        input_df[batting_col] = 1
    if bowling_col in input_df.columns:
        input_df[bowling_col] = 1
    
    # Ensure exact column order as training
    input_df = input_df[model_columns]
    
    return input_df, features


def get_match_factors(features, win_prob):
    """Generate human-readable match situation analysis"""
    factors = []
    
    rrr = features['required_run_rate']
    crr = features['current_run_rate']
    wickets = features['wicket_fallen']
    balls_rem = features['balls_remaining']
    
    # Required run rate analysis
    if rrr < 6:
        factors.append(('good', f'Required run rate ({rrr:.1f}) is very comfortable'))
    elif rrr < 8:
        factors.append(('good', f'Required run rate ({rrr:.1f}) is achievable'))
    elif rrr < 10:
        factors.append(('warning', f'Required run rate ({rrr:.1f}) is challenging'))
    elif rrr < 13:
        factors.append(('bad', f'Required run rate ({rrr:.1f}) is very difficult'))
    else:
        factors.append(('bad', f'Required run rate ({rrr:.1f}) is almost impossible'))
    
    # Run rate comparison
    rate_diff = crr - rrr
    if rate_diff > 1:
        factors.append(('good', f'Scoring faster than required (+{rate_diff:.1f} run rate advantage)'))
    elif rate_diff > -1:
        factors.append(('warning', f'Scoring at par with required rate ({rate_diff:+.1f})'))
    else:
        factors.append(('bad', f'Falling behind required rate ({rate_diff:.1f} run rate deficit)'))
    
    # Wickets analysis
    wickets_remaining = 10 - wickets
    if wickets_remaining >= 7:
        factors.append(('good', f'{wickets_remaining} wickets in hand — strong batting depth'))
    elif wickets_remaining >= 5:
        factors.append(('warning', f'{wickets_remaining} wickets remaining — manageable'))
    elif wickets_remaining >= 3:
        factors.append(('bad', f'Only {wickets_remaining} wickets left — tail exposed'))
    else:
        factors.append(('bad', f'Only {wickets_remaining} wickets left — match nearly over'))
    
    # Overs remaining
    overs_remaining = balls_rem / 6
    if overs_remaining >= 10:
        factors.append(('good', f'{overs_remaining:.1f} overs remaining — plenty of time'))
    elif overs_remaining >= 5:
        factors.append(('warning', f'{overs_remaining:.1f} overs remaining — entering death overs'))
    else:
        factors.append(('bad', f'Only {overs_remaining:.1f} overs left — crunch time'))


    if win_prob >= 0.80:
        factors.append(("good", "Model strongly favors the batting team"))
    elif win_prob >= 0.60:
        factors.append(("good", "Batting team has the advantage"))
    elif win_prob >= 0.40:
        factors.append(("warning", "Match is evenly balanced"))
    elif win_prob >= 0.20:
        factors.append(("bad", "Bowling team currently has the edge"))
    else:
        factors.append(("bad", "Model strongly favors the bowling team"))
    
    return factors


def create_gauge_chart(win_prob, batting_team):
    """Create a professional win probability gauge"""
    
    # Color based on probability
    if win_prob >= 0.65:
        color = "#4CAF50"
        status = "LIKELY WIN"
    elif win_prob >= 0.45:
        color = "#FF9800"  
        status = "CONTEST"
    else:
        color = "#F44336"
        status = "UNDER PRESSURE"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=win_prob * 100,
        number={
            'suffix': '%',
            'font': {'size': 48, 'color': color}
        },
        title={
            'text': f"{batting_team}<br><span style='font-size:0.7em;color:{color}'>{status}</span>",
            'font': {'size': 18, 'color': '#ffffff'}
        },
        delta={
            'reference': 50,
            'increasing': {'color': '#4CAF50'},
            'decreasing': {'color': '#F44336'}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': '#ffffff',
                'tickfont': {'color': '#ffffff'}
            },
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': '#1e1e1e',
            'borderwidth': 2,
            'bordercolor': '#333',
            'steps': [
                {'range': [0, 35],   'color': 'rgba(244,67,54,0.2)'},
                {'range': [35, 50],  'color': 'rgba(255,152,0,0.15)'},
                {'range': [50, 65],  'color': 'rgba(255,152,0,0.15)'},
                {'range': [65, 100], 'color': 'rgba(76,175,80,0.2)'}
            ],
            'threshold': {
                'line': {'color': '#ffffff', 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


# ── Predict Button ──
st.markdown("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_clicked = st.button("🏏 Predict Win Probability", use_container_width=True)

# ── Results ──
if predict_clicked:
    
    # Validation
    if batting_team is None or bowling_team is None:
        st.error("Please select both teams.")
        st.stop()
    if batting_team == bowling_team:
        st.error("⚠️ Batting team and bowling team cannot be the same!")
        st.stop()
    
    if current_score >= target:
        st.error("⚠️ Current score cannot exceed or equal the target!")
        st.stop()
    
    if wickets == 10:
        st.error("⚠️ All wickets fallen — innings is over!")
        st.stop()
    
    # Show loading spinner
    with st.spinner("Calculating win probability..."):
        input_df, features = calculate_features(
            batting_team, bowling_team, target,
            current_score, overs_completed, wickets
        )
        win_prob = model.predict_proba(input_df)[0][1]
        loss_prob = 1 - win_prob
    
    # Results layout
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    result_col1, result_col2 = st.columns([1.2, 1])
    
    with result_col1:
        # Gauge chart
        gauge = create_gauge_chart(win_prob, batting_team)
        st.plotly_chart(gauge, use_container_width=True)
        
        # Both teams probabilities
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; 
                    background:#1e1e1e; padding:15px; border-radius:10px;
                    border:1px solid #333; margin-top:-10px'>
            <div style='text-align:center'>
                <div style='color:#aaa; font-size:0.9em'>🏏 {batting_team}</div>
                <div style='color:#4CAF50; font-size:1.8em; font-weight:bold'>
                    {win_prob*100:.1f}%
                </div>
            </div>
            <div style='text-align:center; color:#555; font-size:2em'>vs</div>
            <div style='text-align:center'>
                <div style='color:#aaa; font-size:0.9em'>🎳 {bowling_team}</div>
                <div style='color:#F44336; font-size:1.8em; font-weight:bold'>
                    {loss_prob*100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col2:
        # Match situation summary
        st.markdown("### 📋 Match Situation")
        
        balls_rem = features['balls_remaining']
        overs_rem = balls_rem // 6
        balls_extra = balls_rem % 6
        
        metrics = [
            ("🎯 Runs Needed", f"{features['runs_remaining']}"),
            ("⏱️ Overs Left", f"{overs_rem}.{balls_extra}"),
            ("🚨 Wickets Left", f"{10 - wickets}"),
            ("📈 Current RR", f"{features['current_run_rate']:.2f}"),
            ("⚡ Required RR", f"{features['required_run_rate']:.2f}"),
            ("📊 RR Difference", f"{features['current_run_rate']-features['required_run_rate']:+.2f}")
        ]
        
        for label, value in metrics:
            col_a, col_b = st.columns([1.5, 1])
            with col_a:
                st.markdown(f"<span style='color:#aaa'>{label}</span>",
                           unsafe_allow_html=True)
            with col_b:
                st.markdown(f"<span style='color:#fff; font-weight:bold'>{value}</span>",
                           unsafe_allow_html=True)
            st.markdown("<hr style='margin:4px 0; border-color:#333'>",
                       unsafe_allow_html=True)
        
        # Match factors
        st.markdown("### 🔍 Key Factors")
        factors = get_match_factors(features, win_prob)
        
        for factor_type, factor_text in factors:
            if factor_type == 'good':
                st.markdown(
                    f'<div class="factor-good">✅ {factor_text}</div>',
                    unsafe_allow_html=True
                )
            elif factor_type == 'warning':
                st.markdown(
                    f'<div class="factor-warning">⚠️ {factor_text}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="factor-bad">❌ {factor_text}</div>',
                    unsafe_allow_html=True
                )


# ── Footer ──
st.markdown("---")
st.markdown("""
---
### 👨‍💻 Developer

**Aditya Kumar Gupta**

### 🏏 Model

- XGBoost Classifier
- Accuracy: 77.95%
- IPL Dataset (2008–2019)
- 5-Fold Group Cross Validation

### ⚙️ Built Using

- Python
- Streamlit
- Plotly
- Pandas
- XGBoost
- Scikit-learn
""")