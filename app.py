import streamlit as st
import pickle
import pandas as pd
import base64

# ===========================
# 🔹 Load Models
# ===========================
# First innings
pipe_first = pickle.load(open('first_innings_pipe.pkl', 'rb'))
first_model = pipe_first['model']
batting_encoder = pipe_first['batting_encoder']
bowling_encoder = pipe_first['bowling_encoder']
city_encoder = pipe_first['city_encoder']
feature_cols_first = pickle.load(open('first_innings_features.pkl', 'rb'))

# Second innings
pipe = pickle.load(open('pipe.pkl', 'rb'))  # your existing second innings pipeline

# ===========================
# 🔹 Teams & Cities
# ===========================
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamshala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

team_logos = {
    "Chennai Super Kings": "logos/CSK.png",
    "Mumbai Indians": "logos/MI.png",
    "Royal Challengers Bangalore": "logos/RCB.png",
    "Kolkata Knight Riders": "logos/KKR.png",
    "Delhi Capitals": "logos/DC.png",
    "Rajasthan Royals": "logos/RR.png",
    "Kings XI Punjab": "logos/PK.png",
    "Sunrisers Hyderabad": "logos/SRH.png"
}


# ===========================
# 🔹 Background Image
# ===========================
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


img = get_base64("ipl.jpg")
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
""", unsafe_allow_html=True)

# ===========================
# 🔹 Title
# ===========================
st.title("🏏 IPL Win Predictor")

# ===========================
# 🔹 Team & City Selection
# ===========================
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    batting_team = st.selectbox("Batting Team", sorted(teams))
    st.image(team_logos[batting_team], width=120)
    selected_city = st.selectbox("Match City", sorted(cities))
with col2:
    st.markdown("<div style='text-align:center; font-size:40px; margin-top:40px;'>🆚</div>", unsafe_allow_html=True)
with col3:
    bowling_team = st.selectbox("Bowling Team", sorted(teams))
    st.image(team_logos[bowling_team], width=120)

# ===========================
# 🔹 Match Stage
# ===========================
stage = st.radio("Match Stage", ["First Innings", "Second Innings"])

# ===========================
# 🔹 Inputs
# ===========================
if stage == "First Innings":
    col4, col5, col6 = st.columns(3)
    with col4:
        score = st.number_input("Current Score", min_value=0)
    with col5:
        overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0)
    with col6:
        wickets_out = st.number_input("Wickets Out", min_value=0, max_value=10)

    crr = score / overs if overs > 0 else 0

    # Encode and create input dataframe
    input_df = pd.DataFrame({
        'batting_team_enc': batting_encoder.transform([batting_team]),
        'bowling_team_enc': bowling_encoder.transform([bowling_team]),
        'city_enc': city_encoder.transform([selected_city]),
        'runs': [score],
        'wickets': [wickets_out],
        'overs': [overs],
        'current_run_rate': [crr]
    })

    # Reorder columns
    input_df = input_df[feature_cols_first]
    model = first_model

else:  # Second Innings
    col4, col5, col6 = st.columns(3)
    with col4:
        target = st.number_input("Target Score", min_value=1)
    with col5:
        score = st.number_input("Current Score", min_value=0)
    with col6:
        overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0)
    wickets_out = st.number_input("Wickets Out", min_value=0, max_value=10)

    runs_left = target - score
    balls_left = int((20 - overs) * 6)
    wickets = 10 - wickets_out
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    model = pipe  # existing second innings pipeline

# ===========================
# 🔹 Validate Teams
# ===========================
if batting_team == bowling_team:
    st.error("❌ Teams must be different!")
    st.stop()

# ===========================
# 🔹 Prediction
# ===========================
if st.button("Predict Winning Probability"):
    try:
        result = model.predict_proba(input_df)
        win_prob = int(result[0][1] * 100)
        loss_prob = int(result[0][0] * 100)

        st.subheader("📊 Winning Probability")
        col7, col8 = st.columns(2)
        with col7:
            st.metric(batting_team, f"{win_prob}%")
        with col8:
            st.metric(bowling_team, f"{loss_prob}%")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# ===========================
# 🔹 Match Insights
# ===========================
st.subheader("📌 Match Insights")
if stage == "First Innings":
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        st.metric("Score", score)
    with col10:
        st.metric("Overs", overs)
    with col11:
        st.metric("Wickets Out", wickets_out)
    with col12:
        st.metric("CRR", round(crr, 2))
else:
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        st.metric("Runs Left", runs_left)
    with col10:
        st.metric("Balls Left", balls_left)
    with col11:
        st.metric("CRR", round(crr, 2))
    with col12:
        st.metric("RRR", round(rrr, 2))