import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle as pk


def get_user_input():
    st.subheader("Enter Player's Information:")
    # Get user input for each variable
    potential = st.slider("Potential", min_value=0, max_value=100, step=1)
    value_eur = st.number_input("Value (in Euros)", min_value=0.0, step=100.0)
    wage_eur = st.number_input("Wage (in Euros)", min_value=0.0, step=100.0)
    age = st.number_input("Age", min_value= 0, step= 1)
    international_reputation = st.slider("International Reputation",min_value = 0, step=1, max_value=5),
    release_clause_eur = st.number_input("Release Clause (in Euros)", min_value=0.0, step=100.0)
    shooting = st.slider("Shooting", min_value=0, step=10, max_value=100),
    passing = st.slider("Passing", min_value=0, max_value=100, step=1)
    dribbling = st.slider("Dribbling", min_value=0, max_value=100, step=1)
    physic = st.slider("Physic", min_value=0, max_value=100, step=1)
    movement_reactions = st.slider("Movement Reactions", min_value=0, max_value=100, step=1)
    mentaility_agression = st.slider("Mentaility Agression", min_value=0, max_value=100, step=1)
    mentality_vision = st.slider("Mentality Vision", min_value=0, max_value=100, step=1)
    mentality_composure = st.slider("Mentality Composure", min_value=0, max_value=100, step=1)
    attacking = st.slider("Attacking", min_value=0, max_value=100, step=1)
    skill = st.slider("Skill", min_value=0, max_value=100, step=1)
    power_average = st.slider("Power Average", min_value=0, max_value=100, step=1)

    # Check if all fields are filled
    if st.button("Enter"):
        if not (potential and value_eur and wage_eur and age and release_clause_eur and international_reputation
                and passing and dribbling and physic and movement_reactions
                and mentaility_agression and mentality_vision and mentality_composure and attacking and skill and power_average):
            st.warning("Please fill in all fields.")
            return None
        else:
            user_inputs = [potential, value_eur, wage_eur, age, international_reputation,release_clause_eur, shooting,
                           passing, dribbling, physic, movement_reactions,
                           mentaility_agression, mentality_vision, mentality_composure, attacking,skill,power_average]
            return user_inputs
    return None

def predict_system(user_input):
    if user_input:
        with open('scaler_model.pkl', 'rb') as scaler_file:
            scaler = pk.load(scaler_file)
        user_inputs = scaler.transform([user_input])
        try:
            with open('best_model.pkl', 'rb') as file:
                loaded_model = pk.load(file)
                return loaded_model.predict(user_inputs)[0]
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            return None
    else:
        return None

st.set_page_config(page_title="Predict a Player's Rating", page_icon="⚽", layout="wide")

st.subheader("Hey Soccer fan, Welcome 😃")
st.title("Predict a Player's rating")
st.write("Here, you can enter some features of a player and we wil predict their ratings!")

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Guidelines")
        st.write("##")
        st.write("""
    Please enter the following features of the player.  
    Apart from Value, Wage and Release clause, all other features are in percentages
  1. Potential: Player's potential skill level or overall capability, often used to project how much a player can improve.

  2. Value in euro: Market value of the player in euros, reflecting their perceived worth in the transfer market.

  3. Wage in euro: Player's weekly or monthly salary, denominated in euros.

  4. Release clause eur: The release clause is the amount of money that another club needs to pay to buy out a player's contract. This is stated in euros.

  5. Passing: Player's ability to make accurate and effective passes.

  6. Dribbling: Indicates a player's skill in maneuvering the ball while running.

  7. Attacking short passing: This is a subset of passing, specifically focusing on short passes used in attacking play.

  8. Movement reactions: Refers to a player's ability to react quickly and move effectively on the field.

  9. Power shot power: Relates to a player's ability to take powerful shots on goal.

 10. Mentality vision: Reflects a player's strategic vision on the field, understanding the game and making intelligent decisions.

 11. Mentality composure: Indicates how composed and focused a player is, especially in high-pressure situations.

 12. Power average: An aggregate measure of a player's physical power across different attributes.
""")

    with right_column:
            user_input = get_user_input
            if st.button("Predict"):
                if user_input is not None:
                    prediction = predict_system(user_input)
                    st.write("Your predicted rating is:", prediction)
