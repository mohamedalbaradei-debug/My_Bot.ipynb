import streamlit as st

st.set_page_config(page_title="My Bot", layout="wide")

st.title("🤖 My Bot")
st.write("Welcome to My Bot Application")

# Add your bot logic here
st.write("This is your Streamlit app. Replace this with your bot code.")

# Example: Simple input/output
user_input = st.text_input("Enter something:")
if user_input:
    st.write(f"You entered: {user_input}")
