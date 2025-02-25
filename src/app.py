import streamlit as st
from shopping_assistant import ShoppingAssistant
import sys
import os

# Add 'src' to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from shopping_assistant import ShoppingAssistant  # Now it will work!


# Initialize assistant
assistant = ShoppingAssistant()

st.title("🛍️ AI Shopping Assistant")
# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask me about beauty products!")

if st.button("Send") and user_input:
    response = assistant.process_query([user_input])  # Process query and return response
    st.session_state.chat_history.append(("You", user_input))

    if response:
        st.session_state.chat_history.append(("Assistant", response))
    else:
        st.session_state.chat_history.append(("Assistant", "❌ No response generated."))


# Display chat history
for sender, msg in st.session_state.chat_history:
    st.write(f"**{sender}:** {msg}")
