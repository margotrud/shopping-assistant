import streamlit as st
import sys
import os

# Add 'src' to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from shopping_assistant import ShoppingAssistant  # Now it will work!

# Initialize assistant
assistant = ShoppingAssistant()

st.title("🛍️ AI Shopping Assistant")

# Store chat history & user preferences
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "preferences" not in st.session_state:
    st.session_state.preferences = {"brand": None, "category": None, "color": None, "exclusions": []}


user_input = st.text_input("Ask me about beauty products!")

if st.button("Send") and user_input:
    # Process the query
    parsed_query = assistant.parse_user_query_with_llm(user_input)

    # Update user preferences with new inputs
    if parsed_query["brand"]:
        st.session_state.preferences["brand"] = parsed_query["brand"]
    if parsed_query["category"]:
        st.session_state.preferences["category"] = parsed_query["category"]
    if parsed_query["color"]:
        st.session_state.preferences["color"] = parsed_query["color"]
    if parsed_query["exclusions"]:
        st.session_state.preferences["exclusions"].extend(parsed_query["exclusions"])

        # ✅ Ensure exclusions are always a list
    exclusions_from_session = st.session_state.preferences.get("exclusions", [])
    exclusions_from_session = exclusions_from_session if isinstance(exclusions_from_session, list) else []

    exclusions_from_query = parsed_query.get("exclusions", [])
    exclusions_from_query = exclusions_from_query if isinstance(exclusions_from_query, list) else []

    # Use stored preferences for better recommendations
    refined_query = {
        "brand": st.session_state.preferences["brand"],
        "category": st.session_state.preferences["category"],
        "color": st.session_state.preferences["color"],
        "exclusions": list(set(st.session_state.preferences.get("exclusions", [])))  # Prevent NoneType error
    }

    # Get recommendations using the stored preferences
    response = assistant.process_query([refined_query])  # Send refined query instead

    # Update chat history
    st.session_state.chat_history.append(("You", user_input))

    if response:
        st.session_state.chat_history.append(("Assistant", response))
    else:
        st.session_state.chat_history.append(("Assistant", "❌ No response generated."))

# Display chat history
for sender, msg in st.session_state.chat_history:
    st.write(f"**{sender}:** {msg}")
