import streamlit as st
import sys
import os

# Add 'src' to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from shopping_assistant import ShoppingAssistant

# Initialize assistant
assistant = ShoppingAssistant()

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None  # Tracks pending clarifications
if "preferences" not in st.session_state:
    st.session_state.preferences = {"brand": None, "category": None, "color": None, "exclusions": []}

# Chat Interface
st.title("🛍️ AI Shopping Assistant")
user_input = st.text_input("Ask me about beauty products!")

if st.button("Send") and user_input:
    st.session_state.chat_history.append(("You", user_input))  # Store user input

    response = None  # Ensure response is always initialized

    # Handle pending clarification question
    if st.session_state.pending_question == "category":
        st.session_state.preferences["category"] = user_input.lower()
        st.session_state.pending_question = None
        response = "✨ Do you have a preferred brand, or should I show all options?"
        st.session_state.pending_question = "brand"

    elif st.session_state.pending_question == "brand":
        st.session_state.preferences["brand"] = user_input.lower() if user_input.lower() != "all options" else None
        st.session_state.pending_question = None
        response = assistant.process_query([st.session_state.preferences])

    else:
        # No pending question, process normally
        parsed_query = assistant.parse_user_query_with_llm(user_input)

        print(f"🔍 Debug - User Query Parsed: {parsed_query}")  # <--- Debugging Here
        print(f"🔍 Debug - Stored Preferences Before Updating: {st.session_state.preferences}")  # <--- Debugging Here

        # **Reset exclusions if user changes product category**
        if parsed_query["category"] and parsed_query["category"] != st.session_state.preferences["category"]:
            st.session_state.preferences["exclusions"] = []  # Reset exclusions when switching categories

        # Update stored preferences
        # Ensure category is updated before checking for missing attributes
        if parsed_query["category"]:
            st.session_state.preferences["category"] = parsed_query["category"]
        elif not parsed_query["category"] and st.session_state.preferences["category"]:
            print("✅ Debug - Keeping previously stored category:",
                  st.session_state.preferences["category"])  # Debugging
        else:
            print("⚠️ Debug - No category detected and none stored!")  # Debugging

        if parsed_query["brand"]:
            st.session_state.preferences["brand"] = parsed_query["brand"]
        if parsed_query["color"]:
            st.session_state.preferences["color"] = parsed_query["color"]
        if parsed_query["exclusions"]:
            st.session_state.preferences["exclusions"].extend(parsed_query["exclusions"])

        # Debugging after updating preferences
        print(f"🔍 Debug - Stored Preferences After Updating: {st.session_state.preferences}")

        # Identify missing attributes **AFTER** preferences are updated
        if not st.session_state.preferences["category"]:
            print("⚠️ Debug - Category is still missing after update!")  # Debugging
            response = "🤔 What type of product are you looking for? (e.g., lipstick, foundation)?"
            st.session_state.pending_question = "category"

        elif not st.session_state.preferences["brand"]:
            response = "✨ Do you have a preferred brand, or should I show all options?"
            st.session_state.pending_question = "brand"

        else:
            response = assistant.process_query([st.session_state.preferences])  # Get recommendations

        # Identify missing attributes
        if not st.session_state.preferences["category"]:
            response = "🤔 What type of product are you looking for? (e.g., lipstick, foundation)?"
            st.session_state.pending_question = "category"

        elif not st.session_state.preferences["brand"]:
            response = "✨ Do you have a preferred brand, or should I show all options?"
            st.session_state.pending_question = "brand"

        else:
            response = assistant.process_query([st.session_state.preferences])  # Get recommendations

    # Store assistant's response
    if response:
        st.session_state.chat_history.append(("Assistant", response))
        st.write(f"**Assistant:** {response}")

# Display full chat history
for sender, msg in st.session_state.chat_history:
    st.write(f"**{sender}:** {msg}")
