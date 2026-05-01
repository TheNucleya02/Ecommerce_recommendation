import streamlit as st

SUGGESTIONS = {
    "A": "alpha",
    "B": "beta",
}

user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)
user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)
user_first_interaction = (
    user_just_asked_initial_question or user_just_clicked_suggestion
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

has_message_history = len(st.session_state.chat_history) > 0

if not user_first_interaction and not has_message_history:
    st.write("NO HISTORY")
    st.chat_input("Ask a question...", key="initial_question")
    st.pills("Examples", options=SUGGESTIONS.keys(), key="selected_suggestion")
    st.stop()

st.write("PAST STOP")
user_message = st.chat_input("Ask follow up")
if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    elif user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

st.write(f"USER MESSAGE: {user_message}")

for m in st.session_state.chat_history:
    st.write(m)

if user_message:
    st.session_state.chat_history.append(user_message)
    st.write(f"Added {user_message}")

if st.button("Clear"):
    st.session_state.chat_history = []
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None
    st.rerun()

