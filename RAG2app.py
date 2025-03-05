import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Load Mistral API Key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "t0wNrTSvDhVXjkljdyjO0i00ckjcGoSY")
client = MistralClient(api_key=MISTRAL_API_KEY)

# List of policy URLs
policy_urls = {
    "Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Transfer Policy": "http://udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
    "Graduate Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
    "Library Space Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy",
    "Academic Integrity Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-integrity-policy",
    "Examination Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy",
    "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
    "Graduation Policy": "http://udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
}


@st.cache_data
def scrape_policy(url):
    """Fetches policy content from the given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [
            p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()
        ]
        return (
            "\n".join(paragraphs)
            if paragraphs
            else "Policy content could not be retrieved."
        )
    except requests.RequestException:
        return "Policy content could not be retrieved."


@st.cache_data
def get_policy_text(policy_name):
    """Gets the full policy text as a string."""
    return scrape_policy(policy_urls[policy_name])


def ask_mistral(query, policy_text):
    """Uses Mistral AI to generate a response based on the policy content."""
    messages = [
        ChatMessage(
            role="system",
            content="You are a helpful AI assistant answering questions based on university policies.",
        ),
        ChatMessage(
            role="user",
            content=f"Here is the policy text:\n{policy_text}\n\nQuestion: {query}\nAnswer in a clear and concise manner.",
        ),
    ]

    response = client.chat(model="mistral-tiny", messages=messages)
    return (
        response.choices[0].message.content
        if response.choices
        else "No relevant information found."
    )


# Streamlit UI
st.title("UDST Policy Chatbot")
st.write("Select a policy and ask questions!")

selected_policy = st.selectbox("Choose a policy:", list(policy_urls.keys()))
user_query = st.text_input("Enter your query:")

if st.button("Ask"):
    if user_query.strip():
        policy_text = get_policy_text(selected_policy)
        answer = ask_mistral(user_query, policy_text)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a query.")
