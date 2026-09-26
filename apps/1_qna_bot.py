from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

st.title("AskBuddy - AI QnA Bot")
st.markdown("My QnA Both with Langchain and Google Gemini!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    # Read the sender's role and message text from the saved chat message.
    role = message["role"]
    content = message['content']
    # Re-render each previous message in the chat interface.
    st.chat_message(role).markdown(content)

query=st.chat_input("Ask Anything")

# if(query):
#     print(query)
# question="Who is the PM of India?"
# result=llm.invoke(question)
# print(result.content)

if query:
    #chats of user getting stored
    st.session_state.messages.append({'role':'user','content':query})
    # 1. Display user message in the UI
    with st.chat_message("user"):
        st.markdown(query)
        
    # 2. Fetch the AI response safely
    with st.spinner("Thinking..."):
        res = llm.invoke(query)
    
    # 3. Display AI response in the UI
    with st.chat_message("assistant"):
        st.markdown(res.text)

##Chats of assistant getting stored 
    st.session_state.messages.append({'role': 'ai', 'content':res.text})
