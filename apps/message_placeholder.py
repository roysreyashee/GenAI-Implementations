from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#chat template
chatTemplate = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])
#load chathistory
chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)
#create prompt

prompt = chatTemplate.invoke({'chat_history': chat_history, 'query': "Where is my refund?"})
print(prompt)