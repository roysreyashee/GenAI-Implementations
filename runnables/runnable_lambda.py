from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
load_dotenv()

def word_count(text):
    return len(text.split())

prompt1 = ChatPromptTemplate.from_template(
        'Write a joke about {topic}'
)

model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser = StrOutputParser()

joke_gen_chain= RunnableSequence(prompt1, model, parser)

parallel_chain= RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain= RunnableSequence( joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'AI'})

# print(result)

final_result = """{} \n word count- {}""".format(result['joke'], result['word_count'])

print(final_result)

final_chain.get_graph().print_ascii()

###Flow###
#  +-------------+                
#               | PromptInput |                
#               +-------------+                
#                       *                      
#                       *                      
#                       *                      
#            +--------------------+            
#            | ChatPromptTemplate |            
#            +--------------------+            
#                       *                      
#                       *                      
#                       *                      
#          +------------------------+          
#          | ChatGoogleGenerativeAI |          
#          +------------------------+          
#                       *                      
#                       *                      
#                       *                      
#             +-----------------+              
#             | StrOutputParser |              
#             +-----------------+              
#                       *                      
#                       *                      
#                       *                      
#      +--------------------------------+      
#      | Parallel<joke,word_count>Input |      
#      +--------------------------------+      
#               **            ***              
#             **                 **            
#           **                     **          
# +-------------+              +------------+  
# | Passthrough |              | word_count |  
# +-------------+              +------------+  
#               **            ***              
#                 **        **                 
#                   **    **                   
#     +---------------------------------+      
#     | Parallel<joke,word_count>Output |      
#     +---------------------------------+
