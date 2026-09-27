from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

model1= ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

model2= ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

prompt1 = ChatPromptTemplate.from_template(
    'Generate short and simple text from the following \n {text}'
)

prompt2 = ChatPromptTemplate.from_template(
    'Generate 5 short question answers from the following \n {text}'
)

prompt3 = ChatPromptTemplate.from_template(
    'Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}'
)

parser=StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

final_chain = parallel_chain | merge_chain

text= """
    The landscape of modern cinema underwent a massive transformation in 1994, a year widely regarded as one of the most influential in Hollywood history. Audiences were treated to a diverse slate of films that pushed artistic boundaries and shattered box office expectations. Quentin Tarantino’s "Pulp Fiction" revitalized independent filmmaking with its non-linear narrative and gritty, stylized dialogue, ultimately winning the Palme d'Or at Cannes. Simultaneously, Disney’s animated epic "The Lion King" captivated families worldwide, grossing over $766 million during its initial run and setting new benchmarks for musical animation. Meanwhile, Frank Darabont’s "The Shawshank Redemption" initially struggled at the box office but went on to achieve legendary status through home video rentals, solidifying its place at the top of many all-time greatest movies lists. This convergence of high art, mainstream blockbusters, and cult classics proved that cinema could simultaneously satisfy commercial studios and high-minded critics.
"""

result = final_chain.invoke({'text': text})

print(result)

final_chain.get_graph().print_ascii()

######  FLOW ######
#                      +---------------------------+                       
#                      | Parallel<notes,quiz>Input |                       
#                      +---------------------------+                       
#                        ****                  ****                        
#                    ****                          ****                    
#                  **                                  **                  
#   +--------------------+                       +--------------------+    
#   | ChatPromptTemplate |                       | ChatPromptTemplate |    
#   +--------------------+                       +--------------------+    
#              *                                            *              
#              *                                            *              
#              *                                            *              
# +------------------------+                   +------------------------+  
# | ChatGoogleGenerativeAI |                   | ChatGoogleGenerativeAI |  
# +------------------------+                   +------------------------+  
#              *                                            *              
#              *                                            *              
#              *                                            *              
#     +-----------------+                         +-----------------+      
#     | StrOutputParser |                         | StrOutputParser |      
#     +-----------------+                         +-----------------+      
#                        ****                  ****                        
#                            ****          ****                            
#                                **      **                                
#                     +----------------------------+                       
#                     | Parallel<notes,quiz>Output |                       
#                     +----------------------------+                       
#                                     *                                    
#                                     *                                    
#                                     *                                    
#                         +--------------------+                           
#                         | ChatPromptTemplate |                           
#                         +--------------------+                           
#                                     *                                    
#                                     *                                    
#                                     *                                    
#                       +------------------------+                         
#                       | ChatGoogleGenerativeAI |                         
#                       +------------------------+                         
#                                     *                                    
#                                     *                                    
#                                     *                                    
#                           +-----------------+                            
#                           | StrOutputParser |                            
#                           +-----------------+                            
#                                     *                                    
#                                     *                                    
#                                     *                                    
#                        +-----------------------+                         
#                        | StrOutputParserOutput |                         
#                        +-----------------------+                 
###