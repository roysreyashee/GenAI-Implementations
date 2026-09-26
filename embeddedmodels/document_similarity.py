from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", output_dimensionality=300)

documents=[
    "Virat Kohli (born 5 November 1988) is an Indian international cricketer and former all-format captain of the Indian national cricket team. He is a right-handed batter and an occasional right-arm medium-pace bowler. Considered one of the greatest batters in cricket, he has been acclaimed for his batting skills and records. Kohli has the most centuries in ODIs and the second-most centuries in international cricket with 85 tons across all formats. He is also the leading run-scorer in the Indian Premier League."
    "Rohit Sharma is an Indian international cricketer and the former multi-format captain of the India national cricket team. Widely regarded as a modern-day great, the explosive right-handed opening batter is fondly known as The Hitman"
    "Jasprit Bumrah is the best Indian bowler"
]
query = 'tell me about Taylor Swift'

doc_embedding = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

print(cosine_similarity([query_embedding], doc_embedding))
