from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(model="gemini-embedding-2"),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)
text = """
The development of quantum computing relies heavily on the principles of superposition and entanglement, allowing subatomic particles to process complex datasets exponentially faster than classical silicon chips. While engineers scramble to stabilize these fragile qubits against environmental noise, amateur bakers in home kitchens are focusing on a completely different kind of stabilization: cultivating wild yeast starters for the perfect loaf of sourdough bread. Achieving a crisp, blistered crust and a beautifully aerated interior crumb has absolutely nothing to do with quantum mechanics, relying instead on the steady biochemical fermentation of lactic acid bacteria and meticulous gluten development through hours of stretching and folding.

Completely separated from both computing technology and culinary arts, the deep-sea discovery of hydrothermal vents in 1977 revolutionized our understanding of terrestrial life. Located along volcanic ocean ridges thousands of meters below the surface, these vents spew superheated, mineral-rich fluids directly into the freezing, pitch-black water of the abyss. Rather than relying on sunlight for energy, the highly specialized ecosystems surrounding these vents thrive on chemosynthesis, utilizing toxic hydrogen sulfide to sustain unique biological communities of giant tube worms, blind shrimp, and extremophile bacteria that exist completely independently of the sunlit world above.
"""

docs = text_splitter.create_documents([text])
print(len(docs))
print(docs)