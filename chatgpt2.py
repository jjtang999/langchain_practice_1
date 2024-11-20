import os
import constants
import sys  

from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

# Get the query from command-line arguments
query = sys.argv[1]

# Initialize an embedding model
embeddings = OpenAIEmbeddings()

# Initialize an LLM (Language Model)
llm = OpenAI(temperature=0)

# Load the text file and create the index
loader = TextLoader('data/data.txt')
index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])

# Query the index using the LLM
print(index.query(query, llm=llm))
