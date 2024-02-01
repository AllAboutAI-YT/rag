from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os

os.environ["OPENAI_API_KEY"] = "OPENAI API KEY"

# initializing the embeddings
embeddings = OpenAIEmbeddings()

# default model = "gpt-3.5-turbo"
llm = ChatOpenAI()

directory = "YOUR DOCUMENTS PATH"

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)

def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

db = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings
)

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
    similar_docs = db.similarity_search(query, k=2) # get two closest chunks
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer
    
print("Private Q&A chatbot")
prompt = input("Enter your query here: ")

if prompt:
    answer = get_answer(prompt)
    print(f"Answer: {answer}")   
