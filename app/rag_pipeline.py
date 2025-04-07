from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Model variables
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "tiiuae/falcon-7b-instruct"

class RAGPipeline:
    def __init__(self, data_dir='data/'):
        self.load_documents(data_dir)
        self.create_embeddings()
        self.setup_llm()
        self.create_chain()

    def load_documents(self, data_dir):
        loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.text_chunks = splitter.split_documents(documents)

    def create_embeddings(self):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store = FAISS.from_documents(self.text_chunks, embeddings)

    def setup_llm(self):
        hf_pipeline = pipeline("text-generation", model=GENERATION_MODEL_NAME, max_new_tokens=256)
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

    def create_chain(self):
        prompt_template = PromptTemplate(
            template="""Use the following context to answer the user's question.
If you don't know the answer, say you don't know. Don't make anything up.

Context: {context}
Question: {question}

Helpful Answer:""",
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

    def query(self, question: str):
        return self.qa_chain({"query": question})