import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


class RAGSystem:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0,
        )
        self.vectorstore = None
        self.qa_chain = None

    def load_document(self, file_path: str, file_ext: str):
        ext = file_ext.lower()
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext in ("docx", "doc"):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader.load()

    def process_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        self._build_chain()

    def _build_chain(self):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
        )

    def query(self, question: str) -> dict:
        if not self.qa_chain:
            raise ValueError("No documents loaded. Please upload documents first.")
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": result["source_documents"],
        }

    def clear(self):
        self.vectorstore = None
        self.qa_chain = None
