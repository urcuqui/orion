# from langchain_community.llms import Ollama
# from langchain_community.document_loaders import SeleniumURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
import os
from xml.parsers.expat import model
from prompts import system
from langchain.memory import ConversationBufferMemory
#from ollama import Ollama
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

INDEX_PERSIST_DIRECTORY = "D:\chroma"
RUTA_CACHE = "D:/torch"

def generate_text(prompt, model_name="WhiteRabbitNeo/WhiteRabbitNeo-V3-7B"):
    device = (torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu"))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,             # reduce picos en CPU
        device_map=None,    
        cache_dir=RUTA_CACHE
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=RUTA_CACHE, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token          # define el token
    model.config.pad_token_id = tokenizer.pad_token_id
    messages = [
        {"role": "system", "content": "You are an expert agent for adversarial machine learning. "
        "Your task is to provide insights to the user according to their specific needs and how to address them using MITRE ATLAS."
        "You MUST NOT provide explanations or additional information."},
        {"role": "user", "content": prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,   
    add_generation_prompt=True
    )
    print(model.device)
    model_inputs = tokenizer([inputs], return_tensors="pt", padding=True).to(model.device)

    out = model.generate(
    input_ids=model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,
    max_new_tokens=256
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, out)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def get_model():
    """
    Initializes and returns a language model.

    Returns:
        Ollama: An instance of the Ollama language model with the specified model name.
    """
    llm = Ollama(
            model = "deepseek-r1:7b",   
            system=system.SYSTEM_TOOLS,
            temperature=0.2
        )
    return llm

def get_retriever():
    """
    Initializes and returns a retriever for document search.

    If the vector store exists at the specified directory, it loads and returns it.
    Otherwise, it loads data from specified URLs, processes the data, and creates a new vector store.

    Returns:
        Chroma: An instance of the Chroma vector store.
    """
    if os.path.exists(INDEX_PERSIST_DIRECTORY):        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=INDEX_PERSIST_DIRECTORY,embedding_function=embeddings)
        return vectordb.as_retriever() 
    else:        
        urls = [
            "https://github.com/mitre-atlas/atlas-data/blob/main/data/tactics.yaml",
            "https://github.com/mitre-atlas/atlas-data/blob/main/data/mitigations.yaml",
            "https://github.com/mitre-atlas/atlas-data/blob/main/data/techniques.yaml"
        ]

        loader = SeleniumURLLoader(urls=urls)

        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(data)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=INDEX_PERSIST_DIRECTORY
            )
        vectordb.persist()
        return vectordb.as_retriever()
    
def get_conversational_model():    
    """
    Initializes and returns a conversational model.

    The conversational model is created using the language model and retriever.
    It also initializes an empty chat history.

    Returns:
        ConversationalRetrievalChain: An instance of the ConversationalRetrievalChain with the specified language model and retriever.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_model(),
        retriever=get_retriever(),
        memory=memory
    )    
    return qa_chain