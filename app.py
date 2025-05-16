# app.py

import streamlit as st
import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
if not load_dotenv():
    logging.warning("No .env file found or error loading it")

# Constants
MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

def ingest_pdfs_from_folder(folder_path):
    """Carrega e processa todos os PDFs da pasta."""
    if not os.path.exists(folder_path):
        logging.error(f"Pasta não encontrada: {folder_path}")
        return []

    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                loader = UnstructuredPDFLoader(file_path=file_path)
                docs = loader.load()
                all_docs.extend(docs)
                logging.info(f"{filename} carregado com sucesso.")
            except Exception as e:
                logging.warning(f"Erro ao processar {filename}: {str(e)}")

    if not all_docs:
        logging.error("Nenhum documento foi carregado da pasta.")
    return all_docs

def split_documents(documents):
    """Split documents into smaller chunks."""
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Documentos divididos em {len(chunks)} chunks.")
    return chunks

@st.cache_resource
def load_vector_db():
    """Load or create the vector database with error handling."""
    try:

        embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

        if os.path.exists(PERSIST_DIRECTORY):
            try:
                vector_db = Chroma(
                    embedding_function=embedding,
                    collection_name=VECTOR_STORE_NAME,
                    persist_directory=PERSIST_DIRECTORY,
                )
                # Verify the collection is not empty
                if vector_db._collection.count() == 0:
                    raise ValueError("Banco vetorial existe mas está vazio.")
                logging.info("Banco vetorial existente carregado.")
                return vector_db
            except Exception as e:
                logging.error(f"Erro ao carregar banco vetorial existente: {str(e)}")
                st.warning("Problema ao carregar banco vetorial existente. Recriando...")
                # Fall through to recreation

        # Create new vector store
        data = ingest_pdfs_from_folder("data")
        if not data:
            return None

        chunks = split_documents(data)
        if not chunks:
            st.error("Nenhum conteúdo válido encontrado no PDF.")
            return None

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Novo banco vetorial criado e persistido.")
        return vector_db

    except Exception as e:
        logging.error(f"Erro crítico ao lidar com banco vetorial: {str(e)}")
        st.error(f"Erro crítico: {str(e)}")
        return None

def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
        Você é João Fazendeiro, um assistente inteligente e amigável que ajuda pessoas
        a entender informações. Sua tarefa é gerar cinco versões diferentes
        da pergunta do usuário, usando linguagem clara e simples para que qualquer pessoa -
        independente de educação ou background - possa entender. Estas versões alternativas
        devem refletir diferentes formas de fazer a mesma pergunta, para melhorar as chances
        de recuperar a informação correta de um banco vetorial, mesmo se o usuário não usar
        termos técnicos. Forneça as cinco perguntas alternativas, cada uma em uma nova linha.
        Pergunta original: {question}""",
    )

    try:
        retriever = vector_db.as_retriever(search_kwargs={"k": 10}) 
        logging.info("Retriever criado com sucesso.")
        return retriever
    except Exception as e:
        logging.error(f"Erro ao criar retriever: {str(e)}")
        st.error("Erro ao configurar o sistema de busca.")
        return None

def create_chain(retriever, llm):
    """Create the RAG chain."""
    if not retriever or not llm:
        return None

    template = """Responda a pergunta baseado APENAS no seguinte contexto:
{context}

Pergunta: {question}

Se você não souber a resposta, diga apenas "Não encontrei informações sobre isso no material disponível."
"""

    try:
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logging.info("Cadeia RAG criada com sucesso.")
        return chain
    except Exception as e:
        logging.error(f"Erro ao criar cadeia RAG: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="João Fazendeiro - Assistente PRODEPE",
        page_icon="🌾",
        layout="centered"
    )

    # CSS para estilo do chat
    st.markdown("""
    <style>
        .stChatMessage {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            max-width: 80%;
        }
        [data-testid="stChatMessage-user"] {
            background-color: #f0f2f6;
            margin-left: auto;
            border-bottom-right-radius: 0;
        }
        [data-testid="stChatMessage-assistant"] {
            background-color: #e1f5fe;
            margin-right: auto;
            border-bottom-left-radius: 0;
        }
        [data-testid="stHorizontalBlock"] {
            gap: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌾 João Fazendeiro")
    st.caption("Assistente virtual do PRODEPE - SEFAZ/PE")

    # Inicializa estado da sessão
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0

    # Container para mostrar o histórico de mensagens (acima do input)
    chat_container = st.container()
    with chat_container:
        for exchange in st.session_state.conversation:
            with st.chat_message(exchange["role"]):
                st.write(exchange["content"])

    # Campo de entrada com chave dinâmica (ajuda na limpeza após envio)
    input_key = f"user_input_{st.session_state.input_counter}"
    with st.form(key="question_form", clear_on_submit=False):
        user_input = st.text_input(
            "Digite sua pergunta sobre o PRODEPE:",
            key=input_key,
            placeholder="Ex: Como posso me inscrever no PRODEPE?",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Enviar")

    # Processa a pergunta
    if (submitted or user_input) and user_input.strip():
        st.session_state.conversation.append({"role": "user", "content": user_input})

        with st.spinner("Pensando..."):
            try:
                # 1. Inicializa o modelo
                llm = ChatGoogleGenerativeAI(
                    model=MODEL_NAME,
                    temperature=0.3,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )

                # 2. Carrega o banco vetorial
                vector_db = load_vector_db()
                if not vector_db:
                    raise ValueError("Banco vetorial não disponível.")

                # 3. Cria retriever
                retriever = create_retriever(vector_db, llm)
                if not retriever:
                    raise ValueError("Retriever não configurado.")

                # 4. Cria cadeia RAG
                chain = create_chain(retriever, llm)
                if not chain:
                    raise ValueError("Cadeia RAG não configurada.")

                # 5. Responde
                response = chain.invoke(user_input)

                st.session_state.conversation.append({"role": "assistant", "content": response})

            except Exception as e:
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": f"Desculpe, ocorreu um erro: {str(e)}"
                })

        # Limpa o input e força atualização
        st.session_state.pop(input_key, None)
        st.session_state.input_counter += 1
        st.rerun()


if __name__ == "__main__":
    main()