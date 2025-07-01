import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import google.generativeai as genai
import tiktoken
import os

def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        print(f'Loading {file}')
        loader = TextLoader(file, encoding='utf-8')
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def ask_with_gemini(context_docs, question, gemini_api_key):
    """Handle Gemini API calls directly"""
    try:
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate response
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error with Gemini API: {str(e)}"


def ask_and_get_answer(vector_store, q, k=3, llm_choice='openai', api_key=None):
    # Get relevant documents
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    relevant_docs = retriever.get_relevant_documents(q)
    
    if llm_choice == 'openai':
        # Use LangChain for OpenAI
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7)
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        answer = chain.run(q)
        return answer
        
    elif llm_choice == 'gemini':
        if not api_key:
            raise ValueError('Gemini API key is required for Gemini LLM.')
        # Use direct Gemini API call
        answer = ask_with_gemini(relevant_docs, q, api_key)
        return answer
    else:
        raise ValueError('Invalid LLM choice.')


def calculate_embedding_cost(texts):
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(), override=True)
    st.set_page_config(page_title='Domain Specific AI Search Engine', layout='wide')
    st.subheader('Domain Specific AI Search Engine')

    with st.sidebar:
        # LLM Selection
        llm_choice = st.selectbox('Choose LLM:', ['openai', 'gemini'])
        
        # API Key Input based on selection
        if llm_choice == 'openai':
            api_key = st.text_input('OpenAI API Key:', type='password')
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
        elif llm_choice == 'gemini':
            api_key = st.text_input('Gemini API Key:', type='password')
        
        # File upload and processing options
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk_size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        # File processing
        if uploaded_file and add_data:
            # Check if API key is provided
            if not api_key:
                st.error(f'Please provide {llm_choice.upper()} API Key.')
            else:
                with st.spinner('Reading, Chunking and embedding file ...'):
                    try:
                        bytes_data = uploaded_file.read()
                        file_name = os.path.join('./', uploaded_file.name)
                        with open(file_name, 'wb') as f:
                            f.write(bytes_data)

                        data = load_document(file_name)
                        if data is None:
                            st.error('Failed to load document')
                        else:
                            chunks = chunk_data(data, chunk_size=chunk_size)
                            st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                            tokens, embedding_cost = calculate_embedding_cost(chunks)
                            st.write(f'Embedding cost: ${embedding_cost:.4f}')

                            vector_store = create_embeddings(chunks)

                            st.session_state.vs = vector_store
                            st.session_state.llm_choice = llm_choice
                            st.session_state.api_key = api_key
                            st.success('File uploaded, chunked and embedded successfully')
                            
                        # Clean up temporary file
                        if os.path.exists(file_name):
                            os.remove(file_name)
                            
                    except Exception as e:
                        st.error(f'Error processing file: {str(e)}')

    # Question input and answering
    q = st.text_input('Ask a question about the content of your file:')
    if q:
        if 'vs' in st.session_state:
            if not api_key:
                st.error(f'Please provide {llm_choice.upper()} API Key.')
            else:
                try:
                    vector_store = st.session_state.vs
                    st.write(f'k: {k}')
                    
                    with st.spinner('Generating answer...'):
                        answer = ask_and_get_answer(
                            vector_store, 
                            q, 
                            k, 
                            llm_choice=llm_choice, 
                            api_key=api_key
                        )
                        st.text_area('LLM Answer: ', value=answer)
                        
                except Exception as e:
                    st.error(f'Error generating answer: {str(e)}')
                    answer = ""
        else:
            st.warning('Please upload and process a file first.')
            answer = ""

    # Chat history
    st.divider()
    if 'history' not in st.session_state:
        st.session_state.history = ''
    
    if q and 'answer' in locals():
        value = f'Q: {q}\nA: {answer}'
        st.session_state.history = f'{value}\n{"-" * 100}\n{st.session_state.history}'
    
    h = st.session_state.history
    st.text_area(label='Chat History', value=h, key='history', height=400)