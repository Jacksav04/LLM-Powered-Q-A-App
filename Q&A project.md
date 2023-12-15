# Project: Question-Answering on Private Documents


```python
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

os.environ['PINECONE_API_KEY'] = '4db3c405-3489-44cb-b461-40532f772fe9'
os.environ['PINECONE_ENV'] = 'gcp-starter'
os.environ['OPENAI_API_KEY'] = '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'
```


```python
pip install pypdf -q
```


```python
pip install docx2txt -q
```


```python
pip install wikipedia -q
```


```python
pip install --upgrade langchain
```


```python
def load_document(file):
    import os
    name, extension = os.path.splitext(file)
    
    
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtloader
        print(f'loading {file}')
        loader = Docx2Loader(file)
    else:
        print('Document format not supported!')
        return None

    data = loader.load()
    return data

#wikipedia
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data
```


```python
def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks
```

### Calculating Cost


```python
def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0003:.6f}')
    
```

### Embedding and Uploading to a Vector Database


```python
def insert_or_fetch_embeddings(index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    

    embeddings = OpenAIEmbeddings()
    
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    
    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        print(f'Creating index {index_name} and embeddings ... ', end='')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
        
    return vector_store

    
```


```python
def delete_pinecone_index(index_name='all'):
    import pinecone
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    
    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ... ')
        for index in indexes:
            pinecone.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ... ', end='')
        pinecone.delete_index(index_name)
        print('Ok')

```

### Asking and Getting Answers


```python
def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    
    answer = chain.run(q)
    return answer

def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    
    
    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})
    
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    
    result = crc({'question': question, 'chat history': chat_history})
    chat_history.append((queston, result['answer']))
    
    return result, chat_history
```

### Running Code


```python
data = load_document(r'C:\Users\Jacksav\OneDrive\Desktop\Programming_Projects\jupyter\PrivateDocumentQA\WW2.pdf')
print(data[1].page_content)
```


```python
#data = load_from_wikipedia('GPT-4')
#print(data[0].page_content)
```


```python
chunks = chunk_data(data)
print(len(chunks))
print(chunks[10].page_content)
```


```python
print_embedding_cost(chunks)
```


```python
delete_pinecone_index()
```


```python
index_name = 'askadocument'
vector_store = insert_or_fetch_embeddings(index_name)
```


```python
q = 'What is the whole document about?'
answer = ask_and_get_answer(vector_store, q)
print(answer)
```


```python
import time
i = 1
print('Write Quit or Exit to quit.')
while True:
    q = input(f'Question #{i}: ')
    i +=1
    if q.lower() in ['quit', 'exit']:
        print('Quitting ... Bye!')
        time.sleep(2)
        break
        
    answer = ask_and_get_answer(vector_store, q)
    print(f'\nAnswer: {answer}')
    print(f'\n {"-" * 50} \n')
```


```python
delete_pinecone_index()
```


```python
data = load_from_wikipedia('ChatGPT')
chunks = chunk_data(data)
index_name = 'chatgpt'
vector_store = insert_or_fetch_embeddings(index_name)
```


```python
q = 'what is chatgpt?'
answer = ask_and_get_answer(vector_store, q)
print(answer)
```


```python
# asking with memory
chat_history = []
question = 'How many died in pearl harbour?'
result, chat_history = ask_with_memory(vector_store, question, chat_history)
print(result['answer'])
print(chat_history)
```


```python

```
