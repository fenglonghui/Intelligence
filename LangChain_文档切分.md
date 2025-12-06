### LangChain 文档切分

  1.文档处理
       TextSpliter
       pip install --upgrade langchain-text-splitters
        
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=512,
           chunk_overlap=200, 
           length_function=len,
           add_start_index=True,
        )
      
      paragraphs = text_splitter.create_documents([pages[0].page_content])
      for para in paragraphs:
          print(para.page_content)
          print('-------')


  2.向量数据库与向量检索

    # !pip install dashscope
    # !pip install faiss-cpu

    import os
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyMuPDFLoader
    
    # 加载文档
    loader = PyMuPDFLoader("./data/deepseek-v3-1-4.pdf")
    pages = loader.load_and_split()
    
    # 文档切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    
    texts = text_splitter.create_documents(
        [page.page_content for page in pages[:1]]
    )
    
    # 灌库
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    index = FAISS.from_documents(texts, embeddings)
    
    # 检索 top-5 结果
    retriever = index.as_retriever(search_kwargs={"k": 5})
    
    docs = retriever.invoke("deepseek v3有多少参数")
    
    for doc in docs:
        print(doc.page_content)
        print("----")


  参考资料: /Users/fenglonghui/deep_learnin_models/人工智能基础/06_LangChain多任务应用开发/index.ipynb
           https://www.alang.ai/langchain/101/
