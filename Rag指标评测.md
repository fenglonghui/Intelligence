### Ragas评估
    
     Ragas (Retrieval-Augmented Generation Assessment) 它是一个框架，它可以帮助我们来快速评估RAG系统的性能，为了评估RAG系统，Ragas需要以下信息:
    
       - question：用户输入的问题。
    
       - answer：从 RAG 系统生成的答案(由LLM给出)。
    
       - contexts：根据用户的问题从外部知识源检索的上下文即与问题相关的文档。
    
       - ground_truths： 人类提供的基于问题的真实(正确)答案。 这是唯一的需要人类提供的信息。 

     官网地址：https://www.ragas.io/

    
#### Ragas 评估指标
     
     Ragas提供了五种评估指标包括：
     
        - 忠实度(faithfulness)
          高忠实度 答案正确, 低忠实度 答案中有错误
          
        - 答案相关性(Answer relevancy)
          答案完整,不缺失, 没有冗余信息、噪声信息
          
        - 上下文精度(Context precision)
          
        - 上下文召回率(Context recall)
          上下文召回率(Context recall)衡量检索到的上下文(Context)与人类提供的真实答案(ground truth)的一致程度。
          
        - 上下文相关性(Context relevancy)

#### LangChain实现的RAG评测

     1.安装依赖
        pip install pypdf
        pip install ragas
        pip install Pillow
        pip install dashscope
        pip install chromadb
        pip install langchain
        pip install langchain-chroma

     2.加载文档
        from langchain_community.document_loaders import PyPDFLoader

        docs = PyPDFLoader("./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf").load()
        docs

     3.检索策略
     
        3.1 检索完整文档
        3.2 检索较大的文档块


    ```
            import os
            from langchain_community.document_loaders.pdf import PyPDFLoader
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_community.llms import Tongyi
            from langchain_chroma import Chroma
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.retrievers import ParentDocumentRetriever
            from langchain.storage import InMemoryStore

            from langchain.prompts import ChatPromptTemplate
            from langchain.schema.runnable import RunnableMap
            from langchain.schema.output_parser import StrOutputParser


            # 初始化大语言模型
            DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
            llm = Tongyi(
                model_name="qwen-max",
                dashscope_api_key=DASHSCOPE_API_KEY
            )
            
            # 创建嵌入模型
            embeddings = DashScopeEmbeddings(
                model="text-embedding-v1",
                dashscope_api_key=DASHSCOPE_API_KEY
            )
             
            # 创建主文档分割器
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512)
             
            # 创建子文档分割器
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)
             
            # 创建向量数据库对象
            vectorstore = Chroma(
                collection_name="split_parents", embedding_function = embeddings
            )
            # 创建内存存储对象
            store = InMemoryStore()
            # 创建父文档检索器
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={"k": 2}
            )
             
            # 添加文档集
            retriever.add_documents(docs)

            # 切割出来主文档的数量
            print(len(list(store.yield_keys())))


            # 创建prompt模板（RAG Prompt）
            template = """You are an assistant for question-answering tasks. 
                  Use the following pieces of retrieved context to answer the question. 
                  If you don't know the answer, just say that you don't know. 
                  Use two sentences maximum and keep the answer concise.
                  Question: {question}
                  Context: {context}
                  Answer:
            """
             
            # 由模板生成prompt
            prompt = ChatPromptTemplate.from_template(template)
             
            # 创建 chain（LCEL langchain 表达式语言）
            chain = RunnableMap({
                "context": lambda x: retriever.get_relevant_documents(x["question"]),
                "question": lambda x: x["question"]
            }) | prompt | llm | StrOutputParser()


            query = "客户经理被投诉了，投诉一次扣多少分？"
            response = chain.invoke({"question": query})
            print(response)

    ```

####  准备评估的QA数据集
       
        from datasets import Dataset

        # 保证问题需要多样性，场景化覆盖
        questions = [
            "客户经理被投诉了，投诉一次扣多少分？",
            "客户经理每年评聘申报时间是怎样的？",
            "客户经理在工作中有不廉洁自律情况的，发现一次扣多少分？",
            "客户经理不服从支行工作安排，每次扣多少分？",
            "客户经理需要什么学历和工作经验才能入职？",
            "个金客户经理职位设置有哪些？"
        ]
        
        ground_truths = [
            "每投诉一次扣2分",
            "每年一月份为客户经理评聘的申报时间",
            "在工作中有不廉洁自律情况的每发现一次扣50分",
            "不服从支行工作安排，每次扣2分",
            "须具备大专以上学历，至少二年以上银行工作经验",
            "个金客户经理职位设置为：客户经理助理、客户经理、高级客户经理、资深客户经理"
        ]
        
        answers = []
        contexts = []
        
        # Inference
        for query in questions:
            answers.append(chain.invoke({"question": query}))
            contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])
         
        # To dict
        data = {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": ground_truths
        }
         
        # Convert dict to dataset
        dataset = Dataset.from_dict(data)
        dataset


    数据结构:
        Dataset({
            features: ['user_input', 'response', 'retrieved_contexts', 'reference'],
            num_rows: 6
        })

#### 评测结果
     from ragas import evaluate
     from ragas.metrics import (
         faithfulness,
         answer_relevancy,
         context_recall,
         context_precision,
     )
     
     result = evaluate(
        dataset = dataset, 
        metrics=[
            context_precision, # 上下文精度
            context_recall, # 上下文召回率
            faithfulness, # 忠实度
            answer_relevancy, # 答案相关性
        ],
        embeddings=embeddings
     )
 
     df = result.to_pandas()
     print(df)
