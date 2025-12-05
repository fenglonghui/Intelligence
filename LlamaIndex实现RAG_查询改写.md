#### 基于 LlamaIndex 实现一个功能较完整的 RAG 系统

     功能要求：
      - 加载指定目录的文件
      - 支持 RAG-Fusion
      - 使用 Qdrant 向量数据库，并持久化到本地
      - 支持检索后排序
      - 支持多轮对话


```
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance
    
    EMBEDDING_DIM = 1536
    COLLECTION_NAME = "full_demo"
    PATH = "./qdrant_db"
    
    client = QdrantClient(path=PATH)

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.response_synthesizers import ResponseMode
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core import Settings
    from llama_index.core import StorageContext
    from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.chat_engine import CondenseQuestionChatEngine
    from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
    from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
    
    # 1. 指定全局llm与embedding模型
    Settings.llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX,api_key=os.getenv("DASHSCOPE_API_KEY"))
    Settings.embed_model = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1)
    
    # 2. 指定全局文档处理的 Ingestion Pipeline
    Settings.transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=200)]
    
    # 3. 加载本地文档
    documents = SimpleDirectoryReader("./data").load_data()
    
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
    
    # 4. 创建 collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
    
    # 5. 创建 Vector Store
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    
    # 6. 指定 Vector Store 的 Storage 用于 index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    
    # 7. 定义检索后排序模型
    reranker = LLMRerank(top_n=2)                         # 重排序
    # 最终打分低于0.6的文档被过滤掉
    sp = SimilarityPostprocessor(similarity_cutoff=0.6)          # 检索后处理
    
    # 8. 定义 RAG Fusion 检索器
    fusion_retriever = QueryFusionRetriever(          ## 查询改写技术
        [index.as_retriever()],
        similarity_top_k=5,                           # 检索召回 top k 结果
        num_queries=3,                                # 指定 生成 query 数
        use_async=False,
        # query_gen_prompt="",                      # 可以自定义 query 生成的 prompt 模板
    )
    
    # 9. 构建单轮 query engine
    query_engine = RetrieverQueryEngine.from_args(
        fusion_retriever,
        node_postprocessors=[reranker],
        response_synthesizer=get_response_synthesizer(
            response_mode = ResponseMode.REFINE
        )
    )
    
    # 10. 对话引擎
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine, 
        # condense_question_prompt="" # 可以自定义 chat message prompt 模板
    )


    # 测试多轮对话
    # User: deepseek v3有多少参数
    # User: 每次激活多少
    
    while True:
        question=input("User:")
        if question.strip() == "":
            break
        response = chat_engine.chat(question)
        print(f"AI: {response}")

```

