### 构建知识库分享总结

    使用LLamaIndex 调用本地大模型, 向大模型直接提问, 如果大模型在没接触过的知识领域, 同样也会回复,但回答的结果是一本正紧的胡说八道, 即产生大模型幻觉

    同样的逻辑: 大模型本地私有化部署,同样会产生大模型幻觉, 这个问题的解决方案就是构建知识库(即 构建RAG)

    RAG有一定的执行流程:

        用户向大模型提问, 大模型对用户问题语义进行理解并分析, 然后交给知识库, 知识库会进行向量检索查询, 大模型依据知识库检索查询结果进行匹配、总结,最后输出答复结果

    要实现这整个流程, 就需要有知识库, 即 RAG!!!

#### 1.构建知识库
       要构建知识库, 就需要用到LLamaIndex中的"数据连接器", 本项目是读取本地Markdown文件内容(说明:内容质量混乱), 
       所以就需要使用数据连接器 SimpleDirectoryReader(并行处理数据)

       SimpleDirectoryReader 数据连接器: 解析读取本地文件数据
           
       该数据连接器支持文件类型:  markdown, csv, pdf txt

#### 词嵌入模型Embedding

     Embedding 模型大小和输入数据的维度大小有关
     唯一的作用功能: 将文本转为词向量, 跟微调的效果没有啥关系

     选择Embedding模型主要还得看物料的实际情况而定, 是纯中文, 还是纯英文 或者是中文和英文都存在
         1. 如果物料都是中文, 就找一个支持中文比较好的Embedding模型
         2. 如果物料都是英文, 就找一个支持英文比较好的Embedding模型
         3. 如果物料中中文、英文都有, 就找一个中英文都支持的Embedding模型

     本项目选择 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 词嵌入模型, 这是个中文、英文都支持的嵌入模型

     每个大模型的结构中的第一部分都是Embedding部分, 大模型也可以作为词嵌入模型

#### 纯RAG知识库
        ```
          安装依赖环境:
             python 虚拟环境                         3.10
             llama-index-embeddings-huggingface     0.6.1
             llama-index-llms-huggingface           0.6.1
             llama-index                            0.14.4
             huggingface-hub                        0.36.0
             transformers                           4.56.2
    
             pip install llama-index-llms-huggingface==0.6.1
             pip install "transformers[torch]==4.56.2"
             pip install "huggingface_hub[inference]==0.36.0"
             pip install llama-index==0.14.4
        ```
     
     执行调用过程如下:
     ```
         from llama_index.embeddings.huggingface import HuggingFaceEmbedding
         from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex    # 数据连接器
         from llama_index.llms.huggingface import HuggingFaceLLM
        
        
         # 词嵌入模型
         embed_model = HuggingFaceEmbedding(
           # 指定了一个预训练sentence-transformer模型的路径
           model_name="/root/autodl-tmp/llms/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
         )
        
         Settings.embed_model = embed_model
         # 读取并加载文档
         documents = SimpleDirectoryReader("/root/autodl-tmp/project/data01").load_data()
         # 创建VectorStoreIndex, 并使用之前加载的文档来构建向量索引
         # 此索引将文档转换为向量, 并存储这些向量以便于快速检索
         index = VectorStoreIndex.from_documents(documents)  # 调用Embedding模型Settings.embed_model, 对文本进行转换, 将其转换成向量数据
         # 创建一个查询引擎, 这个引擎可接收以查询并返回相关文档的响应
         query_engine = index.as_query_engine()  
         # 这行代码操作会调用大模型Settings.llm对用户问题进行语义化理解并转换成词向量, 然后根据问题词向量和向量库的数据进行相似度检索查询并得到检索结果, 
         # 最后大模型根据检索结果进行匹配、总结输出回复
         rsp = query_engine.query("xtuner是什么?")  
         print(rsp)
     ```

     同样的, 对于本地私有化的大模型没有接触过的知识领域, 向大模型提问,大模型会一本正经的胡说八道, 即 会产生达模型幻觉 

#### RAG核心流程
     用户提问  --->   问题解析    --->    RAG检索    --->    生成答案     ---->    引用溯源


#### 关于RAG本质认识
     问题: 文档特别大,查出来的回答效果特别差, 怎么办?
           首先要明确以下几点:
               1.文档大小与性能有关, 即索引检索的速度快慢有关, 和检索效果好坏没有关系
                 文档特别大, 性能差, 即检索速度慢
               2.检索查询的效果好坏和以下两点有关
                   a. 与知识库物料有关, 如果物料有错误, 查询出来的结果必然不正确, 必须保证物料库的正确性,这样检索出的结果才能保证其正确性
                   b. 与大模型能力有关, 即模型对用户问题语义的理解程度不足, 会导致检索的结果不正确或者不完整

###### 知识库的本质: 知识库中存储的数据是大模型回答问题的答案的依据, 并不是存储的答案 !!! 
       一个好的知识库, 必然是对数据物料有一定要求的, 即物料数据的质量必须要达标
       那如何才能使数据质量达标呢?
       知识库数据所涵盖的满足业务相关需求所涉及的整个场景, 这样的数据知识库数据才算是数据质量达标(即数据全集), 否则就不达标

       构建一个好的是RAG系统, 要根据当前业务需求, 以及一定的规则、标准、边界, 对数据集进行结构化处理, 这样才能提供一个高效的、可靠的、准确、高精度的知识库数据
     
#### 微调的功能作用
     通过微调, 提升大模型领域理解能力!!!
        
     1.微调可以改变大模型的回答方式(包含回答时的规则、格式)、对话风格
     2.微调可以改善大模型对问题语义的理解程度, 使其能更好的理解问题的语义, 避免大模型对问题理解的不彻底或理解错误, 而导致检索出不正确的结果!!!
     3.微调不能改变大模型自身的智能程度, 想通过微调改善大模型回答效果,其效果是有限的,不明显的
     4.但若 1.知识库的数据质量达标,
           2.微调了大模型(提供了需求所涉及的大多数主要场景的核心问题), 改善了一定程度的因理解问题不彻底导致检索效果不佳的问题, 
           3.但有些问题大模型还是理解不到位, 这时候就要考虑大模型的能力了(换能力更强的大模型)
           
