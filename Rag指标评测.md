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

#### RAG评测实现

     *************
