### 情绪对话模型-项目实现过程

#### 技术选型对比总结, RAG和大模型微调的优缺点:
     1.为什么不直接使用大模型微调?
        1.1 联网的话, 其本质是RAG, 网络内容就是知识库, 无法保证内容真实可靠
        1.2 不联网, 通过询问大模型, 可能会出现大模型幻觉 (就是一本正经的胡说八道)
        1.3 大模型是通过一些固定的数据训练出来的, 但问题是, 这些数据会过时

     2.纯使用RAG查询
       对于用户提出的专业问题, RAG对用户问题语义理解不到位, 这时候就需要借助微调大模型来实现

     微调大模型使用场景:
        1.涉及大模型自身变化, 模型名称,功能介绍说明  (比较固定)
        2.对话风格 (如: 带有明显的情绪色彩对话)     (比较固定)
        3.在使用RAG解决不了用户问题时, 这时候就需要的借助微调大模型的能力来辅助解决


#### 实现步骤: 
    1.准备数据
        自己收集数据, 本项目是通过大模型生成微调训练数据集
        具体步骤如下:
            1.下载开源数据集 LCCC-base_train (data/raw/train/LCCC-base_train.json)
            2.从开源数据集中抽取部分数据
                从LCCC-base_train训练数据集中, 随机抽取1000条不重复的数据存储在本地文件中
                使用程序: utils/extral_data.py 采集1000条数据
            3.针对者1000条数据, 使用智谱大模型和词嵌入模型生成与提示词中聊天风格相似的带有明显感情色彩的结果数据集并存在本地文件中
                大模型: ZhipuAI
                词嵌入模型: sungw111/text2vec-base-chinese-sentence 没有归一化层
                改造词嵌入模型(加入归一化层): embedding_models/zy/text2vec-base-chinese-sentence-normalize 
                模型改造程序: utils/model_convert.py
                生成数据集程序: utils/get_data.py
            4.将生成的结果数据集转化为符合Xtuner格式的数据集, 并保存在本地文件中
                数据格式转化: {"user": "******?", "assistant": "*****.", "style": "温柔"}  => {"conversation": [{"input": "*****？", "output": "*****."}]}
                使用程序:  utils/data_convert.py
            
            得到最终的数据集: data/dataset.json, 在Xtuner工具上进行微调训练
    2.模型选择
        2.1 如何进行模型选型?
            根据当前的任务特点,选择合适的评测数据以及预期的候选模型
            当前项目考虑 模型的中文理解能力 CLU以及在开源数据集上的评测分,根据opencompass的评测分
            原则按主流模型选即可, 如 qwen系列、deepseek系列、智谱系列, 一般选最新发布的
            
        2.2 模型大小如何选?
            根据项目服务器的配置大小选
            任务的复杂度

        本项目选择的大模型是: Qwen/qwen1_5_1_8b_chat
    3.训练、评测
        模型训练框架有 LLamaFactory、Xtuner, 
        因为当前项目的任务结果更偏向于主观评测, Xtuner就提供了在训练过程中的主观评测, 因此选 Xtuner 训练框架
        所以接下来就把原始数据转化为Xtuner格式数据集data/style_data/dataset.json, 然后进行训练
    
        命令最好放置在 xtuner 根目录下执行
            即 cd xtuner
            
####        微调训练参数配置
               基座模型: LLM = /root/autodl-tmp/llm/Qwen/Qwen1___5-1___8B-Chat
               配置文件: config_model = /root/autodl-tmp/llm/Qwen/qwen1_5_1_8b_chat_lora_alpaca_e3.py
               GPU数量: gpu_num = 3
               加速可选项及其参数 options: --deepspeed deepspeed_zero2
               指定GPU卡的ID: gpu_ids = 0,2    指定使用哪几张GPU卡
               Xtuner训练的模型: PTH_FILE = /root/autodl-tmp/xtuner/work_dirs/qwen1_5_1_8b_chat_lora_alpaca_e3/iter_18000.pth

            1.单卡训练
                命令形式: xtuner train ${config_model}
                训练命令: xtuner train /root/autodl-tmp/llm/Qwen/qwen1_5_1_8b_chat_lora_alpaca_e3.py
            
                注意: 训练过程中, 会在xtuner 根目录下生成work_dirs目录, 将生成的模型文件存放在该目录中

            2.多卡微调
                命令形式: NPROC_PER_NODE=${gpu_num} xtuner train ${config_model} Options
                训练命令: NPROC_PER_NODE=3 xtuner train ${config_model} ${Options}
            
            3.多卡指定显卡微调
                命令形式: CUDA_VISIBLE_DEVICES=${gpu_ids} NPROC_PER_NODE=${gpu_num} xtuner train ${config_model} ${options}
                训练命令: CUDA_VISIBLE_DEVICES=0,2 NPROC_PER_NODE=2 xtuner train /root/autodl-tmp/llm/Qwen/qwen1_5_1_8b_chat_lora_alpaca_e3.py --deepspeed deepspeed_zero2
                
            注意: 训练完成后, 会生成Pytorch格式的模型文件, 需要转换成 HuggingFace格式的模型文件 (基座模型是HuggingFace格式的, 所以xtuner训练好的模型需要和基座模型合并, 下一步需要模型转换)          
    
        模型转换:
           命令形式: xtuner convert pth_to_hf ${config_model} ${PTH_FILE} ${SAVE_DIR}
           转换命令: xtuner convert pth_to_hf /root/autodl-tmp/llm/Qwen/qwen1_5_1_8b_chat_lora_alpaca_e3.py /root/autodl-tmp/xtuner/work_dirs/qwen1_5_1_8b_chat_lora_alpaca_e3/iter_18000.pth /root/autodl-tmp/xtuner/work_dirs/qwen1_5_1_8b_chat_lora_alpaca_e3/iter_18000_hf
        
        模型合并:
           如果使用了LoRA / QLoRA 微调，则模型转换后将得到 adapter 参数，而并不包含原 LLM 参数。如果您
           期望获得合并后的模型权重（例如用于后续评测），那么可以利用 xtuner convert merge ：

           命令形式: xtuner convert merge ${LLM} ${CONV_LLM_HF} ${SAVE_DIR}
           合并命令: xtuner convert merge /root/autodl-tmp/llm/Qwen/Qwen1___5-1___8B-Chat /root/autodl-tmp/xtuner/work_dirs/qwen1_5_1_8b_chat_lora_alpaca_e3/iter_18000_hf /root/autodl-tmp/llm/Qwen/Qwen1___5-1___8B-Chat_merged

    4.模型推理部署
        使用Lmdeploy 或者 vllm 进行部署, 当前项目选择Lmdeploy, 它效率快
        LMDeploy 支持两种添加对话模版的形式:
            1.自己写一个json文件
            2.利用xtuner/utils/templates.py文件中找到qwen对话模版字符串,写一个脚本并将qwen对话模版字符串转换成json文件形式的模版

        部署的时候, 要进行对话模版的对齐操作:
           在xtuner 根目录下 -> xtuner/utils/templates.py文件中找到qwen对话模版字符串, 将其转化位json格式文件(chat_template.json)
           转换程序: utils/chat_template_utils.py

           启用服务, 推理模型
            命令形式: lmdeploy serve api_server ${merged_model} --chat-template ${chat_template.json}
            启动服务命令: lmdeploy serve api_server /root/autodl-tmp/llm/Qwen/Qwen1___5-1___8B-Chat_merged --chat-template /root/autodl-tmp/pro/lora/chat_template.json
        
            服务启动: 
            ```
                /root/autodl-tmp/conda/envs/lmdeploy/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
                import pynvml  # type: ignore[import]
                `torch_dtype` is deprecated! Use `dtype` instead!
                [WARNING] gemm_config.in is not found; using default GEMM algo                                   
                HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
                HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
                HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
                INFO:     Started server process [6463]
                INFO:     Waiting for application startup.
                INFO:     Application startup complete.
                INFO:     Uvicorn running on http://0.0.0.0:23333 (Press CTRL+C to quit)
            ```
    5.前端访问
         推荐不使用OpenWebUI(它有自己的对话模版(不能自定义), 会产生测试效果和推理效果差距特别大), 使用 streamlit 即可
        
         streamlit run /root/autodl-tmp/pro/chat_app.py
         ```
            You can now view your Streamlit app in your browser.

            Local URL: http://localhost:8501
            Network URL: http://172.17.0.9:8501
            External URL: http://1.27.207.190:8501
         ```

#### 数据来源 (日常对话)
    a.甲方提供
    b.自己收集 (成本会较高, 难度较大)
      1.指定数据标准 (与甲方沟通)
      2.数据获取方式: 
        2.1 手动采集(难度较大, 时间成本高, 采集难度大)
            手动采集到的数据,需要清洗, 并进行人工处理
        2.2 爬虫 (采集数据有限)
            通过爬虫获取的数据, 需要清洗, 并进行人工处理
        2.3 数据接口 (市场开放了一部分数据, 需要收费)
            数据干净, 可以采用自动化标注
        2.4 AI生成
            数据干净, 可以采用自动化标注
    c.数据清洗、标注
        标注 (1.可以自动化标注; 2.只能人工处理)
    d.指定数据集格式
        依据微调框架(微调工具)指定数据集格式, 如 llamafactory格式数据集, xtuner格式数据集
#### 本项目数据来源
    1.人工指定
    2.基于现有开源数据, 让AI实现情绪数据制作
    
    注意: 如果让AI来帮助处理数据, 尽可能选择效果较好的API接口, 不要使用本地的大模型来处理

    具体数据处理操作:
       1.准备部分现有对话数据
       2.数据审查
         是否为空? 
         是否符合长度要求?
         是否符合情绪核心词标注?
       3.去重
         去重核心: 相似度比较
         文本去重实现流程:
            3.1 先对文本进行编码(通过embedding模型实现: 将文本转为词向量)
            3.2 使用数学算法比较向量相似度(余弦相似度或者欧氏距离)
            3.3 设定阈值 (例如相似度高于0.7就排除)

#### 安装依赖
     pip install zhipuai
     pip install streamlit      python web前端
    
     streamlit 详细介绍: https://zhuanlan.zhihu.com/p/448853407
                
        
                
