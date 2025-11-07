## 人工智能大模型微调介绍

#### 安装到指定的位置

      在 AutoDL 上将安装的依赖包和虚拟环境安装在数据盘上，可以按照以下步骤操作:

         1.使用 Conda 环境
            
            a.创建目录：在数据盘上创建用于存放包和虚拟环境的目录。
            
            ```
                  mkdir -p /root/autodl-tmp/conda/pkgs
                  mkdir -p /root/autodl-tmp/conda/envs
            ```
            
            b.配置 Conda：将包缓存和虚拟环境的路径设置到数据盘上
            
            ```
                  conda config --add pkgs_dirs /root/autodl-tmp/conda/pkgs
                  conda config --add envs_dirs /root/autodl-tmp/conda/envs
            ```
            
            c.创建虚拟环境：创建一个新的 Conda 虚拟环境。
            
            ```
                  conda create -n myenv python=3.x
                  conda activate myenv

                  其中 myenv 是你自定义的虚拟环境名称，python=3.x 指定 Python 版本。
            ```
            
            d.使用 conda install 命令安装特定的包


         2.使用 pip 将依赖库安装到数据盘上，可以通过设置 --target 参数来指定安装路径   (这种方式, 弊端: 安装的依赖命令, 不能在命令行是识别执行)

            步骤 1：创建目标目录
                在数据盘上创建一个目录，用于存放安装的依赖库。

                假设数据盘挂载在 /root/autodl-tmp，你可以创建一个名为 site-packages 的目录
         
         ```
                mkdir -p /root/autodl-tmp/site-packages
         ```

            步骤 2：安装依赖库到指定目录
         
         ```
            # 批量安装
            pip install --target=/root/autodl-tmp/site-packages -r requirements.txt

            # 单个安装
            pip install --target=/root/autodl-tmp/site-packages numpy
         ```

            步骤 3：配置 Python 解释器
                为了让 Python 解释器能够找到安装在数据盘上的依赖库，需要将该目录添加到 PATH 环境变量中。你可以在终端中运行以下命令：
         
         ```
                export PATH=/root/autodl-tmp/site-packages:$PATH
         ```

                如果你希望这个设置在每次登录时自动生效，可以将上述命令添加到你的 ~/.bashrc 或 ~/.bash_profile 文件中：
         
         ```
               echo 'export PYTHONPATH=/root/autodl-tmp/site-packages:$PYTHONPATH' >> ~/.bashrc
               source ~/.bashrc
         ```

         3.验证安装
          
           检查安装的包：在终端中运行以下命令，检查是否成功安装了依赖库：
         
         ```
               pip list

               # 或者直接在 Python 解释器中导入并测试：
               import numpy
               print(numpy.__version__)
         ```

         4.检查路径, 确保 Python 解释器能够找到安装在数据盘上的包：
         
         ```
               import sys
               print(sys.path)
         ```

         通过以上步骤，你可以将依赖库安装到数据盘上，并确保 Python 解释器能够正确地找到和使用这些库。


        5.清理Autodl系统盘数据

        一.快速定位大文件

           1.检查关键目录

             du -sh /tmp/ /root/.cache查看临时文件和缓存占用‌

             若/tmp目录爆满（如/tmp/pymp-*临时文件），直接删除：rm -rf /tmp/pymp-*

           2.‌清理pip和HuggingFace缓存

             pip缓存：rm -rf /root/.cache/pip/*‌

             HuggingFace模型缓存：rm -rf ~/.cache/huggingface/hub/*

#### 环境搭建

     1.conda下载及安装
        安装 Miniconda 或 Anaconda
        Miniconda是一个轻量级的 Anaconda 发行版,专为需要灵活管理 Python 环境和包的用户设计。

        下载地址: https://www.anaconda.com/download/success

     2.验证安装
      
      ```
       conda --version
      ```

      环境初始化
      miniconda3/bin/conda init

     3.使用 Conda, 创建虚拟环境
       conda create --name name python=3.8
       # 激活环境
       conda activate name  或者 source activate name

     4.在环境中安装包
       conda install numpy

     4.1 安装插件
       pip instal1 torch==1.12.0 torchvision==0.13.0 numpy==1.21.5 matplotlib==3.5.1 requests==2.25.1 pandas==2.4

     5.退出环境
       conda deactivate
       列出环境列表
       conda env list
       删除环境
       conda remove --name 环境名称 --all

     5.shell 配置：
       source ~/.bashrc  # 对于 bash shell
       source ~/.zshrc   # 对于 zsh shell




#### 大模型的下载

      1.通过Huggingface平台下载, 国外平台
      2.通过ModelScope平台下载, 国内平台

      安装 Hugging Face 库
        pip install transformers
        pip install transformers datasets tokenizers

        1.下载与加载模型到指定文件夹
        
        ```
            from transformers import AutoModel, AutoTokenizer
            # 替换为你选择的模型名称
            model_name = "bert-base-uncased"
            # 指定模型保存路径
            cache_dir = "./my_model_cache"
            # 下载并加载模型和分词器到指定文件夹
            model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        ```

        2.Hugging Face API 使用

          a1.匿名访问 API
          
          ```
              import requests
              API_URL = "https://api-inference.huggingface.co/models/bert-base-chinese"
              # 不使用 Authorization 头以进行匿名访问
              response = requests.post(API_URL, json={"inputs": "你好，Hugging Face!"})
              print(response.json())
          ```

          a2.使用 Inference API
          
          ```
              from transformers import pipeline
              # 替换为你的实际 API Token
              API_TOKEN = "your_api_token_here"
              # 使用 API Token
              generator = pipeline("text-generation", model="gpt2", use_auth_token=API_TOKEN)
              output = generator("The future of AI is", max_length=50)
              print(output)
          ```

          a3.文本生成/分类

          ```
            生成, 在线访问, 使用 Hugging Face 的 Inference API 调用中文文本生成模型：
                import requests

                API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinesecluecorpussmall"
                API_TOKEN = "your_api_token_here" # 替换为你的实际 API Token
                headers = {"Authorization": f"Bearer {API_TOKEN}"}

                # 发送文本生成请求

                response = requests.post(API_URL, headers=headers, json={"inputs": "你好，我是一款
                语言模型，"})
                print(response.json())


            模型下载到本地访问, 你可以将模型下载到本地，然后使用 pipeline 进行文本生成:
                from transformers import pipeline
                # 本地加载中文GPT-2模型
                generator = pipeline("text-generation", model="uer/gpt2-chinesecluecorpussmall", cache_dir="./my_model_cache")
                # 生成文本
                output = generator("你好，我是一款语言模型，", max_length=50, num_return_sequences=1)
                print(output)

            分类,在线访问:
                import requests
                API_URL = "https://api-inference.huggingface.co/models/uer/roberta-basefinetuned-cluener2020-chinese"
                API_TOKEN = "your_api_token_here" # 替换为你的实际 API Token
                headers = {"Authorization": f"Bearer {API_TOKEN}"}
                # 发送文本分类请求
                response = requests.post(API_URL, headers=headers, json={"inputs": "我喜欢用
                Hugging Face的transformers库！"})
                print(response.json())


            分类, 下载到本地访问, 你可以将模型下载到本地，然后使用 pipeline 进行文本分类
                from transformers import pipeline
                # 本地加载中文RoBERTa模型
                classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetunedcluener2020-chinese", cache_dir="./my_model_cache")
                # 进行情感分析
                result = classifier("我喜欢用Hugging Face的transformers库！")
                print(result)

          ```

      ModelScope：一站式中文模型平台
         阿里达摩院开源的模型即服务（MaaS）平台，集成 300+ 中文优化模型，覆盖 NLP/CV/多模态任务。

        核心功能详解:
            1.模型生态
              覆盖 InternVL2-26B（多模态）、Qwen、DeepSeek 等国产 SOTA 模型，支持免费下载与微调。
              提供行业数据集（如阿里电商数据），预训练模型免环境配置在线运行。

              ModelScope 模型下载:
              
              ```
                  from modelscope import snapshot_download
                  model_dir = snapshot_download('Qwen/Qwen3-0.6B', cache_dir="/root/autodl-tmp/llm")

              ```

            2.高效推理 API
              
              ```
                  from modelscope.pipelines import pipeline
                  # 大语言模型调用
                  text_gen = pipeline('text-generation', model='deepseek-ai/DeepSeek-R1')
                  print(text_gen("人工智能的未来趋势"))
              ```

#### datasets 库核心方法

    1.加载数据集, load_dataset 方法加载任何数据集：

      ```
        from datasets import load_dataset
        # 加载GLUE数据集
        dataset = load_dataset("glue", "mrpc")
        print(dataset)

      ```
    2.加载磁盘数据
      
      ```
        from datasets import load_from_disk
        # 从本地磁盘加载数据集
        dataset = load_from_disk("./my_dataset")
        print(dataset)
      ```

#### 本地化部署(大模型)

      Ollama & vLLM & LMDeploy + ModelScope

      实战操作:
      #================================ 调用模型 api接口服务 ================================#

      1.启动 ollama 服务:
        
        ollama serve
        持久化调用: ollama run qwen3:0.6b

      
      2.启动vllm服务:
        
        持久化调用: vllm serve /root/autodl-tmp/llm/Qwen/Qwen3-0.6B --port 8000
        单次服务调用:  python test02.py

      3.启动lmdeploy服务:
        
        持久化调用: lmdeploy serve api_server /root/autodl-tmp/llm/Qwen/Qwen3-0.6B

      #================================ 调用模型 api接口服务 ================================

      # 1.ollama 服务调用
      # client = OpenAI(base_url="http://localhost:11434/v1", api_key="self_key123")
      # chat_completion = client.chat.completions.create(
      #    messages=[{"role": "user", "content": "你好, 请介绍一下你自己."}], model="qwen3:0.6b"
      # )

      # 2.vllm 服务调用
      # client = OpenAI(base_url="http://localhost:8000/v1", api_key="self_key123")      # api key 随便给
      # chat_completion = client.chat.completions.create(
      #    messages=[{"role": "user", "content": "你好, 请介绍一下你自己."}], model="/root/autodl-tmp/llm/Qwen/Qwen3-0.6B"
      # )

      # 3.lmdeploy 服务调用
      client = OpenAI(base_url="http://localhost:23333/v1", api_key="self_key123")      # api key 随便给
      chat_completion = client.chat.completions.create(
         messages=[{"role": "user", "content": "帮我实现一个贪吃蛇小游戏."}], model="/root/autodl-tmp/llm/Qwen/Qwen3-0.6B"
      )

      print(chat_completion.choices[0])



##### 一、Ollama：轻量级本地化部署框架

         注意: 采用 curl -fsSL https://ollama.com/install.sh | sh 这种方式下载太慢, 不建议使用

              Ollama下载安装推荐方式: 在ModelScope上,选择模型 modelscope/ollama-linux 下载
              
         1.下载命令
         
         ```
               pip install modelscope  # 安装 ModelScope
               modelscope download --model=modelscope/ollama-linux --local_dir /root/autodl-tmp/ollama-linux --revision v0.11.5

         ```
              
         2.安装 Ollama
         
         ```
               # 运行ollama安装脚本
               cd ollama-linux
               sudo chmod 777 ./ollama-modelscope-install.sh
               ./ollama-modelscope-install.sh
         ```


      详细部署流程:
         
         1. 安装与环境配置
            采用"Ollama下载安装"推荐方式

         2. 模型加载与交互
         
         ```
            # 下载并运行模型（如 DeepSeek-R1）
            ollama run deepseek-r1:1.5b
            # 命令行对话示例
            >>> "解释量子纠缠现象"
            >>> /bye # 退出会话
         ```
         
         3. API 服务化部署
         
         ```
            # 启动服务（默认端口 11434）
            export OLLAMA_HOST="0.0.0.0:11434" # 开放远程访问
            ollama serve
            # 远程调用示例（JSON 格式）
            curl http://192.168.1.100:11434/api/generate -d '{
            "model": "deepseek-r1",
            "prompt": "写一首关于春天的诗",
            "stream": false
            }'
         ```

##### 二、vLLM：高性能分布式推理框架

          加州伯克利分校研发的推理引擎，通过 PagedAttention 算法优化 KV 缓存，吞吐量较 HuggingFace 提升 24 倍，适合高并发生产环境。

      详细部署流程:
         
         1. 环境依赖安装
         
         ```
            # 创建虚拟环境
              conda create -n vllm python=3.10
              source activate vllm
            
            # 安装 PyTorch 与 vLLM（需 CUDA 12.4）
              pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
            
              pip install vllm==0.8.5 
         ```

         2. 模型加载与离线推理
         
         ```
            from vllm import LLM, SamplingParams
            # 初始化模型（以 DeepSeek-R1-Distill-Qwen-7B 为例）
            llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            trust_remote_code=True, max_model_len=4096)
            # 批量推理
            prompts = ["量子计算的优势是什么？", "如何训练 GPT 模型？"]
            outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95,
            max_tokens=100))
         ```

         3. 启动 OpenAI 兼容 API 服务
         
         ```
            # 单卡启动（DeepSeek-R1-Distill-Qwen-7B）
            vllm serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000
            # 多卡张量并行（DeepSeek-R1-Distill-Qwen-32B，4 卡）
            vllm serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --port 8000 --
            tensor-parallel-size 4
         ```

          # 首先启动vllm服务, 带着微调时的聊天模版 chat_template
          # 4. 在vllm上部署训练好的qwen模型:
          #    vllm serve /root/autodl-tmp/llm/Qwen/Qwen2.5-1.5B-Iora --chat-template /root/autodl-tmp/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/lora/train_2025-10-26-11-33-38/chat_template.jinja
          #
          #    vllm serve /root/autodl-tmp/llm/Qwen/Qwen2.5-1.5B-Iora --chat-template /root/autodl-tmp/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/lora/chat.jinja
          # 在新命令行,执行 python test02.py
          # 开始多轮循环 

         4. 在ollama上训练、打包好模型, 在vllm上部署启动服务(带着ollama框架的chat_template, 为了保持对话模版一致, 否则会出现回答差异性出现):

            vllm serve /root/autodl-tmp/llm/Qwen/Qwen2.5-1.5B-Iora --chat-template /root/autodl-tmp/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/lora/chat.jinja

            vllm serve /root/autodl-tmp/llm/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-lora --chat-template /root/autodl-tmp/llm/deepseek-ai/chat_r1.jinja

         5. 调用vllm上部署的服务, 让Qwen2.5-1.5B-Iora训练模型响应


##### 三、LMDeploy：生产级量化与国产硬件适配

      由 InternLM 团队推出的端到端推理框架，专注模型压缩与异构硬件部署，支持昇腾（Ascend）
      NPU，显存优化达 90%+。

      详细部署流程:
         
         1. 环境配置与安装
         
         ```
            # 创建虚拟环境
              conda create -n lmdeploy python=3.10 -y
              source activate lmdeploy
            # 安装 LMDeploy（x86 环境）
              pip install lmdeploy[all]==0.5.3
            # 昇腾环境需额外安装 DLInfer
              pip install dlinfer-ascend
         ```

         2. 模型量化实战
         
         ```
            # W4A16 量化（以 InternLM2-5-7B 为例）
            lmdeploy lite auto_awq internlm2_5-7b-chat --w-bits 4 --work-dir ./model_4bit
            # 启动量化模型对话
            lmdeploy chat ./model_4bit --model-format awq
         ```

         3. API 服务部署
         
         ```
            # 启动 API 服务（含量化）
            lmdeploy serve api_server ./model_4bit --server-port 23333 --quant-policy 4
            # 客户端调用（Python）
            from openai import OpenAI
            client = OpenAI(base_url="http://localhost:23333/v1", api_key="YOUR_KEY")
            response = client.chat.completions.create(model="default", messages=[{"role":"user", "content":"解释强化学习原理"}])
         ```

#### 大模型微调-LLama Factor 微调 Qwen

      1. LoRA微调的基本原理
      2. LLaMA Factory介绍

      环境配置
      
      ```
         # 创建环境
         conda create -n llama_factory python=3.10
         source activate llama_factory
         # 安装依赖
         git clone https://github.com/hiyouga/LLaMA-Factory.git      # LLaMA-Factory 最新版本: 0.9.4
         cd LLaMA-Factory
         pip install -e .
      ```

      通过Web UI配置

         1. 启动Web服务

         ```
            cd LLaMA-Factory
            llamafactory-cli webui
         ```

         2.合并模型
           
           将基座模型和LoRA训练模型进行合并打包

              基座模型/原模型: Qwen/Qwen2.5-7B-Instruct
              
              LoRA训练模型:   train20151020092215/checkpoint-1000 
           
           在llamafactory webui界面,  
             
             主要操作: 

                模型路径: (原模型绝对路径)

                    /root/autodl-tmp/llm/Qwen/Qwen2.5-7B-Instruct
                          
                检查点路径: (LoRA 训练模型绝对路径)
                
                    /root/autodl-tmp/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/lora/train_2025-10-23-23-09-48/checkpoint-1100


                Export选项: 

                    最大分块大小(GB):  5 默认

                    导出项目: /root/autodl-tmp/llm/Qwen/Qwen2.5-7B-Instruct-qlora4bit

                点击"开始导出"按钮, 导出即可


#### Qwen 打包部署（大模型转换为 GGUF 以及使用 ollama 运行）
      
      # 创建虚拟环境:
        conda create -n llama.cpp python=3.12
        source activate llama.cpp

      1. 将hf模型转换为GGUF
         
         1.1 需要用llama.cpp仓库的convert_hf_to_gguf.py脚本来转换
         
         ```
            git clone https://github.com/ggerganov/llama.cpp.git
            pip install -r /root/autodl-tmp/llama.cpp/requirements.txt
            
         ```

         1.2 执行转换
         
         ```
            # 如果不量化，保留模型的效果
            python /root/autodl-tmp/llama.cpp/convert_hf_to_gguf.py /root/autodl-tmp/llm/Qwen/Qwen2.5-7B-Instruct-qlora4bit --outtype f16 --verbose --outfile /root/autodl-tmp/llm/Qwen/Qwen2.5-7B-Instruct-qlora4bit.gguf

            
            # 如果需要量化（加速并有损效果），直接执行下面脚本就可以
            python llama.cpp/convert_hf_to_gguf.py /root/autodl-tmp/llm/Qwen/Qwen2.5-7B-Instruct --outtype
            q8_0 --verbose --outfile /root/autodl-tmp/llm/Qwen/Qwen2.5-7B-Instruct-qlora4bit-gguf_q8_0.gguf

            
            说明:
                这里--outtype是输出类型，代表含义：
                q2_k：特定张量（Tensor）采用较高的精度设置，而其他的则保持基础级别。
                q3_k_l、q3_k_m、q3_k_s：这些变体在不同张量上使用不同级别的精度，从而达到性能和效率的平衡。
                q4_0：这是最初的量化方案，使用 4 位精度。
                q4_1 和 q4_k_m、q4_k_s：这些提供了不同程度的准确性和推理速度，适合需要平衡资源使用的场景。
                q5_0、q5_1、q5_k_m、q5_k_s：这些版本在保证更高准确度的同时，会使用更多的资源并且推理速度较
                慢。
                q6_k 和 q8_0：这些提供了最高的精度，但是因为高资源消耗和慢速度，可能不适合所有用户。
                f16 和 f32: 不量化，保留原始精度。
         ```

      2.使用ollama运行gguf
         
         2.1 安装 Ollama
             采用"Ollama下载安装"推荐方式

         2.2 启动 Ollama 服务
             ollama serve

         2.3 创建 ModelFile
             复制模型路径，创建名为 "ModelFile" 的meta文件，内容如下

         ```
            # GGUF文件路径
            /root/autodl-tmp/llm/Qwen/Qwen2.5-7B-Instruct-qlora4bit.gguf
         ```

         2.4 创建自定义模型
             
             使用 ollama create 命令创建自定义模型
             ollama create qwen7B --file /root/autodl-tmp/ModelFile

             命令执行完之后, 就会将模型Qwen2.5-7B-Instruct-qlora4bit.gguf 安装到Ollama仓库中去

             通过命令: ollama list 可查看到模型

         2.5 运行模型：
             ollama run qwen7B

#### LLamaFactory 模型导出量化

     量化打包导出存在问题:

        使用 LLaMA-Factory 最新版本: 0.9.4, 进行量化导包, 会出现依赖冲突问题, 

        LLaMA-Factory需要的 Numpy >=2.0.0, 而 qptq-model 需要的Numpy < 2.0.0,

        
        解决方案: LLaMA-Factory版本降级

            使用 LLaMA-Factory 0.9.2 版本, 安装依赖: "gptq": ["optimum>=1.17.0", "auto-gptq>=0.5.0"]

            ```
                pip install optimum==1.17.0
                pip install auto-gptq==0.5.0
            ```

            降低LLaMA-Factory版本至0.9.3并安装上述依赖即可解决冲突问题



      Ollama+open-webui部署模型

         1.启动Ollama服务
            
            ollama serve
            ollama list
            ollama run qwen3:0.6b

         2.使用 Open WebUI 部署模型
            
            Open WebUI
            Open WebUI 是一个可扩展的、自托管的 AI 界面，可以适应您的工作流程，同时完全离线操作。
            
            仓库：https://github.com/open-webui/open-webui
            文档：https://docs.openwebui.com/

            2.1 创建虚拟环境
                conda create -n open-webui python==3.11
            
            2.2 安装所有依赖
                source activate open-webui
                pip install -U open-webui torch transformers
            
            2.3 运行 ollama
                ollama serve
                ollama运行后会在本地端口暴露一个 openai API 服务，我们后面使用 open-webui 来连接就可以了
            
            2.4 运行 open-webui
                
                由于 ollama 的运行导致原终端阻塞，因此要另外开一个新终端
                
                ```
                  conda activate open-webui
                  export HF_ENDPOINT=https://hf-mirror.com
                  export ENABLE_OLLAMA_API=True
                  export OPENAI_API_BASE_URL=http://127.0.0.1:11434/v1
                  open-webui serve
                ```
            
            2.5 启动浏览器，开始对话

                一切运行正常后，可以通过浏览器输入 http://127.0.0.1:8080 打开 open-webui 面板进行使用。
                
                如果部署在远程服务器则需要把 127.0.0.1 改成对应的 ip 地址（并考虑防火墙问题）。
                
                关于后台持续运行服务，可以使用 tmux/screen/systemd 工具或者 nuhup ... & 等方法，网上教程
                
                非常多，本文在此不叙述


#### vLLM与LMDeploy如何自定义对话模板

     问题: 使用llamafactory 微调训练出来的模型, 在vllm上使用, 会产生差异性, 根本问题是: 使用的对话模版不一致导致


     1.vLLM聊天模板

        为了使语言模型支持聊天协议，vLLM 要求模型在其 tokenizer 配置中包含聊天模板。聊天模板是一个
        Jinja2 模板，用于指定角色、消息和其他特定于聊天的 token 如何在输入中编码。
        可以在 NousResearch/Meta-Llama-3-8B-Instruct 的示例聊天模板 中找到
        有些模型即使经过指令/聊天微调，也不提供聊天模板。对于这些模型，您可以在 --chat-template 参
        数中使用聊天模板的文件路径或字符串形式的模板手动指定其聊天模板。如果没有聊天模板，服务器将
        无法处理聊天，并且所有聊天请求都会出错。

     ```
        vllm serve <model> --chat-template ./path-to-chat-template.jinja
     ```


     示例:
        vllm serve /root/autodl-tmp/llm/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-qlora4bit --chat-template /root/autodl-tmp/llm/deepseek-ai/chat.jinja

     vLLM 社区为流行的模型提供了一组聊天模板。您可以在 examples 目录下找到它们。


     2.LMDeploy自定义对话模板

        LMDeploy 支持两种添加对话模板的形式：一种是利用现有对话模板，直接配置一个如下的 json 文件使用。

        ```
             {
              
                "model_name": "your awesome chat template name",
                "system": "<|im_start|>system\n",
                "meta_instruction": "You are a robot developed by LMDeploy.",
                "eosys": "<|im_end|>\n",
                "user": "<|im_start|>user\n",
                "eoh": "<|im_end|>\n",
                "assistant": "<|im_start|>assistant\n",
                "eoa": "<|im_end|>",
                "separator": "\n",
                "capability": "chat",
                "stop_words": ["<|im_end|>"]
             }

        ```

        model_name 为必填项，可以是 LMDeploy 内置对话模板名（通过 lmdeploy list 可查阅），

        也可以是新名字。其他字段可选填。 当 model_name 是内置对话模板名时，json文件中各非null

        字段会覆盖原有对话模板的对应属性。 而当 model_name 是新名字时，它会把将

        BaseChatTemplate 直接注册成新的对话模板。其具体定义可以参考BaseChatTemplate。

        这样一个模板将会以下面的形式进行拼接。

        {system}{meta_instruction}{eosys}{user}{user_content}{eoh}{assistant}
          
        {assistant_content}{eoa}{separator}{user}...


        在使用 CLI 工具时，可以通过 --chat-template 传入自定义对话模板，比如：

          lmdeploy serve api_server internlm/internlm2_5-7b-chat --chat-template ${JSON_FILE}

      模型: /root/autodl-tmp/llm/Qwen/Qwen2.5-1.5B-qlora

      llama Factory 操作选项: Evaluate & Predict


      推理结果:

        {
            "predict_bleu-4": 24.230159211178275,
            "predict_model_preparation_time": 0.0062,
            "predict_rouge-1": 47.20637316420573,
            "predict_rouge-2": 21.236351610345853,
            "predict_rouge-l": 36.47826706966604,
            "predict_runtime": 2264.3689,
            "predict_samples_per_second": 4.456,
            "predict_steps_per_second": 0.297
        }


#### Xtuner微调大模型

        xtuner微调大模型步骤:
          1.构建虚拟环境
          2.下载模型
          3.微调
          4.微调训练
          5.模型转换
          6.模型合并

       1.构建虚拟环境
         ```
            conda create --name xtuner-env python=3.10 -y
            conda activate xtuner-env
            或
            source activate xtuner-env

            拉取 XTuner
            git clone https://github.com/InternLM/xtuner.git   版本: <= 0.2.0 (旧版本)
         
            安装依赖的
            cd xtuner
            pip install -e '.[all]'

         ```

       2.下载模型
       ```
          from modelscope import snapshot_download
          model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat1_8b',cache_dir='/root/llm/internlm2-1.8b-chat')
       ```

       3.微调
         xtuner 的文件夹里，打开
         xtuner/xtuner/configs/internlm/internlm2_chat_1_8b/internlm2_chat_1_8b_qlora_alpaca_e3.py，
         复制一份至根目录。
         打开这个文件，然后修改预训练模型地址，数据文件地址等

         ```
            ### PART 1中
            # 预训练模型存放的位置
            pretrained_model_name_or_path = '/root/llm/internlm2-1.8b-chat'#基座模型路径
            # 微调数据存放的位置
            data_files = '/root/public/data/target_data.json'

            # 训练中最大的文本长度
            max_length = 512
            # 每一批训练样本的大小
            batch_size = 2
            # 最大训练轮数
            max_epochs = 3
            # 验证数据
            evaluation_inputs = [
              '只剩一个心脏了还能活吗？', '爸爸再婚，我是不是就有了个新娘？',
              '樟脑丸是我吃过最难吃的硬糖有奇怪的味道怎么还有人买','马上要上游泳课了，昨天洗的泳裤还没
              干，怎么办',
              '我只出生了一次，为什么每年都要庆生'
              ]
            # PART 3中
            dataset=dict(type=load_dataset, path="json",data_files=data_files)
            dataset_map_fn=None

         ```

       4.微调训练
         在当前目录下，输入以下命令启动微调脚本

         # 单卡微调
         xtuner train internlm2_chat_1_8b_qlora_alpaca_e3.py
         
         # 多卡微调
         NPROC_PER_NODE=2 xtuner train /home/cw/utils/xtunermain/qwen1_5_1_8b_chat_qlora_alpaca_e3.py --deepspeed deepspeed_zero2
        
         # 多卡指定显卡微调
         CUDA_VISIBLE_DEVICES=0,2 NPROC_PER_NODE=2 xtuner train /home/cw/utils/xtunermain/qwen1_5_1_8b_chat_qlora_alpaca_e3.py --deepspeed deepspeed_zero2

       5.模型转换
         模型训练后会自动保存成 PTH 模型（例如 iter_2000.pth ，如果使用了 DeepSpeed，
         则将会是一个文件夹），我们需要利用 xtuner convert pth_to_hf 将其转换为 HuggingFace 模型，以便于后续使
         用。具体命令为：

         ```
            xtuner convert pth_to_hf ${FINETUNE_CFG} ${PTH_PATH} ${SAVE_PATH}
            # 例如：xtuner convert pth_to_hf internlm2_chat_7b_qlora_custom_sft_e1_copy.py
            ./iter_2000.pth ./iter_2000_

         ```

       6.模型合并
         如果使用了 LoRA / QLoRA 微调，则模型转换后将得到 adapter 参数，而并不包含原 LLM 参数。
         如果您期望获得合并后的模型权重（例如用于后续评测），那么可以利用 xtuner convert merge ：

         $ xtuner convert merge ${LLM} ${LLM_ADAPTER} ${SAVE_PATH}


         附：xtuner中文文档https://xtuner.readthedocs.io/zh-cn/latest/index.html


#### AutoDL 快速下载(选择清华镜像)
      
     微调 Xtuner 框架的安装 (加速安装)
     pip install -e '.[all]' -i https://pypi.tuna.tsinghua.edu.cn/simple

     # 单卡微调
     xtuner train /root/autodl-tmp/pro/utils/xtuner-main/internlm2_chat_1_8b_qlora_alpaca_e3.py

     # 多卡微调
     NPROC_PER_NODE=2 xtuner train /root/autodl-tmp/pro/utils/xtuner-main/qwen1_5_1_8b_chat_qlora_alpaca_e3.py --deepspeed deepspeed_zero2

     # 多卡指定显卡微调
     CUDA_VISIBLE_DEVICES=0,2 NPROC_PER_NODE=2 xtuner train /root/autodl-tmp/pro/utils/xtuner-main/qwen1_5_1_8b_chat_qlora_alpaca_e3.py --deepspeed deepspeed_zero2


#### Xtuner模型转换

     模型训练后会自动保存成 PTH 模型（例如 iter_2000.pth ，如果使用了 DeepSpeed，则将会是一个文件夹），我们需要利用 xtuner convert pth_to_hf 将其转换为 HuggingFace 模型，以便于后续使用。
     
     具体命令为：

        xtuner convert pth_to_hf ${FINETUNE_CFG} ${PTH_PATH} ${SAVE_PATH}
        
        例如：xtuner convert pth_to_hf internlm2_chat_7b_qlora_custom_sft_e1_copy.py ./iter_2000.pth ./iter_2000_


             单卡:
             xtuner convert pth_to_hf /root/autodl-tmp/pro/utils/xtuner-main/qwen1_5_1_8b_chat_qlora_alpaca_e3.py /root/work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/iter_20.pth /root/work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/iter_20_mul_


             多卡:
             xtuner convert pth_to_hf /root/autodl-tmp/pro/utils/xtuner-main/internlm2_chat_1_8b_qlora_alpaca_e3.py /root/work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/iter_20.pth /root/work_dirs/internlm2_chat_1_8b_qlora_alpaca_e3/iter_20_mul_



#### Xtuner模型合并

     如果使用了 LoRA / QLoRA 微调，则模型转换后将得到 adapter 参数，而并不包含原 LLM 参数。如果您期望获得合并后的模型权重（例如用于后续评测），

     那么可以利用 xtuner convert merge ：

       xtuner convert merge ${LLM} ${LLM_ADAPTER} ${SAVE_PATH}


       # 单卡合并:
         xtuner convert merge /root/autodl-tmp/llm/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/work_dirs/internlm2_chat_1_8b_qlora_alpaca_e3/iter_20 /root/work_dirs/internlm2_chat_1_8b_qlora/iter_20_merge


       # 多卡合并:
       xtuner convert merge /root/autodl-tmp/llm/Qwen/Qwen1.5-1.8B-Chat /root/work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/iter_20_mul_ /root/work_dirs/qwen1_5_1_8b_chat_qlora_hf_


     附：xtuner中文文档https://xtuner.readthedocs.io/zh-cn/latest/index.html



#### 知识蒸馏
     
     知识蒸馏,不是微调是模型预训练, 用于模型的初始开发场景中

     模型压缩方法: 

        1.结构或非结构裁剪,  2.线性或非线性量化, 3.知识蒸馏


     

     
    




