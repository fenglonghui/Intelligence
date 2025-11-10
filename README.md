#### 大模型的分布式推理概述案例场景：
     1.单卡显存不足：如QwQ-32B（320亿参数）需在双A6000显卡上部署。2.高并发请求：在线服务需同时处理多用户请求，分布式推理通过连续批处理（Continuous Batching）提升效率。

     注意: Ollama 不支持并发性(即:多卡部署), 支持量化, Ollama、vLLM、LMDeploy都支持量化,  Ollama 是一个多模型框架, vllm、LMDeploy 是单模型框架

#### vLLM的分布式推理实现vLLM通过PagedAttention和张量并行技术优化显存管理和计算效率，支持多GPU推理。
     1. 核心机制
        张量并行：通过tensor_parallel_size参数指定GPU数量，模型自动拆分到多卡。
        PagedAttention：将注意力机制的键值（KV）缓存分块存储，减少显存碎片，提升利用率。
        连续批处理：动态合并不同长度的请求，减少GPU空闲时间。

#### LMDeploy的分布式推理实现LMDeploy是专为高效部署设计的框架，支持量化技术与分布式推理，尤其适合低显存环境。
      1. 核心机制
         张量并行：通过--tp参数指定GPU数量，支持多卡协同计算。KV Cache量化：支持INT8/INT4量化，降低显存占用。动态显存管理：通过--cache-max-entry-count控制KV缓存比例。

####     对于一个7B（70亿）参数的模型，每个参数使用16位浮点数（等于 2个 Byte）表示，则模型的权重大小约为：
          70×10^9 parameters×2 Bytes/parameter=14GB
          70亿个参数×每个参数占用2个字节=14GB
          所以我们需要大于14GB的显存。
          
#### 部署商业大模型时, 需要考虑该项目所支持的用户量上限
          推理大模型时, 需要用到的显存大约为:  用户数据量 + 模型的参数量

#### LMDeploy 量化部署

      LMDeploy 量化部署
         1. 对量化前的模型部署验证
         2. LMDeploy部署/量化InternLM2.5
            2.1 LMDeploy Lite
            2.2 离线转换TurboMind 格式          不推荐, 一般推荐使用在线量化
            2.3 TurboMind 推理+命令行本地对话
         3. 网页 Demo 演示

####     1. 对量化前的模型部署验证
            查询InternLM2.5-7b-chat的config.json文件可知，该模型的权重被存储为bfloat16格式：
            {
              "architectures": [
              "InternLM2ForCausalLM"
              ],
              "attn_implementation": "eager",
              "auto_map": {
              "AutoConfig": "configuration_internlm2.InternLM2Config",
              "AutoModelForCausalLM": "modeling_internlm2.InternLM2ForCausalLM",
              "AutoModel": "modeling_internlm2.InternLM2ForCausalLM"
              },
              "bias": false,
              "bos_token_id": 1,
              "eos_token_id": 2,
              "hidden_act": "silu",
              "hidden_size": 4096,
              "initializer_range": 0.02,
              "intermediate_size": 14336,
              "max_position_embeddings": 32768,
              "model_type": "internlm2",
              "num_attention_heads": 32,
              "num_hidden_layers": 32,
              "num_key_value_heads": 8,
              "pad_token_id": 2,
              "rms_norm_eps": 1e-05,
              "rope_scaling": {
              "type": "dynamic",
              "factor": 2.0
              },
              "rope_theta": 1000000,
              "tie_word_embeddings": false,
              "torch_dtype": "bfloat16",
              "transformers_version": "4.41.0",
              "use_cache": true,
              "vocab_size": 92544,
              "pretraining_tp": 1
            }

####   创建环境，安装依赖
        conda create -n lmdeploy python=3.10 -y
        conda activate lmdeploy
        conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorchcuda=12.1 -c pytorch -c nvidia -y
        #以上基础环境有的可以忽略
        #安装lmdeploy及其依赖
        pip install timm==1.0.8 openai==1.40.3 lmdeploy[all]==0.5.3

####        这个环境搭建报错!!!

####   使用LMdeploy 验证模型
       在量化工作正式开始前，我们还需要验证一下获取的模型文件能否正常工作。进入创建好的环境并启动 InternLM2_5-7b-chat。

       lmdeploy chat /root/models/internlm2_5-7b-chat

####   2. LMDeploy部署/量化InternLM2.5

          InternLM2.5部署
              lmdeploy serve api_server \
                  /root/models/internlm2_5-7b-chat \
                  --model-format hf \
                  --quant-policy 0 \
                  --server-name 0.0.0.0 \
                  --server-port 23333 \
                  --tp 1

          命令说明：
              lmdeploy serve api_server：这个命令用于启动API服务器。
              /root/models/internlm2_5-7b-chat：这是模型的路径。
              --model-format hf：这个参数指定了模型的格式。hf代表“Hugging Face”格式。
              --quant-policy 0：这个参数指定了量化策略。
              --server-name 0.0.0.0：这个参数指定了服务器的名称。在这里，0.0.0.0是一个特殊的IP地址，它表
              示所有网络接口。
              --server-port 23333：这个参数指定了服务器的端口号。在这里，23333是服务器将监听的端口号。
              --tp 1：这个参数表示并行数量（GPU数量）。

          启动完成日志输出
             访问 http://127.0.0.1:23333/ ，可以看到API 信息

          以命令行形式连接API服务器
             lmdeploy serve api_client http://localhost:23333
####   2.1 LMDeploy Lite
           随着模型变得越来越大，我们需要一些大模型压缩技术来降低模型部署的成本，并提升模型的推理性能。LMDeploy 提供了权重量化和 k/v cache两种策略。          

####     设置最大kv cache缓存大小, 进行量化 (推荐量化)
              kv cache是一种缓存技术，通过存储键值对的形式来复用计算结果，以达到提高性能和降低内存消耗的目的。
              在大规模训练和推理中，kv cache可以显著减少重复计算量，从而提升模型的推理速度。理想情况下，kv cache全部存储于显存，以加快访存速度。
              模型在运行时，占用的显存可大致分为三部分：模型参数本身占用的显存、kv cache占用的显存，以及
              中间运算结果占用的显存。LMDeploy的kv cache管理器可以通过设置 --cache-max-entry-count 参数，
              控制kv缓存占用剩余显存的最大比例。默认的比例为0.8。

#####         设置kv 最大比例为0.4，执行如下命令：
              lmdeploy chat /root/models/internlm2_5-7b-chat --cache-max-entry-count 0.4          # K V 量化

#####         设置在线 kv cache int4/int8 量化
              自 v0.4.0 起，LMDeploy 支持在线 kv cache int4/int8 量化，量化方式为 per-head per-token 的非对称量化。
              此外，通过 LMDeploy 应用 kv 量化非常简单，只需要设定 quant_policy 和 cache-max-entrycount 参数。
              目前，LMDeploy 规定 qant_policy=4 表示 kv int4 量化， quant_policy=8 表示 kv int8量化。

              lmdeploy serve api_server \
                  /root/models/internlm2_5-7b-chat \
                  --model-format hf \
                  --quant-policy 4 \
                  --cache-max-entry-count 0.4\
                  --server-name 0.0.0.0 \
                  --server-port 23333 \
                  --tp 1

          相比使用BF16精度的kv cache，int4的Cache可以在相同4GB的显存下只需要4位来存储一个数值，而BF16需要16位。这意味着int4的Cache可以存储的元素数量是BF16的四倍。

          W4A16 模型量化和部署
              准确说，模型量化是一种优化技术，旨在减少机器学习模型的大小并提高其推理速度。量化通过将模型
              的权重和激活从高精度（如16位浮点数）转换为低精度（如8位整数、4位整数、甚至二值网络）来实
              现。
              那么标题中的W4A16又是什么意思呢？
              W4：这通常表示权重量化为4位整数（int4）。这意味着模型中的权重参数将从它们原始的浮点表
              示（例如FP32、BF16或FP16，Internlm2.5精度为BF16）转换为4位的整数表示。这样做可以显
              著减少模型的大小。
              A16：这表示激活（或输入/输出）仍然保持在16位浮点数（例如FP16或BF16）。激活是在神经网
              络中传播的数据，通常在每层运算之后产生。
              因此，W4A16的量化配置意味着：
              权重被量化为4位整数。
              激活保持为16位浮点数。

         在最新的版本中，LMDeploy使用的是AWQ算法，能够实现模型的4bit权重量化。输入以下指令，执行量化工作。(本步骤耗时较长，请耐心等待)

             lmdeploy lite auto_awq \
                  /root/models/internlm2_5-7b-chat \
                  --calib-dataset 'ptb' \
                  --calib-samples 128 \
                  --calib-seqlen 2048 \
                  --w-bits 4 \
                  --w-group-size 128 \
                  --batch-size 1 \
                  --search-scale False \
                  --work-dir /root/models/internlm2_5-7b-chat-w4a16-4bit

             Loading calibrate dataset ...  执行时, 在加载量化校准数据集(allenai/c4)时, 下载不到该数据集而导致执行命令失败!!!!
             该数据集是在国外的, 国内无法下载下来
             不支持自定义数据集
             通过梯子在本地下载该数据集,然后传到服务器中去也无法使用, 只能通过Huggingface上下载该数据集!!!

            命令解释：
                lmdeploy lite auto_awq: lite这是LMDeploy的命令，用于启动量化过程，而auto_awq代表自动权重量化（auto-weight-quantization）。
                /root/models/internlm2_5-7b-chat: 模型文件的路径。
                --calib-dataset 'ptb': 这个参数指定了一个校准数据集，这里使用的是’ptb’（Penn Treebank，一
                个常用的语言模型数据集）。
                --calib-samples 128: 这指定了用于校准的样本数量—128个样本
                --calib-seqlen 2048: 这指定了校准过程中使用的序列长度—1024
                --w-bits 4: 这表示权重（weights）的位数将被量化为4位。
                --work-dir /root/models/internlm2_5-7b-chat-w4a16-4bit: 这是工作目录的路径，用于存储量
                化后的模型和中间结果。

            等待推理完成，便可以直接在你设置的目标文件夹看到对应的模型文件。
            推理后的模型和原本的模型区别是模型文件大小以及占据显存大小。
            我们可以输入如下指令查看在当前目录中显示所有子目录的大小。

        cd /root/models/
        du -sh *

        输出结果如下。(其余文件夹都是以软链接的形式存在的，不占用空间，故显示为0)
            0 InternVL2-26B
            0 internlm2_5-7b-chat
            4.9G internlm2_5-7b-chat-w4a16-4bit
           (lmdeploy) root@intern-studio-50009084:~/models#


        那么原模型大小呢？输入以下指令查看。
            cd /root/share/new_models/Shanghai_AI_Laboratory/
            du -sh *

        对比发现，模型的大小15G 和 4.9G ,差异还是比较大。
        可以输入下面的命令启动量化后的模型

            lmdeploy chat /root/models/internlm2_5-7b-chat-w4a16-4bit/ --model-format awq

        W4A16 量化+ KV cache+KV cache 量化

            输入以下指令，让我们同时启用量化后的模型、设定kv cache占用和kv cache int4量化。

            lmdeploy serve api_server \
                /root/models/internlm2_5-7b-chat-w4a16-4bit/ \
                --model-format awq \
                --quant-policy 4 \
                --cache-max-entry-count 0.4\
                --server-name 0.0.0.0 \
                --server-port 23333 \
                --tp 1

      2.2 离线转换TurboMind 格式

          离线转换需要在启动服务之前，将模型转为 lmdeploy TurboMind 的格式，如下所示。
          # 转换模型（FastTransformer格式） TurboMind
          lmdeploy convert internlm-chat-7b /path/to/internlm-chat-7b

          执行完成后将会在当前目录生成一个 workspace 的文件夹。这里面包含的就是 TurboMind 和 Triton
          “模型推理”需要到的文件。
          目录如下图所示。

            weights 和 tokenizer 目录分别放的是拆分后的参数和 Tokenizer。如果我们进一步查看 weights 的目录，就会发现参数是按层和模块拆开的，如下图所示


          每一份参数第一个 0 表示“层”的索引，后面的那个0表示 Tensor 并行的索引，因为我们只有一张卡，所以被拆分成 1 份。
          如果有两张卡可以用来推理，则会生成0和1两份，也就是说，会把同一个参数拆成两份。
          比如 layers.0.attention.w_qkv.0.weight 会变成 layers.0.attention.w_qkv.0.weight 和 layers.0.attention.w_qkv.1.weight 。
          执行 lmdeploy convert 命令时，可以通过 --tp 指定（tp 表示 tensor parallel，该参数默认值为1也就是一张卡）。

      2.3 TurboMind 推理+命令行本地对话
          模型转换完成后，我们就具备了使用模型推理的条件，接下来就可以进行真正的模型推理环节。
          我们可以尝试本地对话，在这里其实是跳过 API Server 直接调用 TurboMind
          执行命令如下。

          lmdeploy chat ./workspace

          启动后就可以和它进行对话了。

      3. 网页 Demo 演示

          这一部分主要是将 Gradio 作为前端 Demo 演示。
          # Gradio+ApiServer。必须先开启 Server，此时 Gradio 为 Client
          lmdeploy serve gradio http://0.0.0.0:23333

          
