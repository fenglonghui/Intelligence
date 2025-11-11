### 多模态大模型的概念与本地部署调用

    什么是大模型?
    基于Transformer架构实现的参数量在1B及以上的人工神经网络模型.
    
    什么是多模态?
    事物的表达或感知的方式, 本质是数据形态, 以多种不同的存在形式而存在.
    比如: 传递一个信号, 想把苹果传递过来, 你可以使用一段文本, 一张图片, 一个视频, 一个音频表示出来, 这里的每个数据形式否是模态!!!
    
    总结起来就是 图像、语音、自然语义 这几种形式存在!!!


#### 多模态典型任务

    1.跨模态预训练
      图像/视频与语言预训练。
      跨任务预训练
    
    2.Language-Audio
      Text-to-Speech Synthesis: 给定文本，生成一段对应的声音。
      Audio Captioning：给定一段语音，生成一句话总结并描述主要内容。(不是语音识别)
    
    3.Vision-Audio
      Audio-Visual Speech Recognition(视听语音识别)：给定某人的视频及语音进行语音识别。Video Sound Separation(视频声源分离)：给定视频和声音信号(包含多个声源)，进行声源定位与分离。Image Generation from Audio: 给定声音，生成与其相关的图像。
      Speech-conditioned Face generation：给定一段话，生成说话人的视频。Audio-Driven 3D Facial Animation：给定一段话与3D人脸模版，生成说话的人脸3D动画。

    4.Vision-Language
      Image/Video-Text Retrieval (图(视频)文检索): 图像/视频<-->文本的相互检索。
      Image/Video Captioning(图像/视频描述)：给定一个图像/视频，生成文本描述其主要内容。
      Visual Question Answering(视觉问答)：给定一个图像/视频与一个问题，预测答案。
      Image/Video Generation from Text：给定文本，生成相应的图像或视频。
      Multimodal Machine Translation：给定一种语言的文本与该文本对应的图像，翻译为另外一种语言。
      Vision-and-Language Navigation(视觉-语言导航)： 给定自然语言进行指导，使得智能体根据视觉传感器导航到特定的目标。
      Multimodal Dialog(多模态对话)： 给定图像，历史对话，以及与图像相关的问题，预测该问题的回答。

    5.定位相关的任务
      Visual Grounding：给定一个图像与一段文本，定位到文本所描述的物体。
      Temporal Language Localization: 给定一个视频即一段文本，定位到文本所描述的动作(预测起止时间)。
      Video Summarization from text query：给定一段话(query)与一个视频，根据这段话的内容进行视频摘要，预测视频关键帧(或关键片段)组合为一个短的摘要视频。
      Video Segmentation from Natural Language Query: 给定一段话(query)与一个视频，分割得到query所指示的物体。
      Video-Language Inference: 给定视频(包括视频的一些字幕信息)，还有一段文本假设(hypothesis)，判断二者是否存在语义蕴含(二分类)，即判断视频内容是否包含这段文本的语义。Object Tracking from Natural Language Query: 给定一段视频和一些文本，追踪视频中文本所描述的对象。
      Language-guided Image/Video Editing: 一句话自动修图。给定一段指令(文本)，自动进行图像/视频的编辑。

#### 多模态（多模态大模型的概念与本地部署调用）
      1. 本地部署CogVideoX-5B文生视频模型
      1.1 模型介绍
      1.2 环境安装
      1.3模型下载
      1.4 运行代码
      1.5 生成效果
      2. 使用ollama部署Llama-3.2-11B-Vision-Instruct-GGUF实现视觉问答
      2.1 模型介绍
      2.2 预期用途
      2.3 安装ollama
      2.4 安装 Llama 3.2 Vision 模型
      2.5 运行 Llama 3.2-Vision

##### 1. 本地部署CogVideoX-5B文生视频模型
      模型介绍
          CogVideoX是 清影 同源的开源版本视频生成模型。下表展示我们在本代提供的视频生成模型列表相关信息。

      具体参数参照 modelscope 或 hunggingface上模型的参数

      1.2 环境安装
          # diffusers>=0.30.3
          # transformers>=0.44.2
          # accelerate>=0.34.0
          # imageio-ffmpeg>=0.5.1
          pip install --upgrade transformers accelerate diffusers imageio-ffmpeg

      1.3 模型下载
          git clone https://www.modelscope.cn/ZhipuAI/CogVideoX-5b.git

      1.4 运行代码
          import torch
          from diffusers import CogVideoXPipeline
          from diffusers.utils import export_to_video
          prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a
          wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature
          acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas
          gather, watching curiously and some clapping in rhythm. Sunlight filters through
          the tall bamboo, casting a gentle glow on the scene. The panda's face is
          expressive, showing concentration and joy as it plays. The background includes a
          small, flowing stream and vibrant green foliage, enhancing the peaceful and
          magical atmosphere of this unique musical performance."
          pipe = CogVideoXPipeline.from_pretrained(
          "THUDM/CogVideoX-5b",
          torch_dtype=torch.bfloat16
          )
          pipe.enable_sequential_cpu_offload()
          pipe.vae.enable_tiling()
          pipe.vae.enable_slicing()
          video = pipe(
          prompt=prompt,
          num_videos_per_prompt=1,
          num_inference_steps=50,
          num_frames=49,
          guidance_scale=6,
          generator=torch.Generator(device="cuda").manual_seed(42),
          ).frames[0]
          export_to_video(video, "output.mp4", fps=8)

      1.5 生成效果
          忽略……!!!
        
#### 2. 使用ollama部署Llama-3.2-11B-Vision-InstructGGUF实现视觉问答

        2.1 模型介绍
            Llama 3.2-Vision 是一系列多模态大语言模型（LLM），包括预训练和指令调优的图像推理生成模型，
            大小分别为11B和90B（输入为文本+图像/输出为文本）。Llama 3.2-Vision 指令调优模型针对视觉识
            别、图像推理、字幕生成以及回答关于图像的一般问题进行了优化。这些模型在常见的行业基准测试中
            表现优于许多可用的开源和闭源多模态模型。

            模型开发者: Meta
               模型架构: Llama 3.2-Vision 基于 Llama 3.1 文本模型构建，后者是一个使用优化的Transformer架构的
               自回归语言模型。调优版本使用有监督的微调（SFT）和基于人类反馈的强化学习（RLHF）来与人类对
               有用性和安全性的偏好保持一致。为了支持图像识别任务，Llama 3.2-Vision 模型使用了单独训练的视
               觉适配器，该适配器与预训练的 Llama 3.1 语言模型集成。适配器由一系列交叉注意力层组成，将图像
               编码器表示传递给核心LLM。

            支持的语言: 对于纯文本任务，官方支持英语、德语、法语、意大利语、葡萄牙语、印地语、西班牙语和
            泰语。Llama 3.2 的训练数据集包含了比这八种语言更广泛的语言。注意，对于图像+文本应用，仅支持
            英语。
            开发者可以在遵守 Llama 3.2 社区许可证和可接受使用政策的前提下，对 Llama 3.2 模型进行其他语言
            的微调。开发者始终应确保其部署，包括涉及额外语言的部署，是安全且负责任的。
            模型发布日期: 2024年9月25日


      2.2 预期用途
          预期用途案例： Llama 3.2-Vision旨在用于商业和研究用途。经过指令调优的模型适用于视觉识别、图
          像推理、字幕添加以及带有图像的助手式聊天，而预训练模型可以适应多种图像推理任务。此外，由于
          Llama 3.2-Vision能够接受图像和文本作为输入，因此还可能包括以下用途：
          1. 视觉问答（VQA）与视觉推理：想象一台机器能够查看图片并理解您对其提出的问题。
          2. 文档视觉问答（DocVQA）：想象计算机能够理解文档（如地图或合同）中的文本和布局，并直接
          从图像中回答问题。
          3. 图像字幕：图像字幕架起了视觉与语言之间的桥梁，提取细节，理解场景，然后构造一两句讲述故
          事的话。
          4. 图像-文本检索：图像-文本检索就像是为图像及其描述做媒人。类似于搜索引擎，但这种引擎既理
          解图片也理解文字。
          5. 视觉接地：视觉接地就像将我们所见与所说连接起来。它关乎于理解语言如何引用图像中的特定部
          分，允许AI模型基于自然语言描述来精确定位对象或区域。

####   2.3 安装ollama
          # ollama版本需大于等于0.4.0
          curl -fsSL https://ollama.com/install.sh | sh
          # 查看ollama版本
          ollama --version

####   2.4 安装 Llama 3.2 Vision 模型
           ollama run llama3.2-vision:11b

####   2.5 运行 Llama 3.2-Vision
           将 images.png` 替换为你选择的图像路径。模型将分析图像并根据其理解提供响应。

            ollama run x/llama3.2-vision:latest "Which era does this piece belong to? Give details about the era: images.png"

####   2.6 推荐模型

           1. Wan-AI/Wan2.2-Animate-14B。角色动画和替换, https://modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B
              视频驱动图片

           2. Wan-AI/Wan2.2-S2V-14B, 音频驱动的电影视频生成, https://modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B
              音频驱动的电影视频生成
               

           
          
