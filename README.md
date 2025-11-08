#### OpenCompass文档
    OpenCompass 是一个开源项目，旨在为机器学习和自然语言处理领域提供多功能、易于使用的工具和框架。其中包含的多个开源模型和开源数据集（BenchMarks），方便进行模型的效果评测

    生成式大模型的评估指标
        1. 核心评估指标
           OpenCompass支持以下主要评估指标，覆盖生成式大模型的多样化需求：
              a1 准确率（Accuracy）：用于选择题或分类任务，通过比对生成结果与标准答案计算正确率。在OpenCompass中通过metric=accuracy配置
              a2 困惑度（Perplexity, PPL）：衡量模型对候选答案的预测能力，适用于选择题评估。需使用ppl类型的数据集配置（如ceval_ppl）
              a3 生成质量（GEN）：通过文本生成结果提取答案，需结合后处理脚本解析输出。使用gen类型的数据集（如ceval_gen），配置metric=gen并指定后处理规则
              a4 ROUGE/LCS：用于文本生成任务的相似度评估，需安装rouge==1.0.1依赖，并在数据配置中设置metric=rouge
              a5 条件对数概率（CLP）：结合上下文计算答案的条件概率，适用于复杂推理任务，需在模型配置中启用use_logprob=True

    支持的开源评估数据集及使用差异
        主流开源数据集
        OpenCompass内置超过70个数据集，覆盖五大能力维度：
            a.知识类：C-Eval（中文考试题）、CMMLU（多语言知识问答）、MMLU（英文多选题）。
            b.推理类：GSM8K（数学推理）、BBH（复杂推理链）。
            c.语言类：CLUE（中文理解）、AFQMC（语义相似度）。
            d.代码类：HumanEval（代码生成）、MBPP（编程问题）。
            e.多模态类：MMBench（图像理解）、SEED-Bench（多模态问答）。


    数据集区别与选择
       评估范式差异：
          _gen后缀数据集：生成式评估，需后处理提取答案（如ceval_gen）
          _ppl后缀数据集：困惑度评估，直接比对选项概率（如ceval_ppl）
          领域覆盖：
          C-Eval：侧重中文STEM和社会科学知识，包含1.3万道选择题
          LawBench：法律领域专项评估，需额外克隆仓库并配置路径
  

    1.基础安装
    2. 安装 OpenCompass
    3.数据集准备
    4.配置评估任务

    命令行（自定义HF模型）
    命令行  配置文件
    5.自定义数据集
      数据集格式
         选择题 ( mcq )
         问答题 ( qa )
    6.命令行列表


OpenCompass 使用引导:
    1.基础安装
      使用Conda准备 OpenCompass 运行环境：
      conda create --name opencompass python=3.10 -y
      # conda create --name opencompass_lmdeploy python=3.10 -y
      conda activate opencompass
    
    2. 安装 OpenCompass
       git clone https://github.com/open-compass/opencompass opencompass
       cd opencompass
       pip install -e .
    3.数据集准备
      在 OpenCompass 项目根目录下运行下面命令，将数据集准备至 ${OpenCompass}/data 目录下：
        wget https://github.com/opencompass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core20240207.zip
        unzip OpenCompassData-core-20240207.zip
    4.配置评估任务
      命令行（自定义HF模型）
      对于 HuggingFace 模型，用户可以通过命令行直接设置模型参数，无需额外的配置文件。例如，对于
      internlm/internlm2-chat-1_8b 模型，可以使用以下命令进行评估：
      python run.py \
          --datasets demo_gsm8k_chat_gen demo_math_chat_gen \
          --hf-path internlm/internlm2-chat-1_8b \
          --debug
      请注意，通过这种方式，OpenCompass 一次只评估一个模型，而其他方式可以一次评估多个模型。
    
      命令行
        用户可以使用 --models 和 --datasets 结合想测试的模型和数据集。
        python run.py \
          --models hf_internlm2_chat_1_8b hf_qwen2_1_5b_instruct \
          --datasets demo_gsm8k_chat_gen demo_math_chat_gen \
          --debug
        模型和数据集的配置文件预存于 configs/models 和 configs/datasets 中。用户可以使用tools/list_configs.py 查看或过滤当前可用的模型和数据集配置。
        
        运行 python tools/list_configs.py llama mmlu 将产生如下输出：
        +-----------------+-----------------------------------+
        | Model | Config Path |
        |-----------------+-----------------------------------|
        | hf_llama2_13b | configs/models/hf_llama2_13b.py |
        | hf_llama2_70b | configs/models/hf_llama2_70b.py |
        | ... | ... |
        +-----------------+-----------------------------------+
        +-------------------+---------------------------------------------------+
        | Dataset | Config Path |
        |-------------------+---------------------------------------------------|
        | cmmlu_gen | configs/datasets/cmmlu/cmmlu_gen.py |
        | cmmlu_gen_ffe7c0 | configs/datasets/cmmlu/cmmlu_gen_ffe7c0.py |
        | ... | ... |
        +-------------------+---------------------------------------------------+
    
      用户可以使用第一列中的名称作为 python run.py 中 --models 和 --datasets 的输入参数。对于数据集，同一名称的不同后缀通常表示其提示或评估方法不同。
      
      配置文件
          除了通过命令行配置实验外，OpenCompass 还允许用户在配置文件中编写实验的完整配置，并通过run.py 直接运行它。配置文件是以 Python 格式组织的，并且必须包括 datasets 和 models 字段。
          from mmengine.config import read_base
          with read_base():
          from .datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
          from .datasets.demo.demo_math_chat_gen import math_datasets
          from .models.qwen.hf_qwen2_1_5b_instruct import models as
          hf_qwen2_1_5b_instruct_models
          from .models.hf_internlm.hf_internlm2_chat_1_8b import models as
          hf_internlm2_chat_1_8b_models
          datasets = gsm8k_datasets + math_datasets
          models = hf_qwen2_1_5b_instruct_models + hf_internlm2_chat_1_8b_models
    
    运行任务时，我们只需将配置文件的路径传递给 run.py ：
      python run.py configs/eval_chat_demo.py --debug
    
    5.自定义数据集
      数据集格式
      选择题 ( mcq )
    
      对于选择 ( mcq ) 类型的数据，默认的字段如下：
      question : 表示选择题的题干
      A , B , C , …: 使用单个大写字母表示选项，个数不限定。默认只会从 A 开始，解析连续的字母作
      为选项。
      answer : 表示选择题的正确答案，其值必须是上述所选用的选项之一，如 A , B , C 等。
      对于非默认字段，我们都会进行读入，但默认不会使用。如需使用，则需要在 .meta.json 文件中进行
      指定。
      .jsonl 格式样例如下：
    
          {"question": "165+833+650+615=", "A": "2258", "B": "2263", "C": "2281", "answer":
        "B"}
        {"question": "368+959+918+653+978=", "A": "3876", "B": "3878", "C": "3880",
        "answer": "A"}
        {"question": "776+208+589+882+571+996+515+726=", "A": "5213", "B": "5263", "C":
        "5383", "answer": "B"}
        {"question": "803+862+815+100+409+758+262+169=", "A": "4098", "B": "4128", "C":
        "4178", "answer": "C"}
    
      .csv 格式样例如下:
          question,A,B,C,answer
          127+545+588+620+556+199=,2632,2635,2645,B
          735+603+102+335+605=,2376,2380,2410,B
          506+346+920+451+910+142+659+850=,4766,4774,4784,C
          504+811+870+445=,2615,2630,2750,B
    
      问答题 ( qa )
      对于问答 ( qa ) 类型的数据，默认的字段如下：
      question : 表示问答题的题干
      answer : 表示问答题的正确答案。可缺失，表示该数据集无正确答案。
      对于非默认字段，我们都会进行读入，但默认不会使用。如需使用，则需要在 .meta.json 文件中进行
      指定。
      .jsonl 格式样例如下：
    
        {"question": "752+361+181+933+235+986=", "answer": "3448"}
        {"question": "712+165+223+711=", "answer": "1811"}
        {"question": "921+975+888+539=", "answer": "3323"}
        {"question": "752+321+388+643+568+982+468+397=", "answer": "4519"}
    
      .csv 格式样例如下:
          question,answer
          123+147+874+850+915+163+291+604=,3967
          149+646+241+898+822+386=,3142
          332+424+582+962+735+798+653+214=,4700
          649+215+412+495+220+738+989+452=,4170
    
    6.命令行列表
    
       自定义数据集可直接通过命令行来调用开始评测。
    
       python run.py \
          --models hf_llama2_7b \
          --custom-dataset-path xxx/test_mcq.csv \
          --custom-dataset-data-type mcq \
          --custom-dataset-infer-method ppl
    
      python run.py \
          --models hf_llama2_7b \
          --custom-dataset-path xxx/test_qa.jsonl \
          --custom-dataset-data-type qa \
          --custom-dataset-infer-method gen
    
    在绝大多数情况下， --custom-dataset-data-type 和 --custom-dataset-infer-method 可以省略，OpenCompass 会根据以下逻辑进行设置：
        如果从数据集文件中可以解析出选项，如 A , B , C 等，则认定该数据集为 mcq ，否则认定为qa 。
        默认 infer_method 为 gen 。
