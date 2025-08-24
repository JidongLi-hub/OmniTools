# Omni-Tools 
本项目是一个集成了大量先进agent工具的分布式系统，可以为全模态agent提供有力的工具调用支持
### 代码架构
```sh
tool_server/    # 工具集总文件夹
├──tool_workers/   # 工具集实现和调用文件夹
      ├── offline_workers  # 一些自己本地实现的纯代码工具
      ├── online_workers  # 使用现有的模型和各种开源代码的工具
            ├── *_worker.py  # 某一工具的具体实现
      ├── scripts/launch_scripts  # 启动脚本和配置文件
            ├── config
            ├── start_server_local.py # 工具集启动脚本
```
- 整体思路：启动进程启动后，将每个工具作为一个子进程部署在某个端口上提供服务，所有的工具都通过一个controller进行管理（包括启动后记录工具所在的端口，列出所有服务等）
- 工具集tool_server的实现
- 整体思路：启动一个主进程，通过subprocess.Popen()为每个工具开启一个子进程（具体为执行终端命令运行单个工具的python文件），形成一个分布式工具调用系统，通过FastAPI让每个工具的服务部署在本地的某个端口上，工具调用通过request请求实现。
- 工具的config使用yaml文件配置，包括python程序，端口等
- start_server_local.py中实现了控制器和工具的加载
    - 控制器负责注册各个工具和管理他们的访问地址（localhost:8888）
    - 工具启动后在某个端口监听，接受到request请求，便产生响应回复
- 工具的具体实现（以GroundingDINO为例）
    - 继承BaseToolWorker类，其中使用 FastAPI 框架规定了API 接口，如 /worker_generate；同时还使用一个线程信号量limit_model_concurrency来限制工作节点可以同时处理的请求数
    - 在指定的GPU上加载模型
    - 最终通过实现父类的抽象方法generate来根据post请求中接收到的参数，产生模型的输出
- 其他细节：使用信号量管理对于公共资源的访问，使用log文件记录每个工具的使用日志，程序结束后杀死所有子进程等。
### 如何新增一个自己的新工具 （以Qwen2.5-VL-7B为例，一个工具两个功能，视频caption和视频问答）
- 第一步：在tool_server/tool_workers/online_workers/文件夹下新建自己的工具实现文件
    - 新建Qwen2.5-VL_worker.py
    （注意：模型所有的推理都在本文件中实现，包括对于图片的处理等操作）
    - 创建Qwen2.5VLWorker类，该类必须继承BaseToolWorker，并实现以下关键函数
        - __init__:用于传入一些该工具必须的参数，并初始化
        - init_model:加载模型及其所需要的组件
        - get_tool_instruction:返回对于工具类的调用说明
      - 在该文件中实现工具启动
    - 第二步：在配置文件tool_workers/scripts/launch_scripts/config/all_service_example_local.yaml中加入新工具的启动配置
    - 第三步：启动工具集系统，加载包括新工具在内的所有工具
    - 第四步：编写测试代码，检验功能是否正常
      - 在tool_server/tool_workers/online_workers/test_cases中新建test_QwenVL_messages.py