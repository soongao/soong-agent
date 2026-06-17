# Agent Hub

## 需求
- 类比成一个成员是agent的im系统
- 用户可以在会话中发消息，可以添加worker进入，可以@某个worker去干活
- 可以创建worker，指定woker名称、简介、使用模型等等
- 可以@Orchestrator去拆分任务，然后让Orchestrator自动去派活
- 如果不@任何人，则默认是和Orchestrator对话

## backend
- 调用agent core去实现核心功能
- 注册几个新的sub agent(worker)
- 使用Orchestrator模式
- 调用功能封装成http api，供前端去请求
- 用户要能看到Orchestrator的返回，以及每个worker的返回
- 可以在会话中随时配置worker，修改worker参数，删除新增worker等等

## frontend
- 使用ts语言
- 桌面端，而非网页端