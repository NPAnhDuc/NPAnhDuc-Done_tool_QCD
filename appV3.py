from langchain_community.tools.mcp import MCPTool
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType

# 1. Khởi tạo "Bộ não" (Qwen 3 Local)
llm = ChatOllama(model="qwen2.5") 

# 2. Kết nối "Công cụ Jira" vào bộ não
jira_tool = MCPTool.from_server(
    command="python", 
    args=["jira_mcp_server.py"] # Server bạn vừa tạo
)

# 3. Tạo Agent (Người điều phối)
agent = initialize_agent(
    tools=[jira_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 4. THỬ NGHIỆM: Bây giờ bạn mới thực sự thấy nó làm việc
agent.run("Hãy tìm cho tôi các ticket có summary chứa từ khóa 'Lỗi hệ thống' trên Jira")