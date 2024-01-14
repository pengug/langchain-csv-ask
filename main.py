from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
import pandas as pd

df = pd.read_csv('kalite.csv')
print(df.columns)


openai_key = "open-ai-key"




agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", api_key=openai_key),
    "kalite2.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)


agent.run("dosya kaç satır")