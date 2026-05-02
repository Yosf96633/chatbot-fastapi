from typing import TypedDict , Annotated , List
from langgraph.graph import  add_messages
from langchain_core.messages import BaseMessage
from typing import Union
class chatState(TypedDict):
    messages : Annotated[List[BaseMessage] , add_messages]
    summary : Union[str , None]