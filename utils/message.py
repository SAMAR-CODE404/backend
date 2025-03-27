from dataclasses import dataclass
@dataclass
class HumanMessage:
    content: str
    role: str = "user"

@dataclass
class AIMessage:
    content: str
    role: str = "assistant"
