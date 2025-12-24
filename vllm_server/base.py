from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

class BaseModelRunner(ABC):
    """Abstract base class for model-specific runners."""
    
    def __init__(self, model: str):
        self.model = model
    
    @abstractmethod
    async def run_model_completion(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Any:
        """Run model completion - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def extract_message_content(self, response: Any) -> str:
        """Extract message content from model response."""
        pass
    
    @abstractmethod
    def extract_tool_calls(self, response: Any) -> List[Any]:
        """Extract tool calls from model response."""
        pass
    
    @abstractmethod
    def create_conversation_message(self, response: Any) -> Dict[str, Any]:
        """Create conversation message dict from model response."""
        pass
    
    def get_model_specific_system_prompt(self) -> str:
        """Get model-specific system prompt additions."""
        return ""