from atomic_agents import BaseIOSchema
from pydantic import Field


class ChatInput(BaseIOSchema):
    """User chat message"""

    message: str = Field(..., description="The user's message")


class ChatOutput(BaseIOSchema):
    """Assistant response"""

    response: str = Field(..., description="The assistant's response")
