import os
import instructor
import yaml
from dotenv import load_dotenv
from enum import Enum
from atomic_agents import (
    AtomicAgent,
    AgentConfig,
    BasicChatInputSchema,
    BasicChatOutputSchema,
)
from atomic_agents.context import ChatHistory, SystemPromptGenerator

# Get .env variables
load_dotenv()

# History with message limit (oldest messages removed when exceeded)
history = ChatHistory(max_messages=100)


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"


def get_client(provider=Provider):
    """Get instructor client for specified provider."""
    if provider == Provider.OLLAMA:
        from langfuse.openai import OpenAI

        url = os.getenv("OLLAMA_BASE_URL")  # Get URL from .env
        client = instructor.from_openai(
            OpenAI(base_url=url, api_key="ollama"), mode=instructor.Mode.JSON
        )
        return client, "gpt-oss:20b-cloud"
    raise ValueError(f"Unknown provider: {provider}")


def get_agent_persona(agent_name):
    """Define custom agent personality."""
    with open("data/agents.yaml", "r") as f:
        config_data = yaml.safe_load(f)

    agent_config = config_data.get(agent_name, {})

    # Create system prompt
    agent_prompt = SystemPromptGenerator(
        background=agent_config.get("background", []),
        steps=agent_config.get("steps", []),
        output_instructions=agent_config.get("output_instructions", []),
    )
    return agent_prompt


def create_agent(provider: Provider, agent_name) -> AtomicAgent:
    """Create agent with specified provider."""
    client, model = get_client(provider)

    return AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        config=AgentConfig(
            client=client,
            model=model,
            history=history,
            system_prompt_generator=get_agent_persona(agent_name),
        )
    )
