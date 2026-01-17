from src.ai_config import Provider, create_agent
from src.ai_schema import ChatInput, ChatOutput
from rich.console import Console
from atomic_agents.context import ChatHistory

# Initialize console for pretty outputs
console = Console()


def main():
    # Create agent with type parameters
    agent = create_agent(Provider.OLLAMA, agent_name="AIDen")

    # Start a loop to handle user inputs and agent responses
    while True:
        # Prompt the user for input
        user_input = console.input("[bold blue]You:[/bold blue] ")
        # Check if the user wants to exit the chat
        if user_input.lower() in ["/exit", "/quit"]:
            console.print("Exiting chat...")
            break

        # Process the user's input through the agent and get the response
        input_schema = ChatInput(message=user_input)
        response = agent.run(input_schema)

        # Display the agent's response
        console.print("Agent: ", response.chat_message)


if __name__ == "__main__":
    main()
