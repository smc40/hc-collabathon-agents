from swarms.structs.moa import MOA, Agent as SwarmAgent, Message


class Agent(SwarmAgent):
    """A generic agent that uses Swarms to discuss topic relevance."""

    def __init__(self, name, expertise):
        super().__init__(name=name)  # Initialize with Swarms agent structure
        self.expertise = expertise  # The specific topic or stance of the agent

    def process_message(self, message: Message):
        """Processes a message to evaluate relevance."""
        input_text = message.content
        topic = message.metadata.get("topic")

        # Logic to evaluate relevance based on expertise
        if self.expertise in input_text:  # Simplified check for example
            relevance = "Relevant"
        else:
            relevance = "Not Relevant"

        print(f"{self.name} thinks the input is {relevance} to '{topic}'.")


class MixtureOfAgents(MOA):
    """Environment where agents discuss input relevance to a topic."""

    def __init__(self):
        # Initialize agents with different expertise
        agents = [
            Agent(
                name="Quality_Assurance",
                expertise="QA Officers are responsible for establishing and monitoring quality standards...",
            ),
            Agent(
                name="Clinical_Evaluator",
                expertise="Clinical Evaluators focus on assessing the safety, efficacy, and quality of products...",
            ),
            Agent(
                name="Regulatory_Affairs_Specialist",
                expertise="This role involves coordinating and managing regulatory submissions...",
            ),
        ]
        super().__init__(agents=agents)

    def conduct_discussion(self, input_text, topic):
        print(f"Discussion on topic: '{topic}' for input: '{input_text}'\n")
        
        message = Message(content=input_text, metadata={"topic": topic})
        
        # Dispatch the message to each agent in the MOA
        for agent in self.agents:
            agent.process_message(message)


# Instantiate the environment and conduct a discussion
environment = MixtureOfAgents()
input_text = """
The high-throughput and liver-on-chip systems exhibit enhanced in vivo-like functions and demonstrate the potential utility of these platforms for DILI risk assessment. Tenofovir-inarigivr-associated hepatotoxicity was observed and correlates with the clinical manifestation of DILI observed in patients."""
topic = "DILI"
environment.conduct_discussion(input_text, topic)
