import ollama


class Agent:
    """A generic agent that uses Ollama to discuss topic relevance."""

    def __init__(self, name, expertise):
        self.name = name
        self.expertise = expertise  # The specific topic or stance of the agent

    def discuss(self, input_text, topic):
        """Discusses if the input text relates to a given topic using Ollama."""
        # Define the prompt to ask Ollama
        messages = [
            {
                "role": "system",
                "content": f"You are an expert with the following expertise: {self.expertise}. Please evaluate if the following text is related to {topic}. Respond with 'Relevant' or 'Not Relevant' only.",
            },
            {
                "role": "user",
                "content": input_text,
            },
        ]

        # Call Ollama chat
        response = ollama.chat(
            model="llama3.2",
            messages=messages,
        )

        # Process response
        relevance = response["message"]["content"].strip()
        print(
            f"{self.name} thinks the input is {relevance} to '{topic}'."
        )


class MixtureOfAgents:
    """Environment where agents discuss input relevance to a topic."""

    def __init__(self):
        # Initialize agents with different expertise
        self.agents = [
            Agent(
                name="Quality_Assurance",
                expertise="QA Officers are responsible for establishing and monitoring quality standards and compliance protocols across product lifecycles, from development to market approval. They conduct audits, oversee Good Manufacturing Practice (GMP) compliance, and ensure that product development and manufacturing processes meet regulatory standards. This role also involves identifying and mitigating risks to quality and compliance, often liaising with production and clinical teams to implement corrective and preventive actions as needed.",
            ),
            Agent(
                name="Clinical_Evaluator",
                expertise="Clinical Evaluators focus on assessing the safety, efficacy, and quality of products through reviewing clinical trial data and real-world evidence. They analyze clinical study designs, results, and adverse event reports to provide evidence-based recommendations.",
            ),
            Agent(
                name="Regulatory_Affairs_Specialist",
                expertise="This role involves coordinating and managing regulatory submissions, ensuring that all product data and documentation comply with local and international regulations. Regulatory Affairs Specialists often work closely with product developers to prepare, submit, and follow up on applications with regulatory bodies.",
            ),
        ]

    def conduct_discussion(self, input_text, topic):
        print(f"Discussion on topic: '{topic}' for input: '{input_text}'\n")
        for agent in self.agents:
            agent.discuss(input_text, topic)


# Instantiate the environment and conduct a discussion
environment = MixtureOfAgents()
input_text = """
The high-throughput and liver-on-chip systems exhibit enhanced in vivo-like functions and demonstrate the potential utility of these platforms for DILI risk assessment. Tenofovir-inarigivr-associated hepatotoxicity was observed and correlates with the clinical manifestation of DILI observed in patients."""
topic = "DILI"
environment.conduct_discussion(input_text, topic)