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
                "content": f"You are an expert with the following expertise: {self.expertise}. Please evaluate if the following text is important to {topic}. Respond with 'Relevant' or 'Not Relevant' only.",
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
        return relevance  # Return the relevance for majority voting


class MixtureOfAgents:
    """Environment where agents discuss input relevance to a topic."""

    def __init__(self):
        # Initialize agents with different expertise
        self.agents = [
            Agent(
                name="Hepatocellular_Classifier",
                expertise="Specializes in identifying hepatocellular injuries, which are characterized by liver cell damage predominantly within the hepatocytes. This role involves analyzing liver enzyme levels, primarily ALT (alanine aminotransferase), and assessing data for patterns consistent with hepatocellular damage. This classifier evaluates potential liver injury risks associated with drug compounds."
            ),
            Agent(
                name="Cholestatic_Classifier",
                expertise="Focuses on detecting cholestatic injuries, where the liver's bile flow is obstructed or slowed. This involves assessing indicators like elevated ALP (alkaline phosphatase) levels and other markers of bile duct or canalicular damage. This classifier evaluates drugs for potential adverse impacts on bile flow."
            ),
            Agent(
                name="Mixed_Hepatocellular_Cholestatic_Classifier",
                expertise="Specializes in identifying mixed liver injury patterns that show both hepatocellular and cholestatic characteristics. This classifier assesses cases where both ALT and ALP levels are elevated, indicating a combination of liver cell damage and bile flow obstruction. It provides a balanced evaluation of drug-induced liver injury risks."
            ),
        ]

    def conduct_discussion(self, input_text, topic):
        print(f"Discussion on topic: '{topic}' for input: '{input_text}'\n")
        results = []
        for agent in self.agents:
            relevance = agent.discuss(input_text, topic)
            results.append(relevance)
        self.majority_vote(results)

    def majority_vote(self, results):
        """Determines the majority relevance based on agent responses."""
        relevant_count = results.count("Relevant")
        not_relevant_count = results.count("Not Relevant")

        if relevant_count > not_relevant_count:
            final_decision = "Relevant"
        else:
            final_decision = "Not Relevant"

        print(f"\nMajority decision: The input is {final_decision} to the topic.\n")


# Instantiate the environment and conduct a discussion
environment = MixtureOfAgents()
input_text = """
The high-throughput and liver-on-chip systems exhibit enhanced in vivo-like functions and demonstrate the potential utility of these platforms for DILI risk assessment. Tenofovir-inarigivr-associated hepatotoxicity was observed and correlates with the clinical manifestation of DILI observed in patients."""
topic = "DILI"
environment.conduct_discussion(input_text, topic)
