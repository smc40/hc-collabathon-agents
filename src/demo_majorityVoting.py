import os
from collections import Counter
from swarms import MixtureOfAgents, Agent
from swarm_models import OpenAIChat
from swarms.structs.majority_voting import MajorityVoting

# Initialize OpenAI model
model = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini", temperature=0.1
)

# Define agents with regulatory-specific prompts, returning a dict with 'rationale' and 'classification'
agents = [
    Agent(
        agent_name="RegulatoryAffairs",
        system_prompt=(
            "You ensure that products comply with all regulations and standards required by governing bodies throughout "
            "development and post-market stages. Analyze the text and provide a response as a dictionary with 'rationale' "
            "explaining your decision, and 'classification' as either 'dili' or 'non_dili'."
        ),
        llm=model
    ),
    Agent(
        agent_name="ClinicalRegulatory",
        system_prompt=(
            "Collaborate with clinical teams to design studies that meet regulatory requirements for safety and efficacy, "
            "supporting the submission process for new therapies. Analyze the text and provide a response as a dictionary "
            "with 'rationale' explaining your decision, and 'classification' as either 'dili' or 'non_dili'."
        ),
        llm=model
    ),
    Agent(
        agent_name="RegulatoryCompliance",
        system_prompt=(
            "Oversee adherence to regulatory guidelines and quality standards across all stages of the product lifecycle "
            "to mitigate compliance risks. Analyze the text and provide a response as a dictionary with 'rationale' "
            "explaining your decision, and 'classification' as either 'dili' or 'non_dili'."
        ),
        llm=model
    ),
]
import json
# Helper function to parse agent responses
def parse_agent_responses(agent_responses):
    parsed_responses = []
    for response in agent_responses:
        try:
            # Convert JSON string to dictionary
            parsed_response = json.loads(response[0])
            parsed_responses.append(parsed_response)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    return parsed_responses

# Enhanced majority voting function to use 'classification' field for voting
def majority_voting(answers):
    print("answers")
    print(answers)
    answers = parse_agent_responses(answers)
    if not answers:
        return {"classification": "I don't knooooow", "rationale": "No answers provided by agents."}
    
    # Extract classifications from each agent's response if present
    classifications = [answer['classification'] for answer in answers if 'classification' in answer]
    
    if not classifications:
        return {"classification": "I don't kniiiiiw", "rationale": "No valid classifications in agent responses."}
    
    # Count occurrences of each classification
    classification_counts = Counter(classifications)
    most_common_classification, count = classification_counts.most_common(1)[0]
    
    # Determine if we have a majority for one class
    if count > len(answers) / 2:
        rationale = next(
            answer['rationale'] for answer in answers if answer['classification'] == most_common_classification
        )
        return {
            "classification": most_common_classification,
            "rationale": rationale
        }
    else:
        return {"classification": "I don't know", "rationale": "No clear consensus among agents."}

# Create MajorityVoting instance and override the default output parser
majority_voting_instance = MajorityVoting(agents=agents, output_parser=majority_voting)

# Define the text input for classification
text = """The high-throughput and liver-on-chip systems exhibit enhanced in vivo-like functions and demonstrate the 
potential utility of these platforms for DILI risk assessment. Tenofovir-inarigivr-associated hepatotoxicity was 
observed and correlates with the clinical manifestation of DILI observed in patients."""

# Run the majority voting system
formatted_task = f"Is the following text DILI-related: {text}"
result = majority_voting_instance.run(formatted_task)

# Output the result
print("Result:", result)
