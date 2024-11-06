from swarms import MixtureOfAgents, Agent
from swarm_models import OpenAIChat
import os

MAX_LOOPS = 2

# Initialize OpenAI model
model = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini", temperature=0.1
)

# Define the director agent
director = Agent(
    agent_name="Director",
    system_prompt="Oversees and directs the tasks for regulatory roles.",
    llm=model,
    max_loops=MAX_LOOPS,
    verbose=True,
    dashboard=True,
    streaming_on=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="director.json",
)

# Initialize Regulatory Affairs Specialist agent
regulatory_affairs_specialist = Agent(
    agent_name="RegulatoryAffairsSpecialist",
    system_prompt="Handles regulatory submissions, ensures compliance, and manages documentation.",
    llm=model,
    max_loops=MAX_LOOPS,
    verbose=True,
    dashboard=True,
    streaming_on=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="regulatory_affairs_specialist.json",
)

# Initialize Clinical Evaluator/Scientist agent
clinical_evaluator = Agent(
    agent_name="ClinicalEvaluator",
    system_prompt="Assesses safety, efficacy, and quality of products by reviewing clinical trial data.",
    llm=model,
    max_loops=MAX_LOOPS,
    verbose=True,
    dashboard=True,
    streaming_on=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="clinical_evaluator.json",
)

# Initialize Quality Assurance Officer agent
qa_officer = Agent(
    agent_name="QAOfficer",
    system_prompt="Oversees quality standards, conducts audits, and ensures compliance with regulatory standards.",
    llm=model,
    max_loops=MAX_LOOPS,
    verbose=True,
    dashboard=True,
    streaming_on=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="qa_officer.json",
)

# Initialize the MixtureOfAgents
moe_swarm = MixtureOfAgents(
    reference_agents=[director, regulatory_affairs_specialist, clinical_evaluator, qa_officer],
    aggregator_agent=director,
    aggregator_system_prompt="Based on the inputs, decides, if this is DILI related or not."
)

# Define the task and input for the swarm
task_name = "DILI classification"
task_description = (
    "Classify drug-induced liver injury (DILI) risk based on high-throughput and liver-on-chip systems, "
    "which exhibit enhanced in vivo-like functions and demonstrate potential utility for DILI risk assessment."
)
input_data = (
    "Tenofovir-inarigivr-associated hepatotoxicity was observed and correlates with the clinical manifestation "
    "of DILI observed in patients."
)

# Create the formatted template with detailed context for the task
formatted_task = (
    f"Task: {task_name}\n"
    f"Description: {task_description}\n"
    f"Input data: {input_data}\n\n"
    "Perform the task in a regulatory science context, and provide insights based on the input provided."
)

# Run the swarm with the formatted task
history = moe_swarm.run(task=formatted_task)
print(history)
