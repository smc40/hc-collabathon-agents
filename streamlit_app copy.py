import streamlit as st
import openai
from swarm import Swarm, Agent
import os, re
import PyPDF2
from demo_openai import MixtureOfAgents

# Load environment variables from .env file
with open('.envalex', 'r') as env_file:
    for line in env_file.readlines():
        key, val = re.split('=', line.strip())
        os.environ[key] = val

# Define PDF dictionary
pdf_dict = {
    'tamoxifen': './data/hc/00064472.pdf',
    'atorvastatin': './data/hc/00066863.pdf'
}

# Define prompts for various agents
prompt_main = '''You are a helpful agent to determine which agent to use for the user.
If the user asked for DILI or liver injury classification of a drug, use Agent DILI.
If the user asked for DICT or cardiotoxicity classification of a drug, use Agent DICT.
Otherwise, use Agent Generic.'''

prompt_DILI = '''Transfer the query to Agent TOI to retrieve the full text from the PDF.
Answer the user question based on the information extracted from.'''

prompt_TOI = f'''Identify the most relevant PDF based on the drug name mentioned by the user, 
extract the full text, and return it. {str(pdf_dict)}'''

# Define PDF text extraction function
def extract_pdf_text(file_path):
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            full_text = "".join(page.extract_text() for page in pdf_reader.pages)
        return full_text
    except FileNotFoundError:
        return "Error: PDF file not found."
    except Exception as e:
        return f"Error reading PDF file: {e}"

def get_AE_sections(text):
    res=[]
    prompt = f"""
    You are an expert in drug labeling review. 
    Extract all adverse reaction content that related to drug-induced liver injury (DILI).
    the return response should also include the section title.
    If it is not available in the original text, generate one by summarzing its content.
    ### Output example:
    #S1 [Warnings and Precautions]:[...content...] 
    #S2 [Adverse Reactions]:[...content...]
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": text}
                     ],
            max_tokens=2000,
            temperature=0
        )
        section_content = response.choices[0].message.content.strip()
        st.write('==DILI related sections in the document===\n', section_content, '\n===End===\n')
        sections = re.split(r'\n+', section_content)
        return sections
    except Exception as e:
        st.write(f"Error querying OpenAI API: {e}")
        return "Error extracting sections."
    return res
    
def find_dili_keywords(name, text):
    # Define the prompt to instruct the LLM to identify DILI-specific keywords
    prompt = f"""
    You are an expert in drug-induced liver injury (DILI). 
    Please extract all DILI-related keywords from the following text. 
    Provide the keywords as a list, focusing only on terms directly relevant to liver injury.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": text}
                     ],
            max_tokens=150,
            temperature=0
        )
        keywords = response.choices[0].message.content.strip()
        st.write(f'==DILI information found in {name} section===\n {keywords} \n===End===\n')
        return keywords
    except Exception as e:
        st.write(f"Error querying OpenAI API: {e}")
        return "Error extracting keywords."

# Define Agent transfer functions
def transfer_to_agent_DILI():
    st.write('use Agent DILI')
    return agent_b

def transfer_to_agent_DICT():
    st.write('use Agent DICT')
    return agent_c

def transfer_to_agent_Generic():
    st.write('use Agent Generic')
    return agent_d

def transfer_to_agent_TOI(drug_name):
    st.write('use Agent TOI')
    # Retrieve file path from pdf_dict based on drug name
    file_path = pdf_dict.get(drug_name.lower(), None)
    st.write(f'PDF Retrieval: This file {file_path} is used since drug name "{drug_name}" was found.')
    if file_path:
        # Extract full text from PDF
        full_text = extract_pdf_text(file_path)
        sections = get_AE_sections(full_text)
        keyword_pool = {}
        for section in sections:
            name, content = re.split(r'(?<=\]):(?=\[)', section)
            dili_keywords = find_dili_keywords(name, content)
            keyword_pool[name] = dili_keywords
        st.write(f'\n====Final Reference Used =====\n{str(keyword_pool)}\n=====END======\nFinal Response:\n')
        return str(keyword_pool)
    else:
        return "No relevant PDF found for this drug."

# Define the main agent and sub-agents
agent_main = Agent(
    model="gpt-4o-mini",
    name="Agent Main",
    instructions=prompt_main,
    functions=[transfer_to_agent_DILI, transfer_to_agent_DICT, transfer_to_agent_Generic],
)

agent_b = Agent(
    model="gpt-4o-mini",
    name="Agent DILI",
    instructions=prompt_DILI,
    functions=[transfer_to_agent_TOI]
)

agent_c = Agent(
    model="gpt-4o-mini",
    name="Agent DICT",
    instructions="Answer whether the drug mentioned will cause cardiotoxicity.",
)

agent_d = Agent(
    model="gpt-4o-mini",
    name="Agent Generic",
    instructions="Answer the user's question if it does not fall into any other specific categories.",
)

# Streamlit App
st.title("Drug Information Chatbot")

# User input for drug name
drug_name = st.text_input("Enter the name of the drug:")

# Chatbot response based on input
if st.button("Run Query"):
    if drug_name:
        client = Swarm()

        response = client.run(
            agent=agent_main,  # Main agent
            messages=[{"role": "user", "content": "What is the DILI class of Atorvastatin?"}],
        )

        print(response.messages[-1]["content"])
        input_text = drug_name  # The user's input acts as the query for the main agent
        topic = "DILI"
        environment = MixtureOfAgents()
        # Start the discussion and capture response
        result = environment.conduct_discussion(input_text, topic)
        st.write("Response:", result)
    else:
        st.warning("Please enter a drug name.")
