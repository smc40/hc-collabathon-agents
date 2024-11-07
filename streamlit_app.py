import streamlit as st
import openai
from swarm import Swarm, Agent
import os, re
import PyPDF2
from demo_openai import MixtureOfAgents

# Load environment variables from .env file
with open('.env', 'r') as env_file:
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
        with st.chat_message("assistant"):
            st.write('==DILI related sections in the document===\n', section_content, '\n===End===\n')
        sections = re.split(r'\n+', section_content)
        return sections
    except Exception as e:
        st.write(f"Error querying OpenAI API: {e}")
        return "Error extracting sections."
    return res
    

def find_dili_keywords(name, text):
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
        with st.chat_message("assistant"):
            st.write(f'==DILI information found in {name} section===\n {keywords} \n===End===\n')
        return keywords
    except Exception as e:
        with st.chat_message("assistant"):
            st.write(f"Error querying OpenAI API: {e}")
        return "Error extracting keywords."

# Define Agent transfer functions
def transfer_to_agent_DILI():
    with st.chat_message("assistant"):
        st.write("Agent DILI at work ⚒️")
    return agent_b

def transfer_to_agent_DICT():
    with st.chat_message("assistant"):
        st.write("Agent DICT at work")
    return agent_c

def transfer_to_agent_Generic():
    with st.chat_message("assistant"):
        st.write("Generic Agent at work")
    return agent_d

def transfer_to_agent_TOI(drug_name):
    with st.chat_message("assistant"):
        st.write("Agent TOI at work ⏳ ")
    
    file_path = pdf_dict.get(drug_name.lower(), None)
    with st.chat_message("assistant"):
        st.write(f"I found {file_path} found for drug name '{drug_name}'.")
    
    if file_path:
        full_text = extract_pdf_text(file_path)
        sections = get_AE_sections(full_text)
        keyword_pool = {}
        for section in sections:
            name, content = re.split(r'(?<=\]):(?=\[)', section)
            dili_keywords = find_dili_keywords(name, content)
            keyword_pool[name] = dili_keywords
        
        with st.chat_message("assistant"):
            st.write(f"\n====Final Reference Used =====\n{str(keyword_pool)}\n=====END======\nFinal Response:\n")
        
        return str(keyword_pool)
    else:
        return "No relevant PDF found for this drug."

# Define main and sub-agents
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
st.title("Multi-Agent DILI Detection Prototype")

with st.expander("See Agent Architecture"):
    st.image("./img/architecture.png")


# Chat input for drug name
user_input = st.chat_input("Enter the name of the drug eg. tamaxophin")

with st.sidebar:
    st.markdown("Available drugs:\n" + "\n".join(f"- {drug}" for drug in pdf_dict.keys()))

# Process user input from either chat input or button selection
if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages = st.session_state.get("messages", [])
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Main query execution
    client = Swarm()
    response = client.run(
        agent=agent_main, 
        messages=[{"role": "user", "content": f"What is the DILI class of {user_input}?"}]
    )

    # Display the response in chat UI
    with st.chat_message("assistant"):
        st.write(response.messages[-1]["content"])

    # Additional processing and final result
    input_text = user_input
    topic = "DILI"
    environment = MixtureOfAgents()
    result = environment.conduct_discussion(input_text, topic)
    
    with st.chat_message("assistant"):
        st.write("Final Response:", result)
