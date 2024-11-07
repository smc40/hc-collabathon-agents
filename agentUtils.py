import openai
from swarm import Swarm, Agent
from swarm.types import Result
import os, re
import PyPDF2

with open('../env.txt','r') as env_file:
    for line in env_file.readlines():
        key, val = re.split('=', line.strip())
        os.environ[key]=val

        
prompt_main = f'''
You are a helpful agent to determine which agent to use for user.
If the user asked for DILI or liver injury classification of a drug, use Agent DILI.
If the user asked for DICT or cardiotoxicity classification of a drug, use Agent DICT.
Otherwise, use Agent Generic.
'''

prompt_DILI = f'''
Transfer the query to Agent TOI to retrieve the full text from the PDF.
Answer the user question based on the information extracted from.
'''


# Define PDF text extraction function
def extract_pdf_text(file_path, context_variables):
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text()
        return full_text
    except FileNotFoundError:
        return "Error: PDF file not found."
    except Exception as e:
        return f"Error reading PDF file: {e}"

def get_AE_sections(text, context_variables):
    res=[]
    prompt = f"""
    You are an expert in drug labeling review.
    You focus on two labeling sections that related to report Adverse Events.
    Based on the given texts,
    extract all adverse reaction content that related to drug-induced liver injury (DILI) in each section.
    If there is not DILI information in that section, return "No DILI information was found."
    ### Output example:
    #S1 [Warnings and Precautions]:[No DILI information was found.] 
    #S2 [Adverse Reactions]:[...original content from input texts...]
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
        if context_variables['verbose']:
            print('\n==DILI related sections in the document===\n',section_content,'\n===End===\n')
        sections = re.split(r'\n+',section_content)
        return sections
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return "Error extracting sections."
    return res
    
def find_dili_keywords(name, text, context_variables):
    # Define the prompt to instruct the LLM to identify DILI-specific keywords
    prompt = f"""
    You are an expert in drug-induced liver injury (DILI). 
    Please extract all DILI-related keywords from the following text. 
    Provide the keywords as a list, focusing only on terms directly relevant to liver injury.
    if it mentioned "No DILI information was found.", return None
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
        # print(text[:10], response)
        keywords = response.choices[0].message.content.strip()
        if context_variables['verbose']:
            print(f'\n==DILI information found in {name} section===\n {keywords} \n===End===\n')
        return keywords
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return "Error extracting keywords."

# Define Agent transfer functions
def transfer_to_agent_DILI():
    print('use Agent DILI')
    return agent_b

def transfer_to_agent_DICT():
    print('use Agent DICT')
    return agent_c

def transfer_to_agent_Generic():
    print('use Agent Generic')
    return agent_d

def transfer_to_agent_TOI(drug_name, context_variables):
    print('use Agent TOI')
    # Retrieve file path from pdf_dict based on drug name
    file_path = context_variables['pdf_dict'].get(drug_name.upper(), None)
    print(f'PDF Retrieval: This file {file_path} is used since drug name "{drug_name}" was found.')
    if file_path:
        # Extract full text from PDF
        full_text = extract_pdf_text(file_path, context_variables)
        sections = get_AE_sections(full_text, context_variables)
        keyword_pool={}
        for section in sections:
            print('current section, ', section[:100])
            try:
                name, content = re.split(r'(?<=\]):(?=\[)', section)
                # print(name, content)
                dili_keywords = find_dili_keywords(name, content, context_variables)
                keyword_pool[name] = dili_keywords
            except:
                print(drug_name, "this section content is not working normal.", section)
                continue
        if context_variables['verbose']:
            print(f'\n====Final Reference Used =====\n{str(keyword_pool)}\n=====END======\nFinal Response:\n')
        return Result(
            value=str(keyword_pool),
            context_variables={"final_reference": str(sections),
                               "final_keywords": str(keyword_pool)}
            
        )
        # print(full_text)
        # return full_text
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
    model="gpt-4o",
    name="Agent DILI",
    instructions=prompt_DILI,
    functions=[transfer_to_agent_TOI]  # Properly named function
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