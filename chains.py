import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
  def __init__(self):
    self.llm = ChatGroq(
        temperature=0,
        groq_api_key = os.getenv("GROQ_API_KEY"),
        model_name = 'deepseek-r1-distill-qwen-32b'
    )

  def extract_jobs(self, cleaned_text):
    prompt_extract = PromptTemplate.from_template(
        """

    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}
    ### INSTRUCTION:
    The scrapped text is  from the career's page of a website.
    Your job is to extract the job postings and return them in JSON format containing
    following keys: 'role','experience','skills'  and 'description'.
    Only return the valid JSON.
    ### VALID JSON (NO PREAMBLE):
    """
    )

    chain_extract = prompt_extract | self.llm
    res = chain_extract.invoke({"page_data": cleaned_text})
    try:
      json_parser = JsonOutputParser()
      res = json_parser.parse(res.content)
    except OutputParserException:
      raise OutputParserException("Context.too big. Unable to parse jobs.")
    return res if isinstance(res, list) else [res]

  def write_mail(self, job, links):
    prompt_email = PromptTemplate.from_template(
       """
    ### JOB DESCRIPTION:
    {job_description}

    ### INSTRUCTION:
    You are Gaurav, a Master's student in the Data Science domain at Central University of Haryana. I am looking for a job.
    Your job is to write a cold email to the hiring manager regarding the job mentioned above
    in fulfilling their needs.
    Also, add the most relevant ones from the following links: {link_list} to showcase my projects according to their skills need.
    Remember you are Gaurav, a Data Enthusiast.
    Do not provide a preamble.

    ### EMAIL (NO PREAMBLE):

    """
    )
    chain_email = prompt_email | self.llm
    res = chain_email.invoke({"job_description": str(job), "link_list":links})
    return res.content


if __name__ == "__main__":
  print(os.getenv("GROQ_API_KEY"))