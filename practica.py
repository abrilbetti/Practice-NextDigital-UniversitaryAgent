import streamlit as st  # Needed to create the web interface
import os  # Needed to interact with the files system
from dotenv import load_dotenv  # Used to upload variables from the .env file

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Language model and conversor from text to numbers
from langchain_chroma import Chroma  # Vectorial database to store the content of the PDFs
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Used to fragment the PDF file
from langchain_community.document_loaders import PyPDFLoader  # Required to upload the PDF file
from langchain_community.tools import DuckDuckGoSearchRun  # Needed to search on the Internet
from langchain_core.tools import tool  # Used to convert functions in tools the agent will use
from langchain_core.messages import HumanMessage, AIMessage  # Messages format
from langgraph.prebuilt import create_react_agent  # Used to create a thinking agent

# Initial configuration
load_dotenv()  # I upload the OPENAI_API_KEY from the .env file
st.set_page_config(page_title="Universitary Agent")  # Title of the navigator

# I define the location of the PDF file, such as we did in "sesion 2- 2_conexion_documentos"
PATH_PDF = os.path.join(os.path.dirname(__file__), "resources", "AIPaper.pdf")

# Tools definition
@st.cache_resource  # This line is included so that the following lines of code are executed only once and its results are stored in memory
def setup_retriever():
    """The PDF file was loaded, fragmented in smaller pieces and prepared for semantic research."""
    loader = PyPDFLoader(PATH_PDF)  # We initialise the PDF uploader
    docs = loader.load()  # We read the content of the PDF file
    # The text is fragmented in blocks of 500 characters out of which 50 are overlapped (this is done so we do not loose any context)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    # We are defining how to convert text to vectors using OpenAI, such as we did in "sesion 2 - 1_busqueda semantica"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # We store the fragments in Chroma, a vectorial database
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store.as_retriever(k=3)  # We are returning the capacity of returning the 3 most similar blocks

retriever = setup_retriever() # Retriever object

@tool
def searchInDocument(consulta: str) -> str:
    """Tool to find information in the PDF file. It will be called by the agent if needed."""
    docs = retriever.invoke(consulta)  # Performs the semantic research in Chroma
    return "\n\n".join(f"Fragment: {d.page_content}" for d in docs)  # Unites the results in a text

@tool
def searchInternet(consulta: str) -> str:
    """Tool that enables us to search in google, used if the PDF does not contian the answer."""
    search = DuckDuckGoSearchRun()
    return search.run(consulta)

# Guardrail logic
def isQuestionRelevant(question: str) -> bool:
    """Makes use of LLM to define whether or not the question has something to do with technology or AI."""
    llm_guard = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Quick model with no creativity (since temperature = 0)
   # The model was changed so the program would work
    prompt = (
        "You are a STRICT relevance classifier. The ONLY allowed topics are Artificial Intelligence and technology. "
        "Your ONLY job is to verify if the NEW question is about Computer Science, AI, or Tech. "
        "If the question is about animals, cooking, sports, or anything else NOT directly related to AI/Tech, it is IRRELEVANT. "        
        "The user might ask in Spanish or English." # I am specifying this so the agent is bilingual
        f"Question made by the user: '{question}'. "
        "Do not give explanations."
        "Answer ONLY with 'relevant' or 'irrelevant'."
    )
    # We call the model with the question made by the user
    answer = llm_guard.invoke([HumanMessage(content=prompt)]).content.lower()
    return "relevant" in answer and "irrelevant" not in answer # if answer is relevant, a True is returned. If the answer is irrelevant, a False is returned.

# Streamlit interface
st.title("Univeritary Agent Chatbot")
st.markdown("Search your documents or look on the Internet with guardrails.")

# Research history:
if "messages" not in st.session_state:
    st.session_state.messages = [] # This line enables variables to remain even if the page was refreshed

# The following lines output the messages that were saved in the history
for msg in st.session_state.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

# The following lines define the input field in the inferior part of the screen
if prompt := st.chat_input("What would you like to know?"):
    # We store the message introduced by the user
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Now, I define the logic behind the answer of the agent
    with st.chat_message("assistant"):
        relevant = False # I set the relevant variable to False
        # First, I define the guardrail
        with st.status("Checking the relevance of the question...", expanded=False) as status:
            relevant = isQuestionRelevant(prompt)

            if not relevant:
                status.update(label="The question exceeds the domain", state="error")
            else:
                status.update(label="The question was accepted", state="complete")

        if not relevant:
            error_msg = "I am an agent specialized in AI and technology, I can not help you with that."
            st.markdown(error_msg)
            st.session_state.messages.append(AIMessage(content=error_msg))
        else:
            # Agente ReAct (model that decides what tool to use)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = create_react_agent(
                llm, 
                tools=[searchInDocument, searchInternet],
                prompt=(
                    "You are a universitary expert system. Your priority is to use 'searchInDocument'. "
                    "If the information is not enough or too new, use 'searchInternet'. "
                )                
            )

            # Now, I define the execution of the agent
            with st.spinner("The agent is reasoning..."):
                inputs = {"messages": st.session_state.messages}
                result = agent.invoke(inputs)  # We execute the agent with the history
                    
                finalAnswer = result["messages"][-1].content  # We use the most answer that was generated last
                    
                # The internal reasoning of the agent is shown in a drop-down menu
                with st.expander("See the reasoning of the agent"):
                    for msg in result["messages"][:-1]:
                        st.write(msg)

                st.markdown(finalAnswer)
                st.session_state.messages.append(AIMessage(content=finalAnswer))

# The following lines configure the sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("Clean search history"):
        st.session_state.messages = []  # We delete the search history
        st.rerun()  # This line refreshes the page
    st.info("This agent uses RAG (Chroma) and Web Research (DuckDuckGo).")