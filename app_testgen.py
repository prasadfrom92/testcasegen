import streamlit as st
import pandas as pd
from io import BytesIO
import os
import json
import time

# --- LangChain & Google Generative AI Imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

# --- Pydantic and LangChain Output Parser Imports ---
from pydantic import BaseModel, Field, ValidationError
from langchain.output_parsers import PydanticOutputParser
from typing import List

# --- Environment Setup ---
load_dotenv()
# Configure the Google API key, ensuring it's set in the environment

genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])

# --- Pydantic Model Definitions (Unchanged) ---
class UserStory(BaseModel):
    """A Pydantic model to structure a high-quality user story."""
    title: str = Field(description="A standard-format user story title, e.g., 'As a [user type], I want [to perform an action], so that [I can achieve a goal]'.")
    description: str = Field(description="A concise, clear paragraph describing the feature and its purpose.")
    acceptance_criteria: List[str] = Field(description="A comprehensive list of acceptance criteria in 'Given-When-Then' format.")

class TestCase(BaseModel):
    """A Pydantic model to structure a single test case step."""
    TestCaseID: str = Field(description="A unique ID for the overall test scenario (e.g., E2E-001, TC-001). All steps for a single scenario share the same ID.")
    TestCaseDescription: str = Field(description="A high-level description of the test scenario's objective.")
    TestStepNumber: int = Field(description="The sequential number for this specific test step within the scenario.")
    TestStepDescription: str = Field(description="The detailed action to be performed for this step (e.g., Click, Enter, Navigate).")
    ExpectedResultforEachstep: str = Field(description="The expected system response or state after this specific step is executed.")

class GeneratedTestPlan(BaseModel):
    """A Pydantic model for list of test cases."""
    test_cases: List[TestCase] = Field(description="A comprehensive list of test cases, including all steps.")


# --- INPUT FILE HANDLING & VALIDATION (Unchanged) ---
def validate_and_load_excel(uploaded_file):
    """Validates the uploaded Excel file for required sheets and columns."""
    try:
        xls = pd.ExcelFile(uploaded_file)
        required_sheets = ["Bugs", "TestCases_with_steps"]
        if not all(sheet in xls.sheet_names for sheet in required_sheets):
            st.error(f"❌ Invalid File: The Excel file must contain sheets: {required_sheets}")
            return False

        required_us_cols = ["Jira ID", "Issue Type", "summary of the Issue"]
        us_df = pd.read_excel(xls, sheet_name="Bugs")
        if not all(col in us_df.columns for col in required_us_cols):
            st.error(f"❌ Invalid Sheet: 'Bugs' must have columns: {required_us_cols}")
            return False

        required_tc_cols = ["TestCaseID", "TestCaseDescription", "TestSteps", "StepwiseExpectedResult"]
        tc_df = pd.read_excel(xls, sheet_name="TestCases_with_steps")
        if not all(col in tc_df.columns for col in required_tc_cols):
            st.error(f"❌ Invalid Sheet: 'TestCases_with_steps' must have columns: {required_tc_cols}")
            return False

        st.session_state.user_stories_df = us_df
        st.session_state.test_cases_df = tc_df
        st.session_state.file_processed = True
        return True

    except Exception as e:
        st.error(f"An unexpected error occurred during file validation: {e}")
        return False

# --- MODIFIED: CONTEXTUAL RETRIEVAL & VECTOR STORES ---
def create_vector_stores():
    """Creates two vector stores: one for test case descriptions and one for bug/story summaries."""
    if 'test_cases_df' not in st.session_state or 'user_stories_df' not in st.session_state:
        st.error("Cannot create vector stores, as data is not loaded.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    with st.spinner("🧠 Building context-aware vector stores..."):
        # 1. Vector Store for Test Cases
        tc_df = st.session_state.test_cases_df
        if not tc_df.empty and "TestCaseDescription" in tc_df.columns:
            # We only embed each unique description once to be efficient
            unique_tc_df = tc_df.drop_duplicates(subset=['TestCaseDescription']).copy()
            descriptions = unique_tc_df["TestCaseDescription"].dropna().tolist()
            if descriptions:
                # The metadata's 'index' points to the index in the *original* tc_df
                metadatas = [{"index": i} for i in unique_tc_df.index]
                docs = text_splitter.create_documents(descriptions, metadatas=metadatas)
                st.session_state.tc_vector_store = FAISS.from_documents(docs, embedding=embeddings)
                st.success("Test Case context store created!")
            else:
                st.warning("No test case descriptions found to build a vector store.")
        else:
            st.warning("Test Case data is empty or missing 'TestCaseDescription' column.")


        # 2. Vector Store for User Stories and Bugs (for bug retrieval)
        us_df = st.session_state.user_stories_df
        if not us_df.empty and "summary of the Issue" in us_df.columns:
            summaries = us_df["summary of the Issue"].dropna().tolist()
            if summaries:
                metadatas = [{"index": i} for i, _ in enumerate(summaries)]
                docs = text_splitter.create_documents(summaries, metadatas=metadatas)
                st.session_state.us_bug_vector_store = FAISS.from_documents(docs, embedding=embeddings)
                st.success("Bugs context store created!")
            else:
                st.warning("No summaries found to build the bug/story vector store.")
        else:
            st.warning("User Stories/Bugs data is empty or missing 'summary of the Issue' column.")

# --- LLM Chain Creation (User Story part is unchanged) ---
def get_user_story_enhancer_chain():
    """Returns a chain to enhance a raw user story. (Unchanged)"""
    parser = PydanticOutputParser(pydantic_object=UserStory)
    prompt_template = """
    You are an expert Agile Product Owner. Your task is to transform raw, unstructured user input into a formal, high-quality user story.
    **Chain-of-Thought Process:**
    1.  **Analyze the Input:** Read the raw text and identify the core request.
    2.  **Identify Key Roles:** Determine the user role (WHO?), the action they want (WHAT?), and the benefit (WHY?).
    3.  **Formulate the Title:** Synthesize these into a standard user story title: "As a [role], I want [action], so that [benefit]."
    4.  **Write the Description:** Create a clear, concise paragraph explaining the feature.
    5.  **Draft Acceptance Criteria (AC):** Generate a comprehensive list of ACs using the "Given-When-Then" format. Cover positive, negative, and edge cases.
    ---
    **Your Task:**
    Now, process the following raw input. Analyze it and provide the final, structured JSON output. Do not add any commentary.
    **Raw Input:**
    {raw_input}
    **Format Instructions (JSON):**
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["raw_input"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.1)
    chain = LLMChain(llm=model, prompt=prompt)
    return chain, parser

# --- MODIFIED RAG CHAIN ---
def get_enhanced_rag_chain(temperature, top_p):
    """Returns the main RAG chain for generating test cases, with a modified prompt."""
    parser = PydanticOutputParser(pydantic_object=GeneratedTestPlan)
    prompt_template = """
    You are a Senior QA Automation Engineer tasked with creating a comprehensive test cases. Your analysis must be strictly based on the **ENHANCED USER STORY** and the supplementary **CONTEXT FROM DOCUMENTS**.

    ### 📝 ENHANCED USER STORY:
    {refined_user_story}

    ---

    ### 📚 CONTEXT FROM DOCUMENTS (Historical Test Cases and Bugs):

    #### Relevant Historical Test Cases (from Test Case IDs: {test_case_ids}):
    {formatted_test_cases}

    #### Relevant Historical Bugs:
    {bug_summaries}

    ---

    **Your Goal:** Generate two types of test cases:
    1.  **End-to-End (E2E) Test Cases:** These simulate a complete user journey from start to finish. They consist of multiple steps that string together a full workflow described in the user story (e.g., login -> search for an item -> add to cart -> verify cart).
    2.  **Standalone (Functional) Test Cases:** These are focused, granular tests that verify a single piece of functionality, often mapping directly to a single **Acceptance Criterion**.

    **Chain-of-Thought Process to Follow:**
    1.  **Analyze the Core Goal:** First, read the user story's `Title` and `Description` to understand the overall feature and its purpose. This will inform the E2E scenarios.
    2.  **Identify E2E Journeys:** Based on the core goal, map out the primary user workflows. A typical story has 1-2 E2E paths (e.g., a "happy path" and a key negative path).
    3.  **Identify Standalone Functions:** Go through each `Acceptance Criterion` one-by-one. Each criterion (e.g., "Given-When-Then") is a perfect candidate for a standalone test case, including both positive and negative validation.
    4.  **Extract Specifics from Context:** For ALL test steps, meticulously scan the **CONTEXT FROM DOCUMENTS** to find the precise names of UI elements (buttons, links, fields, IDs), URLs, and expected text messages. **If the context provides a name, you MUST use it.**
    5.  **Construct the E2E Tests:** For each E2E journey identified in step 2, write a sequence of test steps. All steps belonging to the same journey must share the same `TestCaseID` (e.g., "E2E-01"). The `TestCaseDescription` should describe the overall journey.
    6.  **Construct the Standalone Tests:** For each function identified in step 3, write the necessary test steps. Each standalone test should have its own unique `TestCaseID` (e.g., "TC-001", "TC-002").
    7.  **Format the Output:** Assemble the generated tests into the two lists (`e2e_test_cases` and `standalone_test_cases`) as required by the format instructions.

    **CRITICAL RULES:**
    -   **No Hallucination:** Do not invent UI elements, features, or user roles not mentioned in the story or context.
    -   **Context is Truth:** If the User Story and Context seem to conflict, the **CONTEXT FROM DOCUMENTS** is the ultimate source of truth for implementation details like element names.
    -   **Be Meticulous:** Every step must have a clear action and a precise expected result.

    **FORMAT INSTRUCTIONS (JSON):**
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["refined_user_story", "test_case_ids", "formatted_test_cases", "bug_summaries"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=temperature, top_p=top_p)
    chain = LLMChain(llm=model, prompt=prompt)
    return chain, parser


# --- Helper functions (Unchanged) ---
def auto_correct_and_parse_json(json_string: str, parser: PydanticOutputParser):
    """Attempts to parse JSON, with auto-correction for common LLM errors."""
    cleaned_text = json_string.strip().replace("```json", "").replace("```", "").strip()
    try:
        return parser.parse(cleaned_text)
    except Exception as e:
        st.warning(f"Initial JSON parsing failed: {e}. Attempting auto-correction...")
        try:
            data = json.loads(cleaned_text)
            if isinstance(data, dict) and "test_cases" in data:
                if isinstance(data["test_cases"], dict) and "test_cases" in data["test_cases"]:
                    st.warning("🤖 Auto-correcting a nested JSON structure...")
                    data = data["test_cases"]
                    cleaned_text = json.dumps(data)
            return parser.parse(cleaned_text)
        except json.JSONDecodeError:
            if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
                st.warning("🤖 Auto-correcting by wrapping a JSON list in the required root object...")
                cleaned_text = f'{{"test_cases": {cleaned_text}}}'
                return parser.parse(cleaned_text)
        raise e

def display_and_download_response(response_text: str, parser: PydanticOutputParser):
    """Parses the LLM response, displays an editable table, and provides a download button."""
    try:
        parsed_response = auto_correct_and_parse_json(response_text, parser)
        if not parsed_response.test_cases:
            st.warning("The model returned an empty list of test cases.")
            return
        test_cases_data = [tc.dict() for tc in parsed_response.test_cases]
        df = pd.DataFrame(test_cases_data)
        st.subheader("✅ Generated Test Cases (Editable)")
        edited_df = st.data_editor(
            df,
            key="test_case_editor",
            num_rows="dynamic",
            use_container_width=True
        )
        st.session_state.edited_test_cases_df = edited_df
        excel_file = BytesIO()
        edited_df.to_excel(excel_file, index=False, sheet_name="GeneratedTestCases")
        excel_file.seek(0)
        st.download_button(
            label="📥 Download Edited Test Cases as Excel",
            data=excel_file,
            file_name="generated_test_cases.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except (ValidationError, Exception) as e:
        st.error(f"❌ Parsing Error: The response from the LLM could not be processed. Error: {e}")
        st.info("Below is the raw, uncorrected output from the model:")
        st.code(response_text, language="text")

# --- MODIFIED SESSION STATE ---
def init_session_state():
    """Initializes session state with default values."""
    defaults = {
        "file_processed": False,
        "enhanced_story": None,
        "retrieved_test_cases_df": pd.DataFrame(),
        "retrieved_bugs_df": pd.DataFrame(),
        "edited_test_cases_df": pd.DataFrame(),
        "num_test_cases_to_find": 10,
        "num_bugs_to_find": 5,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- MODIFIED HELPER FUNCTION FOR CONTEXT RETRIEVAL ---
def find_and_update_context():
    """Searches for relevant test cases and bugs and updates session state."""
    with st.spinner("Searching for relevant historical context..."):
        if 'tc_vector_store' not in st.session_state or 'us_bug_vector_store' not in st.session_state or not st.session_state.get('enhanced_story'):
            st.warning("Could not retrieve context. Ensure a story is refined and vector stores exist.")
            return

        search_query = st.session_state.enhanced_story.description

        # 1. Find similar Test Cases
        k_tcs = st.session_state.get("num_test_cases_to_find", 3)
        retrieved_tc_docs = st.session_state.tc_vector_store.similarity_search(search_query, k=k_tcs)
        retrieved_tc_indices = [doc.metadata['index'] for doc in retrieved_tc_docs]
        
        # Get the unique descriptions that were found
        matching_descriptions = st.session_state.test_cases_df.loc[retrieved_tc_indices]['TestCaseDescription'].unique()
        # Retrieve all steps for test cases matching these descriptions
        final_tcs_df = st.session_state.test_cases_df[st.session_state.test_cases_df['TestCaseDescription'].isin(matching_descriptions)]
        st.session_state.retrieved_test_cases_df = final_tcs_df
        st.success(f"Found {len(final_tcs_df['TestCaseID'].unique())} relevant test cases.")

        # 2. Find similar Bugs
        k_bugs = st.session_state.get("num_bugs_to_find", 5)
        # Fetch a larger pool to filter down to just bugs
        retrieved_bug_docs = st.session_state.us_bug_vector_store.similarity_search(search_query, k=k_bugs + 10)
        retrieved_bug_indices = [doc.metadata['index'] for doc in retrieved_bug_docs]
        candidate_df = st.session_state.user_stories_df.iloc[retrieved_bug_indices]
        bug_df = candidate_df[candidate_df['Issue Type'] == 'Bug'].head(k_bugs)
        st.session_state.retrieved_bugs_df = bug_df
        st.success(f"Found {len(bug_df)} relevant bugs.")


# --- Main Streamlit App (MODIFIED LOGIC) ---
def main():
    st.set_page_config(page_title="AI Test Case Generator", layout="wide", initial_sidebar_state="expanded")
    init_session_state()
    st.header("🤖 AI-Powered Test Case Generator")
    st.markdown("""
    This tool uses Retrieval-Augmented Generation (RAG) to create test cases for **new** user stories.
    1.  **Upload** your historical data (existing bugs and test cases).
    2.  **Describe** a new feature or requirement. The AI will formalize it into a user story.
    3.  **Curate** the historical test cases and bugs found by the AI to use as context.
    4.  **Generate** a comprehensive test cases for your new story.
    """)

    # --- Sidebar for File Upload & Parameters ---
    with st.sidebar:
        st.title("⚙️ Controls")
        temperature = st.slider(
            label="Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            help="Controls randomness. Lower is more deterministic, higher is more creative."
        )
        top_p = st.slider(
            label="Top-p", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            help="Limits sampling to a subset of tokens with cumulative probability top_p."
        )
        st.title("📂 Step 1: Historical Data")
        uploaded_file = st.file_uploader(
            "Upload your project data", type=["xlsx"], accept_multiple_files=False
        )
        if uploaded_file:
            if st.button("Process Excel File"):
                # Clear old state but preserve loaded dataframes and vector stores
                keys_to_clear = [k for k in st.session_state.keys() if k not in ['file_processed', 'user_stories_df', 'test_cases_df', 'tc_vector_store', 'us_bug_vector_store']]
                for key in keys_to_clear:
                    del st.session_state[key]
                init_session_state() 

                with st.spinner("Validating and loading Excel file..."):
                    if validate_and_load_excel(uploaded_file):
                        create_vector_stores()

    if not st.session_state.get('file_processed', False):
        st.info("👋 Welcome! Please upload your project's Excel file via the sidebar to begin.")
        st.stop()

    st.divider()

    # --- Step 2: Raw User Input for a NEW story ---
    st.subheader("Step 2: Provide Input for a New User Story")
    raw_user_input = st.text_area(
        "Enter your unstructured idea, meeting notes, or requirement here:",
        height=120,
        placeholder="e.g., 'Customers need a way to log in using their Google account.'"
    )

    if st.button("✨ Refine Story & Find Historical data", disabled=not raw_user_input):
        with st.spinner("AI is refining the new user story..."):
            st.session_state.enhanced_story = None
            try:
                enhancer_chain, parser = get_user_story_enhancer_chain()
                response_text = enhancer_chain.invoke({"raw_input": raw_user_input})['text']
                cleaned_text = response_text.strip().replace("```json", "").replace("```", "")
                st.session_state.enhanced_story = parser.parse(cleaned_text)
                st.success("Refinement complete!")
            except Exception as e:
                st.error(f"An error occurred during story refinement: {e}")
                if 'response_text' in locals(): st.code(response_text)
                st.stop()
        
        find_and_update_context()

    # --- Step 3: Review and Curate Context ---
    if st.session_state.get("enhanced_story"):
        st.divider()
        st.subheader("Step 3: Review New Story and Curate Historical Context")
        
        # Display and edit the refined user story
        story = st.session_state.enhanced_story
        st.session_state.enhanced_story.title = st.text_input("**Title**", value=story.title, key="story_title_editor")
        st.session_state.enhanced_story.description = st.text_area("**Description**", value=story.description, height=120, key="story_description_editor")
        with st.expander("View/Edit Acceptance Criteria"):
            ac_text = st.text_area("Acceptance Criteria (one per line)", value="\n".join(story.acceptance_criteria), height=150, key="ac_editor")
            st.session_state.enhanced_story.acceptance_criteria = [line.strip() for line in ac_text.split('\n') if line.strip()]

        st.markdown("---")
        st.markdown("#### Curate Historical Context")
        
        # --- Context Curation Controls ---
        with st.expander("🧠 Fine-tune AI context retrieval", expanded=False):
            with st.form("context_curation_form"):
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input( "📚 Number of Test Cases", min_value=0, max_value=20, value=10, step=1, key="num_test_cases_to_find")
                with col2:
                    st.number_input("🐞 Number of Bugs", min_value=0, max_value=20, value=5, step=1, key="num_bugs_to_find")
                if st.form_submit_button("🔄 Refresh List of TCs & Bugs"):
                    find_and_update_context()

        # --- Curate Test Cases ---
        st.markdown("##### Relevant Historical Test Cases")
        test_cases_df = st.session_state.retrieved_test_cases_df
        if not test_cases_df.empty:
            st.markdown("Remove irrelevant historical test cases before generating the new test cases:")
            # Group by TestCaseID to show one entry per test case
            for test_id, group in test_cases_df.groupby('TestCaseID'):
                row_col1, row_col2 = st.columns([4, 1])
                row_col1.markdown(f"**{test_id}:** {group['TestCaseDescription'].iloc[0]}")
                if row_col2.button("Delete", key=f"delete_tc_{test_id}", help=f"Remove {test_id} from context"):
                    st.session_state.retrieved_test_cases_df = test_cases_df[test_cases_df['TestCaseID'] != test_id]
                    st.rerun()
        else:
            st.write("No relevant historical test cases found.")

        # --- Curate Bugs ---
        st.markdown("##### Relevant Historical Bugs")
        bugs_df = st.session_state.retrieved_bugs_df
        if not bugs_df.empty:
            st.markdown("Remove irrelevant bugs before generating the new test cases:")
            indices_to_keep = list(bugs_df.index)
            for index, row in bugs_df.iterrows():
                row_col1, row_col2 = st.columns([4, 1])
                row_col1.markdown(f"**{row['Jira ID']}:** {row['summary of the Issue']}")
                if row_col2.button("Delete", key=f"delete_bug_{index}", help=f"Remove {row['Jira ID']} from context"):
                    indices_to_keep.remove(index)
                    st.session_state.retrieved_bugs_df = st.session_state.retrieved_bugs_df.loc[indices_to_keep]
                    st.rerun()
        else:
            st.write("No relevant historical bugs found.")

        st.divider()
        
        # --- Step 4: Generate Test Cases ---
        st.subheader("Step 4: Generate Test Cases")
        if st.button("🚀 Generate Test Cases", type="primary"):
            
            curated_tcs_df = st.session_state.retrieved_test_cases_df
            curated_bugs_df = st.session_state.retrieved_bugs_df
            
            refined_story_str = f"Title: {story.title}\nDescription: {story.description}\nAcceptance Criteria:\n- " + "\n- ".join(story.acceptance_criteria)

            # Format historical test cases for the prompt
            formatted_tcs_list = []
            if not curated_tcs_df.empty:
                for test_id, group in curated_tcs_df.groupby('TestCaseID'):
                    desc = group['TestCaseDescription'].iloc[0]
                    tc_str = f"TestCaseID: {test_id}\nTestCaseDescription: {desc}\nSteps:\n"
                    steps_df = group[['TestSteps', 'StepwiseExpectedResult']].reset_index(drop=True)
                    steps_str = steps_df.to_markdown(index=True)
                    tc_str += steps_str
                    formatted_tcs_list.append(tc_str)
            
            formatted_test_cases = "\n---\n".join(formatted_tcs_list) if formatted_tcs_list else "No historical test cases provided in the context."
            test_case_ids_str = ", ".join(curated_tcs_df['TestCaseID'].unique()) if not curated_tcs_df.empty else "N/A"

            # Format bug summaries for the prompt
            bug_summaries = "\n".join([f"- {row['Jira ID']}: {row['summary of the Issue']}" for _, row in curated_bugs_df.iterrows()]) if not curated_bugs_df.empty else "No relevant historical bugs selected."

            with st.spinner("🤖 AI is generating the test test cases... This may take a moment."):
                try:
                    chain, parser = get_enhanced_rag_chain(temperature, top_p)
                    
                    with st.expander("View Full Prompt Sent to LLM"):
                        full_prompt = chain.prompt.format(
                            refined_user_story=refined_story_str,
                            test_case_ids=test_case_ids_str,
                            formatted_test_cases=formatted_test_cases,
                            bug_summaries=bug_summaries,
                            format_instructions=parser.get_format_instructions()
                        )
                        st.code(full_prompt, language='markdown')
                    
                    response = chain.invoke({
                        "refined_user_story": refined_story_str,
                        "test_case_ids": test_case_ids_str,
                        "formatted_test_cases": formatted_test_cases,
                        "bug_summaries": bug_summaries
                    })
                    
                    display_and_download_response(response["text"], parser)

                except Exception as e:
                    st.error(f"An error occurred during test case generation: {e}")

if __name__ == "__main__":
    main()
