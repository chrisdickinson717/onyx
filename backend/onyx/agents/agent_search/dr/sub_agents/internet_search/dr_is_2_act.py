from datetime import datetime
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from onyx.agents.agent_search.dr.models import DRPath
from onyx.agents.agent_search.dr.models import IterationAnswer
from onyx.agents.agent_search.dr.models import SearchAnswer
from onyx.agents.agent_search.dr.sub_agents.internet_search.dr_is_states import (
    BranchInput,
)
from onyx.agents.agent_search.dr.sub_agents.internet_search.dr_is_states import (
    BranchUpdate,
)
from onyx.agents.agent_search.dr.utils import extract_document_citations
from onyx.agents.agent_search.kb_search.graph_utils import build_document_context
from onyx.agents.agent_search.models import GraphConfig
from onyx.agents.agent_search.shared_graph_utils.llm import invoke_llm_json
from onyx.agents.agent_search.shared_graph_utils.utils import (
    get_langgraph_node_log_string,
)
from onyx.agents.agent_search.shared_graph_utils.utils import write_custom_event
from onyx.chat.models import AgentAnswerPiece
from onyx.chat.models import AnswerStyleConfig
from onyx.chat.models import CitationConfig
from onyx.chat.models import DocumentPruningConfig
from onyx.chat.models import LlmDoc
from onyx.chat.models import PromptConfig
from onyx.context.search.models import InferenceSection
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.prompts.dr_prompts import BASIC_SEARCH_PROMPT
from onyx.tools.tool_implementations.internet_search.internet_search_tool import (
    INTERNET_SEARCH_RESPONSE_SUMMARY_ID,
)
from onyx.tools.tool_implementations.internet_search.internet_search_tool import (
    InternetSearchTool,
)
from onyx.tools.tool_implementations.internet_search.models import ProviderType
from onyx.tools.tool_implementations.search.search_tool import SearchResponseSummary
from onyx.utils.logger import setup_logger

logger = setup_logger()


def internet_search(
    state: BranchInput, config: RunnableConfig, writer: StreamWriter = lambda _: None
) -> BranchUpdate:
    """
    LangGraph node to perform a standard search as part of the DR process.
    """

    node_start_time = datetime.now()
    iteration_nr = state.iteration_nr
    parallelization_nr = state.parallelization_nr

    search_query = state.branch_question
    if not search_query:
        raise ValueError("search_query is not set")

    write_custom_event(
        "basic_response",
        AgentAnswerPiece(
            answer_piece=(
                f"SUB-QUESTION {iteration_nr}.{parallelization_nr} "
                f"(INTERNET SEARCH): {search_query}\n\n"
            ),
            level=0,
            level_question_num=0,
            answer_type="agent_level_answer",
        ),
        writer,
    )

    graph_config = cast(GraphConfig, config["metadata"]["config"])
    base_question = graph_config.inputs.prompt_builder.raw_user_query

    logger.debug(
        f"Search start for Internet Search {iteration_nr}.{parallelization_nr} at {datetime.now()}"
    )

    if graph_config.inputs.persona is None:
        raise ValueError("persona is not set")

    retrieved_docs: list[InferenceSection] = []

    # new db session to avoid concurrency issues
    with get_session_with_current_tenant() as search_db_session:

        internet_search_tool = InternetSearchTool(
            db_session=search_db_session,
            persona=graph_config.inputs.persona,
            prompt_config=PromptConfig(
                system_prompt="You are a helpful assistant that can search the internet for information.",
                task_prompt=f"Search the internet for information about the following question: {search_query}",
                datetime_aware=True,
                include_citations=True,
            ),
            llm=graph_config.tooling.primary_llm,
            document_pruning_config=DocumentPruningConfig(),
            answer_style_config=AnswerStyleConfig(
                citation_config=CitationConfig(), structured_response_format=None
            ),
            provider=ProviderType.EXA.value,
            num_results=10,
        )

        for tool_response in internet_search_tool.run(
            internet_search_query=search_query
        ):
            # get retrieved docs to send to the rest of the graph
            if tool_response.id == INTERNET_SEARCH_RESPONSE_SUMMARY_ID:
                response = cast(SearchResponseSummary, tool_response.response)
                retrieved_docs = response.top_sections
                break

    # stream_write_step_answer_explicit(writer, step_nr=1, answer=full_answer)

    document_texts_list = []

    for doc_num, retrieved_doc in enumerate(retrieved_docs[:15]):
        if not isinstance(retrieved_doc, (InferenceSection, LlmDoc)):
            raise ValueError(f"Unexpected document type: {type(retrieved_doc)}")
        chunk_text = build_document_context(retrieved_doc, doc_num + 1)
        document_texts_list.append(chunk_text)

    document_texts = "\n\n".join(document_texts_list)

    logger.debug(
        f"Search end/LLM start for Internet Search {iteration_nr}.{parallelization_nr} at {datetime.now()}"
    )

    # Built prompt

    search_prompt = (
        BASIC_SEARCH_PROMPT.replace("---search_query---", search_query)
        .replace("---base_question---", base_question)
        .replace("---document_text---", document_texts)
    )

    # Run LLM

    search_answer_json = invoke_llm_json(
        llm=graph_config.tooling.primary_llm,
        prompt=search_prompt,
        schema=SearchAnswer,
        timeout_override=40,
        max_tokens=1500,
    )

    logger.debug(
        f"LLM/all done for Internet Search {iteration_nr}.{parallelization_nr} at {datetime.now()}"
    )

    write_custom_event(
        "basic_response",
        AgentAnswerPiece(
            answer_piece=f"ANSWERED {iteration_nr}.{parallelization_nr}\n\n",
            level=0,
            level_question_num=0,
            answer_type="agent_level_answer",
        ),
        writer,
    )

    # get all citations and remove them from the answer to avoid
    # incorrect citations when the documents get reordered by the closer
    citation_string = search_answer_json.citations
    answer_string = search_answer_json.answer

    (
        citation_numbers,
        answer_string,
    ) = extract_document_citations(citation_string + answer_string)
    cited_documents = [
        retrieved_docs[citation_number - 1] for citation_number in citation_numbers
    ]

    return BranchUpdate(
        branch_iteration_responses=[
            IterationAnswer(
                tool=DRPath.INTERNET_SEARCH,
                iteration_nr=iteration_nr,
                parallelization_nr=parallelization_nr,
                question=search_query,
                answer=answer_string,
                cited_documents=cited_documents,
            )
        ],
        log_messages=[
            get_langgraph_node_log_string(
                graph_component="main",
                node_name="search",
                node_start_time=node_start_time,
            )
        ],
    )
