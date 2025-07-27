from datetime import datetime
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from onyx.agents.agent_search.dr.constants import CLARIFICATION_REQUEST_PREFIX
from onyx.agents.agent_search.dr.constants import MAX_CHAT_HISTORY_MESSAGES
from onyx.agents.agent_search.dr.dr_prompt_builder import get_dr_prompt_template
from onyx.agents.agent_search.dr.models import ClarificationGenerationResponse
from onyx.agents.agent_search.dr.models import DRPromptPurpose
from onyx.agents.agent_search.dr.models import DRTimeBudget
from onyx.agents.agent_search.dr.models import OrchestrationClarificationInfo
from onyx.agents.agent_search.dr.states import DRPath
from onyx.agents.agent_search.dr.states import MainState
from onyx.agents.agent_search.dr.states import OrchestrationUpdate
from onyx.agents.agent_search.dr.utils import get_chat_history_string
from onyx.agents.agent_search.models import GraphConfig
from onyx.agents.agent_search.shared_graph_utils.llm import invoke_llm_json
from onyx.agents.agent_search.shared_graph_utils.utils import (
    get_langgraph_node_log_string,
)
from onyx.agents.agent_search.shared_graph_utils.utils import write_custom_event
from onyx.chat.models import AgentAnswerPiece
from onyx.configs.constants import MessageType
from onyx.kg.utils.extraction_utils import get_entity_types_str
from onyx.kg.utils.extraction_utils import get_relationship_types_str
from onyx.tools.tool_implementations.custom.custom_tool import CUSTOM_TOOL_RESPONSE_ID
from onyx.tools.tool_implementations.custom.custom_tool import CustomTool
from onyx.tools.tool_implementations.internet_search.internet_search_tool import (
    INTERNET_SEARCH_RESPONSE_SUMMARY_ID,
)
from onyx.tools.tool_implementations.internet_search.internet_search_tool import (
    InternetSearchTool,
)
from onyx.tools.tool_implementations.search.search_tool import (
    SEARCH_RESPONSE_SUMMARY_ID,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.utils.logger import setup_logger

logger = setup_logger()


def _get_available_tools(graph_config: GraphConfig, kg_enabled: bool) -> list[dict]:

    available_tools = []
    for tool in graph_config.tooling.tools:

        if tool.name == "run_kg_search" and not kg_enabled:
            continue

        # TODO: use a pydantic model instead of dict?
        tool_dict = {}
        tool_dict["name"] = tool.name
        tool_dict["description"] = tool.description
        tool_dict["display_name"] = tool.display_name

        # TODO: add proper KG search tool
        if tool.name == "run_kg_search":
            KG_SEARCH_RESPONSE_SUMMARY_ID = CUSTOM_TOOL_RESPONSE_ID  # unused for now
            tool_dict["summary_signature"] = KG_SEARCH_RESPONSE_SUMMARY_ID
            tool_dict["path"] = DRPath.KNOWLEDGE_GRAPH.value
        elif isinstance(tool, CustomTool):
            tool_dict["summary_signature"] = CUSTOM_TOOL_RESPONSE_ID
            tool_dict["path"] = (
                tool.name.upper()
            )  # TODO: average tool cost will have keyerror
        elif isinstance(tool, InternetSearchTool):
            tool_dict["summary_signature"] = INTERNET_SEARCH_RESPONSE_SUMMARY_ID
            tool_dict["path"] = DRPath.INTERNET_SEARCH.value
        elif isinstance(tool, SearchTool):
            tool_dict["summary_signature"] = SEARCH_RESPONSE_SUMMARY_ID
            tool_dict["path"] = DRPath.SEARCH.value

        available_tools.append(tool_dict)

    return available_tools


def _get_existing_clarification_request(
    graph_config: GraphConfig,
) -> tuple[OrchestrationClarificationInfo, str, str] | None:
    """
    Returns the clarification info, original question, and updated chat history if
    a clarification request and response exists, otherwise returns None.
    """
    # check for clarification request and response in message history
    previous_raw_messages = graph_config.inputs.prompt_builder.raw_message_history
    if (
        len(previous_raw_messages) < 2
        or previous_raw_messages[-1].message_type != MessageType.ASSISTANT
        or CLARIFICATION_REQUEST_PREFIX not in previous_raw_messages[-1].message
    ):
        return None

    # get the clarification request and response
    previous_messages = graph_config.inputs.prompt_builder.message_history
    last_message = previous_raw_messages[-1].message

    clarification = OrchestrationClarificationInfo(
        clarification_question=last_message.split(CLARIFICATION_REQUEST_PREFIX, 1)[
            1
        ].strip(),
        clarification_response=graph_config.inputs.prompt_builder.raw_user_query,
    )
    original_question = graph_config.inputs.prompt_builder.raw_user_query
    chat_history_string = "(No chat history yet available)"

    # get the original user query and chat history string before the original query
    # e.g., if history = [user query, assistant clarification request, user clarification response],
    # previous_messages = [user query, assistant clarification request], we want the user query
    for i, message in enumerate(reversed(previous_messages), 1):
        if (
            isinstance(message, HumanMessage)
            and message.content
            and isinstance(message.content, str)
        ):
            original_question = message.content
            chat_history_string = (
                get_chat_history_string(
                    graph_config.inputs.prompt_builder.message_history[:-i],
                    MAX_CHAT_HISTORY_MESSAGES,
                )
                or "(No chat history yet available)"
            )
            break

    return clarification, original_question, chat_history_string


def clarifier(
    state: MainState, config: RunnableConfig, writer: StreamWriter = lambda _: None
) -> OrchestrationUpdate:
    """
    Perform a quick search on the question as is and see whether a set of clarification
    questions is needed. For now this is based on the models
    """

    node_start_time = datetime.now()

    graph_config = cast(GraphConfig, config["metadata"]["config"])
    original_question = graph_config.inputs.prompt_builder.raw_user_query
    time_budget = graph_config.behavior.time_budget

    # get the connected tools and format for the Deep Research flow
    kg_enabled = graph_config.behavior.kg_config_settings.KG_ENABLED
    available_tools: list[dict[str, str]] = _get_available_tools(
        graph_config, kg_enabled
    )

    all_entity_types = get_entity_types_str(active=True)
    all_relationship_types = get_relationship_types_str(active=True)

    query_path = DRPath.ORCHESTRATOR

    # get clarification (unless time budget is FAST)
    clarification = None
    if time_budget != DRTimeBudget.FAST:
        result = _get_existing_clarification_request(graph_config)
        if result is not None:
            clarification, original_question, chat_history_string = result
        else:
            # generate clarification questions if needed
            chat_history_string = (
                get_chat_history_string(
                    graph_config.inputs.prompt_builder.message_history,
                    MAX_CHAT_HISTORY_MESSAGES,
                )
                or "(No chat history yet available)"
            )

            base_clarification_prompt = get_dr_prompt_template(
                DRPromptPurpose.CLARIFICATION,
                time_budget,
                entity_types_string=all_entity_types,
                relationship_types_string=all_relationship_types,
                available_tools=available_tools,
            )
            clarification_prompt = base_clarification_prompt.replace(
                "---question---", original_question
            ).replace("---chat_history_string---", chat_history_string)

            try:
                clarification_response = invoke_llm_json(
                    llm=graph_config.tooling.primary_llm,
                    prompt=clarification_prompt,
                    schema=ClarificationGenerationResponse,
                    timeout_override=25,
                    max_tokens=1500,
                )
            except Exception as e:
                logger.error(f"Error in clarification generation: {e}")
                raise e

            if (
                clarification_response.clarification_needed
                and clarification_response.clarification_question
            ):
                clarification = OrchestrationClarificationInfo(
                    clarification_question=clarification_response.clarification_question,
                    clarification_response=None,
                )
                query_path = DRPath.END
                write_custom_event(
                    "basic_response",
                    AgentAnswerPiece(
                        answer_piece=(
                            f"{CLARIFICATION_REQUEST_PREFIX} "
                            f"{clarification.clarification_question}\n\n"
                        ),
                        level=0,
                        level_question_num=0,
                        answer_type="agent_level_answer",
                    ),
                    writer,
                )
    else:
        chat_history_string = (
            get_chat_history_string(
                graph_config.inputs.prompt_builder.message_history,
                MAX_CHAT_HISTORY_MESSAGES,
            )
            or "(No chat history yet available)"
        )

    return OrchestrationUpdate(
        original_question=original_question,
        chat_history_string=chat_history_string,
        query_path=[query_path],
        query_list=[],
        iteration_nr=0,
        log_messages=[
            get_langgraph_node_log_string(
                graph_component="main",
                node_name="clarifier",
                node_start_time=node_start_time,
            )
        ],
        clarification=clarification,
        available_tools=available_tools,
    )
