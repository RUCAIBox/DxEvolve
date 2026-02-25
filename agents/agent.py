import pickle
import re
from typing import List, Tuple, Union, Dict, Any, Optional
from hashlib import sha256
import pandas as pd
from datetime import datetime

from agents.executor import CustomAgentExecutor as AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.schema.messages import BaseMessage
from langchain.schema import AgentAction
from langchain.callbacks import FileCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from evaluators.pathology_evaluator import best_effort_cover_fast

from agents.prompts import (
    AFTER_TOOL_TEMPLATE,
    SUMMARIZE_OBSERVATION_TEMPLATE,
    SUMMARIZE_PUBMED_TEMPLATE,
    DIAG_CRIT_TOOL_DESCR,
    TOOL_USE_EXAMPLES,
    DIAG_CRIT_TOOL_USE_EXAMPLE,
)
from agents.prompts_guide import (
    SUMMARIZE_CASE_TEMPLATE,
    CHAT_TEMPLATE,
)
from agents.DiagnosisWorkflowParser import DiagnosisWorkflowParser
from tools.Tools import (
    RunLaboratoryTests,
    RunImaging,
    DoPhysicalExamination,
    ReadDiagnosticCriteria,
    SearchPubMedTool,
    SearchPubMed,
)
from tools.Guides import (
    GuidelineStore,
    GuidelineSearchTool
)
from tools.utils import action_input_pretty_printer
from utils.nlp import calculate_num_tokens, truncate_text
from tools.retrieve import ExperienceStore, ExperienceSearchTool
import json

# model_stop_words = ["\nObservation:", "\nObservations:"]
STOP_WORDS = ["Observation:", "Observations:", "observation:", "observations:", "\nObservation:", "\nObservations:"]



def transform_list_to_dict(data_file):
    with open(data_file) as f:
        data_list = json.load(f)
    result = {}
    for item in data_list:
        patient_id = item.get("patient_id")
        raw_content = item.get("raw_output", "")
        clean_content = raw_content.replace("**", "")
        result[int(patient_id)] = clean_content        
    return result


class TextSummaryCache:
    def __init__(self):
        self.cache = {}

    def hash_text(self, text):
        return sha256(text.encode()).hexdigest()

    def add_summary(self, text, summary):
        text_hash = self.hash_text(text)
        if text_hash in self.cache:
            return
        self.cache[text_hash] = summary

    def get_summary(self, text):
        text_hash = self.hash_text(text)
        return self.cache.get(text_hash, None)


class CustomZeroShotAgent(ZeroShotAgent):
    lab_test_mapping_df: pd.DataFrame = None
    observation_summary_cache: TextSummaryCache = TextSummaryCache()
    stop: List[str]
    max_context_length: int
    tags: Dict[str, str]
    summarize: bool
    experience_store: Optional[ExperienceStore] = None
    summarize_each_step: bool = False
    tool_obs_prompt: Optional[PromptTemplate] = None

    use_doctor: Optional[bool] = True

    class Config:
        arbitrary_types_allowed = True

    # Allow for multiple stop criteria instead of just taking the observation prefix string
    @property
    def _stop(self) -> List[str]:
        return self.stop

    # Need to override to pass input so that we can calculate the number of tokes
    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts, kwargs = self._construct_scratchpad(intermediate_steps, **kwargs)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    def _extract_thought_from_log(self, log: str) -> str:
        if not log:
            return ""
        # 兼容你那种 FORMAT A 输出
        m = re.search(r"Thought:\s*(.*?)\nAction:", log, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        # 兼容偶发没有 Thought 标签
        return ""

    def summarize_tool_observation(
        self,
        inputs: Dict[str, str],
        input_text: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        action: AgentAction,
        observation: str,
    ) -> str:
        """Return summarized observation to be stored into intermediate_steps."""
        if "not available" in observation.lower() and len(observation.split(" ")) < 100:
            return observation
        if "This is the first case. No experience found" in observation:
            return observation
        action_name = getattr(action, "tool", "")
        action_input = ""

        if isinstance(getattr(action, "tool_input", None), dict):
            action_input = action.tool_input.get("action_input", "None")
        thought = action.log.split("\nAction: ")[0].strip()

        previous_steps, _ = self._construct_scratchpad(
            intermediate_steps=intermediate_steps,
            **inputs,
        )

        prompt_str = self.tool_obs_prompt.format(
            input_text=input_text or "",
            previous_steps=previous_steps,
            action=action_name,
            action_input=str(action_input),
            thought=thought,
            observation=(observation or "").strip(),
        )

        # 直接 _call
        llm = self.llm_chain.llm
        summary = llm.invoke(prompt_str, stop=[])  # stop 你也可以传 self._stop
        summary = (summary or "").strip()
        # if "deepseek" in llm.model_name.lower():
        #     pass
        # else:
        #     
        summary = summary.split("## Final Answer")[-1].split("Final Answer\n")[-1].split("Final Answer")[-1].strip()
        
        FIELD_NAMES = [
            "Key Abnormals:",
            "Key Normals/Negatives:",
            "Conflicts/Uncertainty:",
            "Impact on Working Hypotheses:",
            "Search Takeaways:",
            "Suggested Follow-ups:",
            "Raw Trace",
        ]

        def _none_of_fields_present(text: str) -> bool:
            # 只要任意一个字段名出现，就返回 False
            return all(fn not in text for fn in FIELD_NAMES)
        
        if _none_of_fields_present(summary):
            return observation

        return summary

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            if thoughts.endswith(":"):
                thoughts += " "
            thoughts += action.log
            if action.tool == "Laboratory Tests":
                obs, _ = best_effort_cover_fast(action.tool_input.get("action_input", []))
                if obs:
                    observation = f"Recognized as: {', '.join(obs)}. " + observation
            obs_str = str(observation) if not isinstance(observation, str) else observation
            thoughts += f"{self.tags['ai_tag_end']}{self.tags['user_tag_start']}{self.observation_prefix}{obs_str.strip()}{self.tags['user_tag_end']}{self.tags['ai_tag_start']}{self.llm_prefix}"
        if (
            calculate_num_tokens(
                self.llm_chain.llm.tokenizer,
                [
                    self.llm_chain.prompt.format(
                        input=kwargs["input"],
                        agent_scratchpad=thoughts,
                    )
                ],
            )
            >= self.max_context_length - 100
        ) and self.summarize:
            thoughts = self._summarize_steps(intermediate_steps)

        # Worst worst case, we are still over or close to the limit even after summarizing and thus should truncate and force a diagnosis
        if (
            calculate_num_tokens(
                self.llm_chain.llm.tokenizer,
                [
                    self.llm_chain.prompt.format(
                        input=kwargs["input"],
                        agent_scratchpad=thoughts,
                    )
                ],
            )
            >= self.max_context_length - 100
        ):
            prompt_and_input_tokens = calculate_num_tokens(
                self.llm_chain.llm.tokenizer,
                [
                    self.llm_chain.prompt.format(
                        input=kwargs["input"], agent_scratchpad=""
                    )
                ],
            )
            # Could be that input is already over limit and we need to truncate input
            if prompt_and_input_tokens > self.max_context_length - 100:
                prompt_tokens = calculate_num_tokens(
                    self.llm_chain.llm.tokenizer,
                    [self.llm_chain.prompt.format(input="", agent_scratchpad="")],
                )
                kwargs["input"] = truncate_text(
                    self.llm_chain.llm.tokenizer,
                    kwargs["input"],
                    self.max_context_length - prompt_tokens - 200,
                )
                thoughts = ""
            else:
                thoughts = truncate_text(
                    self.llm_chain.llm.tokenizer,
                    thoughts,
                    self.max_context_length - prompt_and_input_tokens - 100,
                )  # give yourself 100 tokens for diagnosis and treatment and tags
            thoughts += f'{self.tags["ai_tag_end"]}{self.tags["user_tag_start"]}Provide a Final Diagnosis and Treatment.{self.tags["user_tag_end"]}{self.tags["ai_tag_start"]}Final'

        # Also return kwargs so if we edited input, the change is propagated
        return " " + thoughts.strip(), kwargs

    def construct_experience(
        self,
        info: Dict[str, Any],
        max_obs_len: int = 30000,
    ) -> str:
        """
        Use SUMMARIZE_CASE_TEMPLATE to generate a concise experience summary via LLM.
        Mirrors the style of prompt_pubmed setup (PromptTemplate + LLMChain + predict).
        """

        # 1) Flatten steps to a readable chronological trace
        def _truncate(s: str, n: int) -> str:
            s = (s or "").strip().replace("\n", " ")
            return (s[:n] + " ...") if len(s) > n else s

        intermediate_steps = info.get("intermediate_steps", [])
        input_text = info.get("input", "")
        output_text = info.get("output", "")
        ground_truth = info.get("ground_truth", "")
        is_correct = info.get("correct", False)
        message = info.get("message", "")
        clinician = info.get("clinician", "")
        doctor=info.get("doctor", "")

        step_lines = []
        for i, (action, observation) in enumerate(intermediate_steps, 1):
            obs = ""
            tool = getattr(action, "tool", "Unknown Tool")
            if hasattr(action, "tool_input") and isinstance(action.tool_input, dict):
                if tool == "Laboratory Tests":
                    action_input = action.log.split("Action Input: ")[-1].strip()
                    modified, _ = best_effort_cover_fast(action.tool_input.get("action_input", ""))
                    obs += f"Recognized as {', '.join(modified)}. "

                else:
                    action_input = action.tool_input.get("action_input", "")
            else:
                action_input = str(getattr(action, "tool_input", ""))
            obs += _truncate(observation, max_obs_len)
            step_lines.append(f"[{i}] {tool}: {action_input} → {obs}")
        steps_str = "\n".join(step_lines)

        prompt_experience = PromptTemplate(
            template=SUMMARIZE_CASE_TEMPLATE,
            input_variables=["input", "intermediate_steps", "output", "ground_truth", "correctness", "message", "clinician"],
            partial_variables={
                "system_tag_start": self.tags["system_tag_start"],
                "system_tag_end": self.tags["system_tag_end"],
                "ai_tag_start": self.tags["ai_tag_start"],
            },
        )
        # 3) Create chain and predict
        chain = LLMChain(llm=self.llm_chain.llm, prompt=prompt_experience)
        correctness = "✅ Correct" if is_correct else "❌ Incorrect"
        summary = chain.predict(
            input=input_text,
            intermediate_steps=steps_str,
            output=output_text,
            ground_truth=ground_truth,
            correctness=correctness,
            message=message,
            clinician=clinician,
            stop=[],
        ).strip()

        if getattr(self, "experience_store", None) is not None:
            metadata = {
                "input": input_text,
                "summary": summary,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "output": output_text,
                "feedback": message,
                "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            self.experience_store.add_experience(metadata=metadata)

        return self.experience_store

    def _summarize_steps(self, intermediate_steps):
        summarize_prompt = SUMMARIZE_OBSERVATION_TEMPLATE
        prompt = PromptTemplate(
            template=summarize_prompt,
            input_variables=["observation"],
            partial_variables={
                "system_tag_start": self.tags["system_tag_start"],
                "system_tag_end": self.tags["system_tag_end"],
                "user_tag_start": self.tags["user_tag_start"],
                "user_tag_end": self.tags["user_tag_end"],
                "ai_tag_start": self.tags["ai_tag_start"],
            },
        )
        if "PubMed Search" in self.allowed_tools:
            prompt_pubmed = PromptTemplate(
                template=SUMMARIZE_PUBMED_TEMPLATE,
                input_variables=["query", "observation"],
                partial_variables={
                    "system_tag_start": self.tags["system_tag_start"],
                    "system_tag_end": self.tags["system_tag_end"],
                    "user_tag_start": self.tags["user_tag_start"],
                    "user_tag_end": self.tags["user_tag_end"],
                    "ai_tag_start": self.tags["ai_tag_start"],
                },
            )
        chain = LLMChain(llm=self.llm_chain.llm, prompt=prompt)
        if "PubMed Search" in self.allowed_tools:
            chain_pubmed = LLMChain(llm=self.llm_chain.llm, prompt=prompt_pubmed)
        summaries = []
        summaries.append("A summary of information I know thus far:")
        for indx, (action, observation) in enumerate(intermediate_steps):
            # Only summarize valid actions
            if action.tool in self.allowed_tools:
                # Keep format as in instruction to re-enforce schema
                summaries.append("Action: " + action.tool)
                if action.tool in [
                    "Laboratory Tests",
                    "Imaging",
                    "Diagnostic Criteria",
                    "PubMed Search",  # 添加PubMed搜索工具
                    "Experience Search",
                    "Guideline Search",
                ]:
                    summaries.append(
                        "Action Input: "
                        + action_input_pretty_printer(
                            action.tool_input["action_input"], self.lab_test_mapping_df
                        )
                    )
                # Check cache to not re-summarize same observation
                summary = self.observation_summary_cache.get_summary(observation)
                if not summary:
                    # Summary of each step should be minimal and should not exceed max_context_length
                    prompt_tokens = calculate_num_tokens(
                        self.llm_chain.llm.tokenizer,
                        [
                            prompt.format(observation=""),
                        ],
                    )

                    observation = truncate_text(
                        self.llm_chain.llm.tokenizer,
                        observation,
                        self.max_context_length
                        - prompt_tokens
                        - 100,  # Gives a max of 100 tokens to generate for the summary if we are near context length limit. Usually only used when model does really weird infinite generations of action inputs and doesnt hit a stop token so shouldnt be much actual info to summarize anyway
                    )
                    if action.tool == "PubMed Search":
                        summary = chain_pubmed.predict(observation=observation, query=action.tool_input["action_input"], stop=[])
                    else:
                        summary = chain.predict(observation=observation, stop=[])
                        
                    # Add to cache
                    self.observation_summary_cache.add_summary(observation, summary)
                summaries.append("Observation: " + summary)
            else:
                # Include invalid requests in summary to not run into infinite loop of same invalid tool being ordered
                invalid_request = action.log
                # Condense invalid request to the action and everything afterwards. Can remove thinking
                if "action:" in action.log.lower():
                    invalid_request = action.log[action.log.lower().index("action:") :]
                summaries.append(
                    f"I tried '{invalid_request}', but it was an invalid request."
                )
                # If invalid tool was final request, remind of valid tools and diagnosis option. Add string to last summary because we dont want to force newlines that the prompt templates maybe do not want
                if indx == len(intermediate_steps) - 1:
                    summaries[-1] = summaries[-1] + (
                        f'{self.tags["ai_tag_end"]}{self.tags["user_tag_start"]}Please choose a valid tool from {self.allowed_tools} or provide a Final Diagnosis and Treatment.{self.tags["user_tag_end"]}{self.tags["ai_tag_start"]}{self.llm_prefix}'
                    )
                    return "\n".join(summaries)
        summaries.append(self.llm_prefix)
        return "\n".join(summaries)


def create_prompt(
    tags, tool_names, add_tool_descr, tool_use_examples, **kwargs
) -> PromptTemplate:
    template = CHAT_TEMPLATE
    # template = CHAT_TEMPLATE_guideOnly_loop         
    template = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad"],
            # "tool_names": action_input_pretty_printer(tool_names, None),
        partial_variables={
            "add_tool_descr": add_tool_descr,
            "examples": tool_use_examples,
            "system_tag_start": tags["system_tag_start"],
            "user_tag_start": tags["user_tag_start"],
            "ai_tag_start": tags["ai_tag_start"],
            "system_tag_end": tags["system_tag_end"],
            "user_tag_end": tags["user_tag_end"],
        },
    )
    return template


def build_agent_executor_ZeroShot(
    patient,
    llm,
    lab_test_mapping_path,
    logfile,
    max_context_length,
    tags,
    include_ref_range,
    bin_lab_results,
    include_tool_use_examples,
    provide_diagnostic_criteria,
    summarize,
    model_stop_words,
    pubmed_api_key,
    use_pubmed,
    use_guidelines: bool = True,
    use_experience_search: bool = True,
    exp_store: Optional[ExperienceStore] = None,
    experience_index_path: str = "./experience_faiss",
    embedding_model_name: str = "BAAI/bge-large-en-v1.5",
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    init_store: Optional[List[str]] = None,
    summarize_observation: bool = False,
    encourage_guideline: bool = False,
    use_full_info: bool = False,
):
    with open(lab_test_mapping_path, "rb") as f:
        lab_test_mapping_df = pickle.load(f)

    # Define which tools the agent can use to answer user queries
    tools = [
        DoPhysicalExamination(action_results=patient),
        RunLaboratoryTests(
            action_results=patient,
            lab_test_mapping_df=lab_test_mapping_df,
            include_ref_range=include_ref_range,
            bin_lab_results=bin_lab_results,
        ),
        RunImaging(action_results=patient),
    ]

    if use_pubmed:
        pubmed_searcher = SearchPubMed(api_key=pubmed_api_key)
        tools.append(SearchPubMedTool(pubmed_searcher=pubmed_searcher))

    if use_experience_search:
        if exp_store is None:
            exp_store = ExperienceStore(embeddings=embeddings, persist_path=experience_index_path)
            if init_store:
                pass
                # exp_store.init_exp(init_store)
        exp_tool = ExperienceSearchTool(experience_store=exp_store)
        tools.append(exp_tool)
    if use_guidelines:
        guidelines = GuidelineStore(embeddings=embeddings)
        guidelines = GuidelineSearchTool(guideline_store=guidelines, llm=llm, tags=tags)
        tools.append(guidelines)

    # Go through options and see if we want to add any extra tools.
    add_tool_use_examples = ""
    add_tool_descr = ""
    if provide_diagnostic_criteria:
        tools.append(ReadDiagnosticCriteria())
        add_tool_descr += DIAG_CRIT_TOOL_DESCR
        add_tool_use_examples += DIAG_CRIT_TOOL_USE_EXAMPLE

    tool_names = [tool.name for tool in tools]

    # Create prompt
    tool_use_examples = ""
    if include_tool_use_examples:
        tool_use_examples = TOOL_USE_EXAMPLES.format(
            add_tool_use_examples=add_tool_use_examples
        )
    prompt = create_prompt(tags, tool_names, add_tool_descr, tool_use_examples)

    # Create output parser
    output_parser = DiagnosisWorkflowParser(lab_test_mapping_df=lab_test_mapping_df)

    # Initialize logging callback if file provided
    handler = None
    if logfile:
        handler = [FileCallbackHandler(logfile)]

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=handler)

    tool_obs_prompt = PromptTemplate(
        template=AFTER_TOOL_TEMPLATE,
        input_variables=["input_text", "action", "action_input", "thought", "observation", "previous_steps"],
        partial_variables={
            "system_tag_start": tags["system_tag_start"],
            "system_tag_end": tags["system_tag_end"],
            "user_tag_start": tags["user_tag_start"],
            "user_tag_end": tags["user_tag_end"],
            "ai_tag_start": tags["ai_tag_start"],
        },
    )

    # Create agent
    agent = CustomZeroShotAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=list(set(STOP_WORDS + model_stop_words)),
        allowed_tools=tool_names,
        verbose=True,
        return_intermediate_steps=True,
        max_context_length=max_context_length,
        tags=tags,
        tool_obs_prompt=tool_obs_prompt,
        lab_test_mapping_df=lab_test_mapping_df,
        summarize=summarize,
        experience_store=exp_store,
        use_doctor=use_full_info,
    )

    # Init agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=20,
        return_intermediate_steps=True,
        callbacks=handler,
    )
    agent_executor.summarize_observation = summarize_observation

    return agent_executor
