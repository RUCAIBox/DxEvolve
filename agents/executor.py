import pydantic
from langchain.agents.agent import *
AgentStep.__config__.arbitrary_types_allowed = True
AgentExecutor.__config__.arbitrary_types_allowed = True
AgentAction.__config__.arbitrary_types_allowed = True
IS_V2 = pydantic.__version__.startswith("2.")


_original_init = AgentStep.__init__

def _patched_init(self, *args, **kwargs):
    if "action" in kwargs:
        action = kwargs["action"]
        if not isinstance(action, dict) and hasattr(action, "tool"):
            kwargs["action"] = {
                "tool": getattr(action, "tool", ""),
                "tool_input": getattr(action, "tool_input", ""),
                "log": getattr(action, "log", "")
            }
    _original_init(self, *args, **kwargs)

AgentStep.__init__ = _patched_init


class CustomAgentExecutor(AgentExecutor):
    summarize_observation: bool = True
    
    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, Any],
        color_mapping: Dict[str, str],
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[Any] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:

        result = super()._take_next_step(
            name_to_tool_map=name_to_tool_map,
            color_mapping=color_mapping,
            inputs=inputs,
            intermediate_steps=intermediate_steps,
            run_manager=run_manager,
        )

        if (not self.summarize_observation) or isinstance(result, AgentFinish):
            return result

        if not isinstance(result, list):
            return result
        
        input_text = inputs.get("input", "")
        new_result = []

        for item in result:
            if isinstance(item, AgentStep):
                if ("not available" in str(item.observation).lower()) or ("no results available" in str(item.observation).lower()):
                    summarized = str(item.observation)
                    new_result.append(AgentStep(action=item.action, observation=summarized))
                    continue
                summarized = self.agent.summarize_tool_observation(
                    inputs=inputs,
                    input_text=input_text,
                    intermediate_steps=intermediate_steps,
                    action=item.action,
                    observation=str(item.observation),
                )
                new_result.append(AgentStep(action=item.action, observation=summarized))
            elif isinstance(item, tuple) and len(item) == 2:
                action, obs = item
                summarized = self.agent.summarize_tool_observation(
                    inputs=inputs,
                    input_text=input_text,
                    intermediate_steps=intermediate_steps,
                    action=action,
                    observation=str(obs),
                )
                new_result.append((action, summarized))
            else:
                new_result.append(item)

        return new_result