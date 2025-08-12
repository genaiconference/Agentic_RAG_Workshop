import time
import re
import asyncio
from typing import List, Union
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    PromptTemplate,
)
from langchain.agents import AgentOutputParser


# 1️⃣  ───── Robust multi-action parser ────────────────────────────
class MultiActionOutputParser(AgentOutputParser):
    """
    Extract *all* Action / Action Input pairs, even when inputs are
    multi-line or there is text between blocks.
    """
    _ACTION_RE = re.compile(
        r"""Action\s*\d*:\s*(.*?)\s*\n        # tool name
            Action\ Input\s*\d*:\s*([\s\S]*?) # tool input (non-greedy)
            (?=\nAction|\nObservation|\nFinal|\Z)  # stop at next block
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        # ---------- collect all tool calls ----------
        matches = self._ACTION_RE.findall(text)
        if matches:
            return [
                AgentAction(
                    tool=tool.strip(),
                    tool_input=tool_input.strip().strip('"'),
                    log=text,
                )
                for tool, tool_input in matches
            ]

        # ---------- or a final answer ----------
        #final = re.search(r"(?:Final Answer|Final Thought):\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        final = re.search(r"(?:Final Answer):\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        
        #final = re.search(r"(?i)Final Answer:\s*(?:\*{2})?\s*\n+(.*)", text, flags=re.IGNORECASE | re.DOTALL)   ###PushToQA 04-07-2025 fixed ** issue
        
        if final:
            return AgentFinish(
                return_values={"output": final.group(1).strip()}, log=text
            )

        # ---------- fallback if no Action or Final Answer but appears like final output ----------
        if text.strip():  # fallback: any non-empty text is assumed to be the final answer
            return AgentFinish(
                return_values={"output": text.strip()}, log=text
            )

        raise ValueError(f"Could not parse agent output: {text!r}")



# 2️⃣  ───── Executor that really runs in parallel ────────────────
class ParallelAgentExecutor(AgentExecutor):
    async def _aiter_next_step(self, name_to_tool_map, inputs):
        llm_output = await self.agent.aplan(
            intermediate_steps=self.intermediate_steps, **inputs
        )
        parsed = self.output_parser.parse(llm_output)

        # ----- run *all* Actions concurrently -----
        if isinstance(parsed, list):
            coros = [self._aperform_agent_action(name_to_tool_map, a) for a in parsed]
            results = await asyncio.gather(*coros)

            for a, r in zip(parsed, results):
                self.intermediate_steps.append((a, r))
                yield f"Observation: {r}\n"

        elif isinstance(parsed, AgentFinish):
            self.intermediate_steps.append(
                (parsed, parsed.return_values["output"])
            )
            yield f"Final Answer: {parsed.return_values['output']}\n"


# 3️⃣  ───── Helper to build the agent + prompt ───────────────────
def get_react_agent(llm, tools, system_prompt, verbose=False):
    """Helper function for creating agent executor"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="conversation_history", optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
        AIMessagePromptTemplate(prompt=PromptTemplate(input_variables=['agent_scratchpad'], template='{agent_scratchpad}')),
    ])
    agent = create_react_agent(llm, tools, prompt, output_parser = MultiActionOutputParser())
    
    executor = ParallelAgentExecutor(agent=agent, 
                             tools=tools, 
                             verbose=verbose, 
                             stream_runnable=True,
                             handle_parsing_errors=True,
                             max_iterations = 20, 
                             return_intermediate_steps=True,
                            )
    return executor

def get_simple_react_agent(llm, tools, system_prompt, verbose=False):
    """Helper function for creating agent executor"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="conversation_history", optional=True),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(input_variables=["input"], template="{input}")
            ),
            AIMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["agent_scratchpad"], template="{agent_scratchpad}"
                )
            ),
        ]
    )
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        stream_runnable=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )
