"""The agent loop for ONE document: ask the model, run its tools, append the results, until it calls finish().

In the last allowed step (or after an answer without a tool call) finish() is forced, so the agent chooses
itself. Without this rule the code used to return every candidate ever mapped (on average 20, F1 ≈ 0.21).
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from openai import BadRequestError, OpenAI

from .tools import ALL_TOOLS, FINISH, ToolRunner, allowed, check_call, new_state, offered_tools

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"

# tool_choice that forces finish().
FORCE_FINISH = {"type": "function", "function": {"name": FINISH}}

# Asked when the model answers without a tool call before anything was mapped.
NOTHING_MAPPED = "Antworte mit einem Werkzeugaufruf. Bilde deine Kandidaten zuerst mit map_to_gnd() ab."


# ---------------------------------------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------------------------------------

def load_prompt(name):
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8").rstrip("\n")


def system_prompt(gnd_context, examples_with_sections, toc=True):
    """system.md with the parts that depend on the conditions (prompts/system_parts.yaml)."""
    parts = yaml.safe_load((PROMPTS / "system_parts.yaml").read_text(encoding="utf-8"))
    with_gnd_context = "gnd_context" if gnd_context else "no_gnd_context"
    return load_prompt("system").format(
        examples_step=parts["examples_step"]["forced" if examples_with_sections else "optional"],
        toc_step=parts["toc_step"]["toc" if toc else "no_toc"],
        check_step=parts["check_step"][with_gnd_context],
        finish_number=parts["finish_number"][with_gnd_context],
        id_sources=parts["id_sources"][with_gnd_context],
    )


def rejection(unknown, known):
    """Tool result when finish() contains IDs that did not come from a tool (shown to the model)."""
    message = (
        f"Nicht abgeschlossen. Diese IDs stammen nicht aus map_to_gnd: {', '.join(unknown)}. "
        "Verwende nur GND-IDs aus map_to_gnd, auch wenn du eine GND-Nummer zu kennen glaubst. "
        "Bilde deine Kandidaten zuerst mit map_to_gnd ab und rufe dann finish erneut auf."
    )
    if known:
        message += f" Gültig waren: {', '.join(known)}."
    return message


# ---------------------------------------------------------------------------------------------------------
# What happened so far for one document
# ---------------------------------------------------------------------------------------------------------

@dataclass
class Progress:
    """What happened so far in the loop for one document."""
    doc_id: str
    state: dict
    messages: list
    tracing: bool
    steps: list = field(default_factory=list)
    trace_steps: list = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    final: list | None = None
    abort: str | None = None
    last_answer: str = ""  # raw text of the last answer without a tool call (for debugging, e.g. a wrong parser)
    asked_again: bool = False  # did the model already answer once without a tool?
    rejected: int = 0
    forced: bool = False  # was finish() forced in the last step?

    def send_tool_result(self, call, content):
        self.messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        if self.tracing:
            self.trace_steps[-1]["results"].append({"tool": call.function.name, "content": content})


def _arguments(call):
    """The arguments of a tool call; {} if the model wrote no valid JSON."""
    try:
        return json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------------------------------------

class Agent:
    def __init__(self, cfg, docs, mapper, gnd, client=None):
        llm, agent = cfg["llm"], cfg["agent"]

        # The model server (OpenAI API of vLLM).
        self.client = client or OpenAI(base_url=llm["base_url"], api_key="unused")
        self.model = llm["model"] or self.client.models.list().data[0].id
        self.reasoning = llm["reasoning"]
        self.max_tokens = llm["max_tokens_reasoning"] if self.reasoning else llm["max_tokens"]

        # The rules of the loop and the conditions.
        self.max_steps = agent["max_steps"]
        self.max_rejections = agent["max_rejections"]
        self.examples_with_sections = agent["examples_with_sections"]
        self.gnd_context = agent["gnd_context"]
        self.toc = agent["toc"]

        self.docs = docs
        self.tools = ToolRunner(docs, mapper, gnd, cfg)

        # The prompts: system.md, then the parts for variant A and for gnd_context.
        parts = [system_prompt(self.gnd_context, self.examples_with_sections, self.toc)]
        if agent["examples_gnd"]:
            parts.append(load_prompt("examples_gnd"))
        if self.gnd_context:
            parts.append(load_prompt("gnd_context") + "\n" + load_prompt("gnd_context_ie"))
        self.system_prompt = "\n\n".join(parts)
        self.first_message = load_prompt("first_message")
        self.last_step = load_prompt("last_step")

    @property
    def prompt_hash(self):
        """Short hash over prompts and tool definitions: which prompt version produced which result?"""
        raw = json.dumps([self.system_prompt, self.first_message, self.last_step, ALL_TOOLS], ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def annotate(self, doc_id, trace=False):
        """Annotates one document. With trace=True the result also contains every step as the model saw it:
        offered tools, its reasoning (if the server returns it), its tool calls and the tool results."""
        start = time.time()
        title = self.docs.title[doc_id]

        # The start: a fresh state and two messages.
        first_message = self.first_message.format(title=title, examples=self.tools.examples(doc_id, "title"))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": first_message},
        ]
        p = Progress(doc_id, new_state(title), messages, trace)

        self._loop(p)

        if p.forced and p.final is None:
            p.abort = "forced_without_valid_ids"
        return self._result(p, time.time() - start)

    # --- the loop ---

    def _loop(self, p):
        for step in range(self.max_steps):
            # 1. Is finish() forced? In the last step, or after an answer without a tool call.
            # With nothing mapped a forced finish could only return nothing, so the model gets its steps.
            final_step = step == self.max_steps - 1
            forced = final_step or (p.asked_again and bool(p.state["mapped"]))
            if final_step and not p.asked_again:
                p.messages.append({"role": "user", "content": self.last_step})

            # 2. Ask the model.
            offered = offered_tools(p.state, forced, self.examples_with_sections, self.gnd_context, self.toc)
            response = self._request(p, offered, forced)
            if response is None:
                # The server refused the request, p.abort says why.
                return

            p.tokens_in += response.usage.prompt_tokens
            p.tokens_out += response.usage.completion_tokens
            msg = response.choices[0].message
            calls = msg.tool_calls or []
            self._remember_answer(p, step, msg, calls, offered, forced)

            # 3. An answer without a tool call: ask once more, the second time give up.
            if not calls:
                p.last_answer = (msg.content or "")[:1000]
                if p.asked_again:
                    p.abort = "no_final_answer"
                    return
                p.asked_again = True
                p.messages.append({"role": "user", "content": self.last_step if p.state["mapped"] else NOTHING_MAPPED})
                continue
            if not forced:
                p.asked_again = False

            # 4. Run the tool calls in order. An accepted finish() ends the document.
            for call in calls:
                args = _arguments(call)
                if call.function.name != FINISH:
                    self._run_tool(p, call, args)
                elif self._finish(p, call, args, forced):
                    return

        p.abort = "max_steps"

    def _request(self, p, offered, forced):
        """One request to the model, or None if the server refused it (then p.abort says why)."""
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=p.messages,
                tools=offered,
                tool_choice=FORCE_FINISH if forced else "auto",
                temperature=0,
                max_tokens=self.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.reasoning}},
            )
        except BadRequestError as e:
            # The conversation got too long: finish with what was mapped.
            if "maximum context length" in str(e):
                p.abort = "context_full"
                return None
            # The server refuses to force the tool: fall back instead of failing.
            if forced:
                p.abort = "forcing_refused"
                p.last_answer = str(e)[:500]
                return None
            raise

    @staticmethod
    def _remember_answer(p, step, msg, calls, offered, forced):
        """Appends the model's answer to the conversation (and to the trace)."""
        assistant = {"role": "assistant", "content": msg.content or ""}
        if calls:
            assistant["tool_calls"] = [c.model_dump() for c in calls]
        p.messages.append(assistant)

        if p.tracing:
            p.trace_steps.append({
                "step": step + 1,
                "offered": [t["function"]["name"] for t in offered],
                "forced": forced,
                # vLLM returns the reasoning separately (the field name depends on the version).
                "reasoning": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
                "content": msg.content,
                "calls": [{"tool": c.function.name, "args": c.function.arguments} for c in calls],
                "results": [],
            })

    def _run_tool(self, p, call, args):
        """Runs one tool call (or refuses it, see check_call) and sends the result to the model."""
        name = call.function.name
        graph_before = set(p.state["graph"])

        refused = check_call(name, args, p.state, self.toc)
        result = refused or self.tools.run(p.doc_id, name, args, p.state)

        # For the analysis only, the model does not see this.
        entry = {"tool": name, "args": args, "response_chars": len(result)}
        if refused:
            entry["refused"] = True
        else:
            entry.update(self.tools.step_details(p.doc_id, name, args, result, p.state, graph_before))
        p.steps.append(entry)

        p.send_tool_result(call, result)

    def _finish(self, p, call, args, forced):
        """Handles a finish() call. Returns True when the document is done, False when the IDs were sent back."""
        ids = [str(i).strip() for i in args.get("gnd_ids", [])]
        p.steps.append({"tool": FINISH, "args": args})

        unknown = [i for i in ids if not allowed(p.state, i)]
        known = [i for i in ids if allowed(p.state, i)]
        p.state["unknown"].extend(unknown)

        # IDs from memory (e.g. GND numbers like 4044038-4) are unchecked: send them back, at most max_rejections
        # times and never in the forced step.
        if unknown and not forced and p.rejected < self.max_rejections:
            p.rejected += 1
            p.send_tool_result(call, rejection(unknown, known))
            return False

        # Accepted: only the valid IDs count.
        p.forced = forced
        p.final = known or None  # without a valid ID the fallback in _result applies
        if not known and not forced:
            p.abort = "finish_without_valid_ids"
        return True

    # --- the result ---

    def _result(self, p, seconds):
        """The line of log.jsonl for this document."""
        state = p.state

        # Fallback without an accepted finish(): every mapped candidate.
        final = list(state["mapped"]) if p.final is None else p.final
        unknown = list(dict.fromkeys(state["unknown"] + [i for i in final if not allowed(state, i)]))
        final = [i for i in dict.fromkeys(final) if allowed(state, i)]

        def suggestion(gnd_id):
            if gnd_id in state["mapped"]:
                return state["mapped"][gnd_id]
            # Taken over from gnd_context (broader, narrower or related term): a real GND ID, no mapping needed.
            name = state["graph"][gnd_id]
            return {
                "candidate": name,
                "gnd_id": gnd_id,
                "term": name,
                "score": 1.0,
                "cosine_similarity": 1.0,
                "source": "gnd_context",
            }

        result = {
            "doc_id": p.doc_id,
            "gnd_ids": final,
            "suggestions": [suggestion(i) for i in final],
            "from_graph": [i for i in final if i not in state["mapped"]],
            "n_context": state["context_calls"],
            "read": sorted(state["read"]),
            "examples": state["examples"],
            "title_example_ids": self.tools.example_ids(p.doc_id, "title"),
            "steps": p.steps,
            "n_steps": len(p.steps),
            "n_mapped": len(state["mapped"]),
            "unknown_ids": unknown,
            "rejected": p.rejected,
            "forced": p.forced,
            "abort": p.abort,
            "last_answer": p.last_answer if p.abort else "",
            "tokens_in": p.tokens_in,
            "tokens_out": p.tokens_out,
            "seconds": round(seconds, 2),
        }

        if p.tracing:
            result["trace"] = {
                "model": self.model,
                "reasoning": self.reasoning,
                "system_prompt": self.system_prompt,
                "first_message": p.messages[1]["content"],
                "steps": p.trace_steps,
            }
        return result
