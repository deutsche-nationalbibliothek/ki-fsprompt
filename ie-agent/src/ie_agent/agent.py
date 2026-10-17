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
FORCE_FINISH = {"type": "function", "function": {"name": FINISH}}
NOTHING_MAPPED = "Antworte mit einem Werkzeugaufruf. Bilde deine Kandidaten zuerst mit map_to_gnd() ab."


def load_prompt(name):
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8").rstrip("\n")


def system_prompt(kag, examples_with_sections, toc=True):
    """system.md with the parts that depend on the conditions (prompts/system_parts.yaml)."""
    parts = yaml.safe_load((PROMPTS / "system_parts.yaml").read_text(encoding="utf-8"))
    with_kag = "kag" if kag else "no_kag"
    return load_prompt("system").format(
        examples_step=parts["examples_step"]["forced" if examples_with_sections else "optional"],
        toc_step=parts["toc_step"]["toc" if toc else "no_toc"], check_step=parts["check_step"][with_kag],
        finish_number=parts["finish_number"][with_kag],
        id_sources=parts["id_sources"][with_kag])


def rejection(unknown, known):
    """Tool result when finish() contains IDs that did not come from a tool (shown to the model)."""
    return (f"Nicht abgeschlossen. Diese IDs stammen nicht aus map_to_gnd: {', '.join(unknown)}. "
            "Verwende nur GND-IDs aus map_to_gnd, auch wenn du eine GND-Nummer zu kennen glaubst. "
            "Bilde deine Kandidaten zuerst mit map_to_gnd ab und rufe dann finish erneut auf."
            + (f" Gültig waren: {', '.join(known)}." if known else ""))


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
    last_answer: str = ""      # raw text of the last answer without a tool call (for debugging, e.g. a wrong parser)
    asked_again: bool = False  # did the model already answer once without a tool?
    rejected: int = 0
    forced: bool = False       # was finish() forced in the last step?

    def send_tool_result(self, call, content):
        self.messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        if self.tracing:
            self.trace_steps[-1]["results"].append({"tool": call.function.name, "content": content})


class Agent:
    def __init__(self, cfg, docs, mapper, gnd, client=None):
        llm, agent = cfg["llm"], cfg["agent"]
        self.client = client or OpenAI(base_url=llm["base_url"], api_key="unused")
        self.model = llm["model"] or self.client.models.list().data[0].id
        self.reasoning = llm["reasoning"]
        self.max_tokens = llm["max_tokens_reasoning"] if self.reasoning else llm["max_tokens"]
        self.max_steps = agent["max_steps"]
        self.max_rejections = agent["max_rejections"]
        self.docs = docs
        self.tools = ToolRunner(docs, mapper, gnd, cfg)
        self.examples_with_sections = agent["examples_with_sections"]
        self.kag = agent["kag"]
        self.toc = agent["toc"]
        parts = [system_prompt(self.kag, self.examples_with_sections, self.toc)]
        if agent["examples_gnd"]:
            parts.append(load_prompt("examples_gnd"))
        if self.kag:
            parts.append(load_prompt("kag") + "\n" + load_prompt("kag_ie"))
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
        p = Progress(doc_id, new_state(title), [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.first_message.format(
                title=title, examples=self.tools.examples(doc_id, "title"))},
        ], trace)
        self._loop(p)
        if p.forced and p.final is None:
            p.abort = "forced_without_valid_ids"
        return self._result(p, time.time() - start)

    def _loop(self, p):
        for step in range(self.max_steps):
            final_step = step == self.max_steps - 1
            # with nothing mapped a forced finish could only return nothing, so the model gets its steps
            forced = final_step or (p.asked_again and bool(p.state["mapped"]))
            if final_step and not p.asked_again:
                p.messages.append({"role": "user", "content": self.last_step})
            offered = offered_tools(p.state, forced, self.examples_with_sections, self.kag, self.toc)
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=p.messages, tools=offered,
                    tool_choice=FORCE_FINISH if forced else "auto",
                    temperature=0, max_tokens=self.max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": self.reasoning}},
                )
            except BadRequestError as e:
                if "maximum context length" in str(e):   # conversation too long: finish with what was mapped
                    p.abort = "context_full"
                    return
                if forced:   # server refuses to force the tool: fall back instead of failing
                    p.abort = "forcing_refused"
                    p.last_answer = str(e)[:500]
                    return
                raise
            p.tokens_in += response.usage.prompt_tokens
            p.tokens_out += response.usage.completion_tokens
            msg = response.choices[0].message
            calls = msg.tool_calls or []
            self._remember_answer(p, step, msg, calls, offered, forced)

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

            for call in calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if call.function.name != FINISH:
                    self._run_tool(p, call, args)
                elif self._finish(p, call, args, forced):
                    return
        p.abort = "max_steps"

    @staticmethod
    def _remember_answer(p, step, msg, calls, offered, forced):
        assistant = {"role": "assistant", "content": msg.content or ""}
        if calls:
            assistant["tool_calls"] = [c.model_dump() for c in calls]
        p.messages.append(assistant)
        if p.tracing:
            p.trace_steps.append({
                "step": step + 1, "offered": [t["function"]["name"] for t in offered], "forced": forced,
                # vLLM returns the reasoning separately (field name depends on the version)
                "reasoning": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
                "content": msg.content, "calls": [{"tool": c.function.name, "args": c.function.arguments}
                                                  for c in calls],
                "results": []})

    def _finish(self, p, call, args, forced):
        """Handles a finish() call. Returns True when the document is done, False when the IDs were sent back."""
        ids = [str(i).strip() for i in args.get("gnd_ids", [])]
        p.steps.append({"tool": FINISH, "args": args})
        unknown = [i for i in ids if not allowed(p.state, i)]
        known = [i for i in ids if allowed(p.state, i)]
        p.state["unknown"].extend(unknown)
        if unknown and not forced and p.rejected < self.max_rejections:
            # IDs from memory (e.g. GND numbers like 4044038-4) are unchecked: send them back
            p.rejected += 1
            p.send_tool_result(call, rejection(unknown, known))
            return False
        p.forced = forced
        p.final = known or None   # without a valid ID the fallback in _result applies
        if not known and not forced:
            p.abort = "finish_without_valid_ids"
        return True

    def _run_tool(self, p, call, args):
        name = call.function.name
        graph_before = set(p.state["graph"])
        refused = check_call(name, args, p.state, self.toc)
        result = refused or self.tools.run(p.doc_id, name, args, p.state)
        entry = {"tool": name, "args": args, "response_chars": len(result)}
        # for the analysis only, the model does not see this
        entry.update({"refused": True} if refused else
                     self.tools.step_details(p.doc_id, name, args, result, p.state, graph_before))
        p.steps.append(entry)
        p.send_tool_result(call, result)

    def _result(self, p, seconds):
        state = p.state
        final = list(state["mapped"]) if p.final is None else p.final   # fallback: every mapped candidate
        unknown = list(dict.fromkeys(state["unknown"] + [i for i in final if not allowed(state, i)]))
        final = [i for i in dict.fromkeys(final) if allowed(state, i)]

        def suggestion(gnd_id):
            if gnd_id in state["mapped"]:
                return state["mapped"][gnd_id]
            # taken over from gnd_context (broader, narrower or related term): a real GND ID, no mapping needed
            return {"candidate": state["graph"][gnd_id], "gnd_id": gnd_id, "term": state["graph"][gnd_id],
                    "score": 1.0, "cosine_similarity": 1.0, "source": "gnd_context"}

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
            result["trace"] = {"model": self.model, "reasoning": self.reasoning, "system_prompt": self.system_prompt,
                               "first_message": p.messages[1]["content"], "steps": p.trace_steps}
        return result
