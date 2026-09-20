"""Voyager with Typesafe AI for decisions and an OpenRouter LLM for code.

  - curriculum: which unlocked task to work on next   -> Typesafe Choice
  - critic:     did the task succeed                  -> Typesafe Noul
  - action:     writes the mineflayer JavaScript      -> LLM via OpenRouter

Typesafe only classifies text, so it cannot write code; the original
ActionAgent (prompts, parsing) is reused with its LLM pointed at OpenRouter.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import voyager.utils as U
from .control_primitives import load_control_primitives
from .env import VoyagerEnv

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Task:
    name: str
    item: str  # item that must be in the inventory when the task is done
    count: int = 1
    needs: Dict[str, int] = field(default_factory=dict)  # inventory prerequisites


TECH_TREE: List[Task] = [
    Task("Mine 3 wood logs", "log", 3),
    Task("Craft 4 oak planks", "oak_planks", 4, {"log": 1}),
    Task("Craft a crafting table", "crafting_table", 1, {"oak_planks": 4}),
    Task("Craft 4 sticks", "stick", 4, {"oak_planks": 2}),
    Task("Craft a wooden pickaxe", "wooden_pickaxe", 1, {"stick": 2, "oak_planks": 3, "crafting_table": 1}),
    Task("Mine 5 cobblestone", "cobblestone", 5, {"wooden_pickaxe": 1}),
    Task("Craft a stone pickaxe", "stone_pickaxe", 1, {"stick": 2, "cobblestone": 3, "crafting_table": 1}),
    Task("Craft a furnace", "furnace", 1, {"cobblestone": 8}),
    Task("Mine 3 coal", "coal", 3, {"wooden_pickaxe": 1}),
    Task("Mine 3 raw iron", "raw_iron", 3, {"stone_pickaxe": 1}),
    Task("Smelt 3 iron ingots", "iron_ingot", 3, {"raw_iron": 3, "coal": 1, "furnace": 1}),
    Task("Craft an iron pickaxe", "iron_pickaxe", 1, {"stick": 2, "iron_ingot": 3, "crafting_table": 1}),
]


class TypeSafeDecider:
    """Thin wrapper over the Typesafe SDK. Skips the API when there is nothing to decide."""

    def __init__(self, api_key: Optional[str] = None, client=None):
        if client is None:
            from typesafe_sdk import TypeSafeClient  # pip install typesafe-sdk

            key = api_key or os.environ.get("TYPESAFE_API_KEY")
            client = TypeSafeClient(api_key=key) if key else TypeSafeClient()
        self.client = client

    def choose(self, state: str, instructions: str, options: Dict[str, str]) -> str:
        if len(options) == 1:
            return next(iter(options))
        from typesafe_sdk import Choice

        response = self.client.system_one(
            state=state,
            questions={"pick": Choice(instructions=instructions, criteria=options)},
        )
        picked = response.answers["pick"].choice
        return picked if picked in options else next(iter(options))

    def yes(self, state: str, instructions: str) -> bool:
        from typesafe_sdk import Noul

        response = self.client.system_one(
            state=state, questions={"answer": Noul(instructions=instructions)}
        )
        return response.answers["answer"].noul >= 0.5


def _have(inventory: Dict[str, int], item: str) -> int:
    if item == "log":  # any wood type counts
        return sum(n for k, n in inventory.items() if k.endswith("_log"))
    return inventory.get(item, 0)


def render_state(events, task: Optional[Task] = None, extra: str = "") -> str:
    obs = events[-1][1]
    status = obs["status"]
    lines = [
        f"Biome: {status['biome']}",
        f"Time: {status['timeOfDay']}",
        f"Health: {status['health']:.1f}/20, Hunger: {status['food']:.1f}/20",
        f"Inventory: {obs['inventory'] or 'Empty'}",
        f"Nearby blocks: {', '.join(obs['voxels']) if obs['voxels'] else 'None'}",
    ]
    if task:
        lines.append(f"Task: {task.name} (need {task.count} {task.item})")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


class TypeSafeCurriculum:
    def __init__(self, decider: TypeSafeDecider, tasks: List[Task] = None, max_failures=3):
        self.decider = decider
        self.tasks = tasks or TECH_TREE
        self.max_failures = max_failures
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []

    def propose_next_task(self, events) -> Optional[Task]:
        inventory = events[-1][1]["inventory"]
        candidates = [
            t
            for t in self.tasks
            if t.name not in self.completed_tasks
            and self.failed_tasks.count(t.name) < self.max_failures
            and _have(inventory, t.item) < t.count
            and all(_have(inventory, k) >= n for k, n in t.needs.items())
        ]
        if not candidates:
            return None
        picked = self.decider.choose(
            render_state(events),
            "Which task should the Minecraft bot do next? Prefer tasks that "
            "unlock progress and are safe given health, hunger and time of day.",
            {t.name: f"Work towards {t.count} {t.item}" for t in candidates},
        )
        return next(t for t in candidates if t.name == picked)

    def update_exploration_progress(self, task: Task, success: bool):
        (self.completed_tasks if success else self.failed_tasks).append(task.name)


class TypeSafeCritic:
    def __init__(self, decider: TypeSafeDecider):
        self.decider = decider

    def check_task_success(self, events, task: Task):
        """Returns (success, critique). The critique is templated, not generated."""
        errors = [e["onError"] for t, e in events if t == "onError"]
        if errors:
            return False, f"Execution error: {errors[-1]}"
        have = _have(events[-1][1]["inventory"], task.item)
        state = render_state(
            events, task, f"Required: at least {task.count} {task.item} (currently {have})."
        )
        if self.decider.yes(
            state, "Does the inventory show the task was completed successfully?"
        ):
            return True, ""
        return False, f"Task not complete: have {have}/{task.count} {task.item}."


class TypeSafeVoyager:
    def __init__(
        self,
        mc_port: int = None,
        azure_login: Dict[str, str] = None,
        server_port: int = 3000,
        typesafe_api_key: str = None,
        openrouter_api_key: str = None,
        action_agent_model_name: str = "anthropic/claude-sonnet-4.5",
        action_agent_temperature: float = 0,
        env_wait_ticks: int = 20,
        env_request_timeout: int = 600,
        max_iterations: int = 60,
        task_max_retries: int = 4,
        ckpt_dir: str = "ckpt",
        decider: TypeSafeDecider = None,
    ):
        key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("Set OPENROUTER_API_KEY or pass openrouter_api_key")
        # the OpenAI-compatible langchain client used by ActionAgent reads these
        os.environ["OPENAI_API_KEY"] = key
        os.environ["OPENAI_API_BASE"] = OPENROUTER_BASE_URL
        from .agents import ActionAgent  # imported late so the env vars above apply

        self.env = VoyagerEnv(
            mc_port=mc_port,
            azure_login=azure_login,
            server_port=server_port,
            request_timeout=env_request_timeout,
        )
        self.env_wait_ticks = env_wait_ticks
        self.max_iterations = max_iterations
        self.task_max_retries = task_max_retries
        decider = decider or TypeSafeDecider(api_key=typesafe_api_key)
        self.curriculum = TypeSafeCurriculum(decider)
        self.critic = TypeSafeCritic(decider)
        self.action = ActionAgent(
            model_name=action_agent_model_name,
            temperature=action_agent_temperature,
            ckpt_dir=ckpt_dir,
        )
        self.recorder = U.EventRecorder(ckpt_dir=ckpt_dir, resume=False)
        self.ckpt_dir = ckpt_dir
        self.control_primitives = load_control_primitives()
        self.skills: Dict[str, str] = {}  # program_name -> code, from successful tasks

    @property
    def programs(self) -> str:
        return "\n\n".join(list(self.skills.values()) + self.control_primitives)

    def close(self):
        self.env.close()

    def rollout(self, task: Task, events):
        skills = list(self.skills.values())
        code, critique = "", ""
        for _ in range(self.task_max_retries):
            messages = [
                self.action.render_system_message(skills=skills),
                self.action.render_human_message(
                    events=events, code=code, task=task.name, context="", critique=critique
                ),
            ]
            parsed = self.action.process_ai_message(self.action.llm(messages))
            if not isinstance(parsed, dict):
                critique = parsed  # parse error, let the LLM retry
                continue
            code = parsed["program_code"]
            events = self.env.step(
                code + "\n" + parsed["exec_code"], programs=self.programs
            )
            self.recorder.record(events, task.name)
            self.action.update_chest_memory(events[-1][1]["nearbyChests"])
            success, critique = self.critic.check_task_success(events, task)
            if success:
                self.skills[parsed["program_name"]] = code
                U.dump_json(self.skills, f"{self.ckpt_dir}/typesafe_skills.json")
                return True, events
        return False, events

    def learn(self):
        self.env.reset(options={"mode": "hard", "wait_ticks": self.env_wait_ticks})
        events = self.env.step("")
        for _ in range(self.max_iterations):
            task = self.curriculum.propose_next_task(events)
            if task is None:
                print("No more available tasks.")
                break
            print(f"\033[35mStarting task: {task.name}\033[0m")
            try:
                success, events = self.rollout(task, events)
            except Exception as e:
                time.sleep(3)  # wait for mineflayer to exit
                print(f"\033[41m{e}\033[0m")
                success = False
                events = self.env.reset(
                    options={
                        "mode": "hard",
                        "wait_ticks": self.env_wait_ticks,
                        "inventory": events[-1][1]["inventory"],
                        "equipment": events[-1][1]["status"]["equipment"],
                        "position": events[-1][1]["status"]["position"],
                    }
                )
            self.curriculum.update_exploration_progress(task, success)
        return {
            "completed_tasks": self.curriculum.completed_tasks,
            "failed_tasks": self.curriculum.failed_tasks,
            "skills": list(self.skills),
        }
