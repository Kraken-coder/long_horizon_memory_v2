# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Long Horizon Memory Environment Implementation.

This version is aligned with the compressed-memory action schema:
- append: append current message to memory
- rewrite: replace memory with provided compressed memory
- noop: skip the current message

Rewards are shaped for stable training and include:
- per-step relevance rewards
- rewrite quality and growth penalties
- memory budget pressure
- dense potential-based shaping
- terminal QA reward from hybrid semantic matching
"""

import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import LongHorizonMemoryAction, LongHorizonMemoryObservation
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import LongHorizonMemoryAction, LongHorizonMemoryObservation
    except (ImportError, ModuleNotFoundError):
        from long_horizon_memory.models import LongHorizonMemoryAction, LongHorizonMemoryObservation


class LongHorizonMemoryEnvironment(Environment):
    """Environment where an agent manages compressed long-horizon memory."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MEMORY_TOKEN_BUDGET = 250
    MAX_REWRITE_GROWTH_RATIO = 1.40

    APPEND_RELEVANT_REWARD = 0.18
    APPEND_IRRELEVANT_PENALTY = -0.14
    NOOP_IRRELEVANT_REWARD = 0.05
    NOOP_RELEVANT_PENALTY = -0.16

    REWRITE_RELEVANT_BASE_REWARD = 0.08
    REWRITE_IRRELEVANT_PENALTY = -0.02
    REWRITE_GROWTH_PENALTY_MAX = 0.25

    QUALITY_DELTA_WEIGHT = 0.15
    POTENTIAL_SHAPING_WEIGHT = 0.45
    TERMINAL_WEIGHT = 0.50

    # Counterfactual and LLM-based rewards
    COUNTERFACTUAL_WEIGHT = 0.20
    USE_LLM_JUDGE = True  # ALWAYS ON - LLM answers questions, similarity judges
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    QA_MODEL = os.getenv("QA_MODEL", "llama3.2:1b")
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.2:1b")  # Not used anymore (kept for compatibility)

    def __init__(self):
        episodes_path = Path(__file__).with_name("episodes_extreme_30.json")
        with episodes_path.open("r", encoding="utf-8") as f:
            self.episodes = json.load(f)

        self._task_name = os.getenv("LONG_HORIZON_MEMORY_TASK", "all").strip().lower() or "all"
        seed_env = os.getenv("LONG_HORIZON_MEMORY_SEED")
        self._seed = int(seed_env) if seed_env and seed_env.lstrip("-").isdigit() else None
        self._rng = random.Random(self._seed)
        self._episode_id_override = os.getenv("LONG_HORIZON_MEMORY_EPISODE_ID")

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        self.episode = 0
        self.current_difficulty = "easy"
        self.messages: List[Dict[str, Any]] = []
        self.key_facts: List[Dict[str, Any]] = []
        self.questions: List[Dict[str, Any]] = []

        self.total_message_number = 0
        self.total_relevant_in_episode = 0
        self.memory_text = ""

        self.last_action_error: Optional[str] = None
        self._last_reward_breakdown: Dict[str, float] = {}
        self._last_quality_score = 0.0
        self._last_potential_score = 0.0
        self._done = False
        self._idf: Dict[str, float] = {}
        self._idf_default = 1.0

        self._set_random_episode()

    def _infer_difficulty(self, episode_data: Dict[str, Any], episode_index: int) -> str:
        explicit = str(episode_data.get("difficulty", "")).strip().lower()
        if explicit in {"easy", "medium", "hard"}:
            return explicit
        if episode_index <= 1:
            return "easy"
        if episode_index <= 3:
            return "medium"
        return "hard"

    def _candidate_indices_for_task(self) -> List[int]:
        if self._task_name not in {"easy", "medium", "hard", "all"}:
            self._task_name = "all"

        if self._task_name == "all":
            return list(range(len(self.episodes)))

        return [
            i
            for i, episode_data in enumerate(self.episodes)
            if self._infer_difficulty(episode_data, i) == self._task_name
        ]

    def _set_random_episode(self) -> None:
        candidates = self._candidate_indices_for_task()
        if not candidates:
            candidates = list(range(len(self.episodes)))

        chosen_episode: Optional[int] = None
        if self._episode_id_override:
            for idx in candidates:
                if str(self.episodes[idx].get("episode_id", idx)) == str(self._episode_id_override):
                    chosen_episode = idx
                    break

        self.episode = chosen_episode if chosen_episode is not None else self._rng.choice(candidates)
        episode_data = self.episodes[self.episode]

        self.current_difficulty = self._infer_difficulty(episode_data, self.episode)
        self.messages = list(episode_data.get("messages", []))
        self.key_facts = list(episode_data.get("key_facts", []))
        self.questions = list(episode_data.get("questions", []))

        self.total_message_number = 0
        self.total_relevant_in_episode = sum(1 for m in self.messages if bool(m.get("isRelevant", False)))
        self.memory_text = ""
        self.last_action_error = None
        self._last_reward_breakdown = {}
        self._done = len(self.messages) == 0
        self._build_episode_idf()
        self._last_quality_score = self._quality_score(self.memory_text)
        self._last_potential_score = self._potential_score(self.memory_text)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _token_count(self, text: str) -> int:
        return len(self._tokenize(text))

    def _normalize_memory(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _build_episode_idf(self) -> None:
        docs: List[str] = []
        docs.extend(str(m.get("text", "")) for m in self.messages)
        docs.extend(str(f.get("text", "")) for f in self.key_facts)
        docs.extend(str(q.get("question", "")) for q in self.questions)
        docs.extend(str(q.get("answer", "")) for q in self.questions)
        docs = [d for d in docs if d.strip()]

        if not docs:
            self._idf = {}
            self._idf_default = 1.0
            return

        df: Counter[str] = Counter()
        for doc in docs:
            uniq = set(self._tokenize(doc))
            for tok in uniq:
                df[tok] += 1

        n_docs = len(docs)
        self._idf = {
            tok: math.log((n_docs + 1.0) / (count + 1.0)) + 1.0
            for tok, count in df.items()
        }
        self._idf_default = math.log((n_docs + 1.0) / 1.0) + 1.0

    def _tfidf_vector(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = float(sum(tf.values()))
        vec: Dict[str, float] = {}
        for tok, count in tf.items():
            idf = self._idf.get(tok, self._idf_default)
            vec[tok] = (count / total) * idf
        return vec

    def _cosine_sparse(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        shared = set(a).intersection(b)
        dot = sum(a[t] * b[t] for t in shared)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))

    def _token_f1(self, a_text: str, b_text: str) -> float:
        a = Counter(self._tokenize(a_text))
        b = Counter(self._tokenize(b_text))
        if not a or not b:
            return 0.0
        overlap = sum(min(a[t], b[t]) for t in set(a).intersection(b))
        p = overlap / max(1, sum(a.values()))
        r = overlap / max(1, sum(b.values()))
        if p + r <= 0:
            return 0.0
        return 2.0 * p * r / (p + r)

    def _char_ngram_cosine(self, a_text: str, b_text: str, n: int = 3) -> float:
        def grams(s: str) -> Counter[str]:
            s = re.sub(r"\s+", " ", s.lower()).strip()
            if len(s) < n:
                return Counter()
            return Counter(s[i : i + n] for i in range(len(s) - n + 1))

        a = grams(a_text)
        b = grams(b_text)
        if not a or not b:
            return 0.0
        shared = set(a).intersection(b)
        dot = sum(float(a[g] * b[g]) for g in shared)
        na = math.sqrt(sum(float(v * v) for v in a.values()))
        nb = math.sqrt(sum(float(v * v) for v in b.values()))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))

    def _hybrid_similarity(self, a_text: str, b_text: str) -> float:
        if not a_text.strip() or not b_text.strip():
            return 0.0
        tfidf = self._cosine_sparse(self._tfidf_vector(a_text), self._tfidf_vector(b_text))
        tok_f1 = self._token_f1(a_text, b_text)
        chr_sim = self._char_ngram_cosine(a_text, b_text)
        return max(0.0, min(1.0, 0.60 * tfidf + 0.25 * tok_f1 + 0.15 * chr_sim))

    def _memory_segments(self, memory_text: str) -> List[str]:
        segs = [s.strip() for s in re.split(r"[\n\.!?;]+", memory_text) if s.strip()]
        if not segs and memory_text.strip():
            segs = [memory_text.strip()]
        return segs

    def _memory_relevance_similarity(self, memory_text: str) -> float:
        relevant_memory = "\n".join(str(f.get("text", "")) for f in self.key_facts)
        if not relevant_memory.strip():
            return 0.0
        return self._hybrid_similarity(memory_text, relevant_memory)

    def _fact_coverage(self, memory_text: str) -> float:
        if not self.key_facts:
            return 0.0

        segments = self._memory_segments(memory_text)
        if not segments:
            return 0.0
        sims: List[float] = []
        for fact in self.key_facts:
            fact_text = str(fact.get("text", ""))
            best = max((self._hybrid_similarity(seg, fact_text) for seg in segments), default=0.0)
            sims.append(best)
        return sum(sims) / len(sims)

    def _answer_question(self, memory_text: str, question: str) -> str:
        if not memory_text.strip():
            return ""

        candidates = self._memory_segments(memory_text)
        if not candidates:
            return ""

        ranked = sorted(
            candidates,
            key=lambda s: self._hybrid_similarity(s, question),
            reverse=True,
        )
        top = [s for s in ranked[:2] if s]
        return ". ".join(top).strip()

    def _number_overlap_score(self, a_text: str, b_text: str) -> float:
        a_nums = set(re.findall(r"\d+(?:\.\d+)?", a_text))
        b_nums = set(re.findall(r"\d+(?:\.\d+)?", b_text))
        if not a_nums or not b_nums:
            return 0.0
        inter = len(a_nums.intersection(b_nums))
        union = len(a_nums.union(b_nums))
        return inter / max(1, union)

    def _qa_similarity_score(self, memory_text: str) -> float:
        metrics = self._qa_metrics(memory_text)
        return metrics["qa_score"]

    def _qa_metrics(self, memory_text: str) -> Dict[str, float]:
        if not self.questions:
            return {"qa_score": 0.0, "exact_hit_rate": 0.0, "answerable_rate": 0.0}

        scores: List[float] = []
        exact_hits = 0
        answerable = 0
        for q in self.questions:
            question = str(q.get("question", ""))
            expected_answer = str(q.get("answer", "")).strip()
            predicted = self._answer_question(memory_text, question)

            if not expected_answer:
                continue

            if predicted.strip():
                answerable += 1

            sem = self._hybrid_similarity(predicted, expected_answer)
            num = self._number_overlap_score(predicted, expected_answer)
            sim = 0.80 * sem + 0.20 * num
            if expected_answer.lower() in predicted.lower() or predicted.lower() in expected_answer.lower():
                exact_hits += 1
                sim = max(sim, 1.0)
            scores.append(sim)

        if not scores:
            return {"qa_score": 0.0, "exact_hit_rate": 0.0, "answerable_rate": 0.0}
        return {
            "qa_score": sum(scores) / len(scores),
            "exact_hit_rate": exact_hits / len(scores),
            "answerable_rate": answerable / len(scores),
        }

    def _memory_overflow_penalty(self, memory_text: str) -> float:
        token_count = self._token_count(memory_text)
        overflow = max(0, token_count - self.MEMORY_TOKEN_BUDGET)
        if overflow == 0:
            return 0.0
        ratio = overflow / max(1.0, float(self.MEMORY_TOKEN_BUDGET))
        return min(0.45, 0.10 * ratio + 0.35 * (ratio ** 2))

    def _quality_score(self, memory_text: str) -> float:
        fact_coverage = self._fact_coverage(memory_text)
        qa_score = self._qa_metrics(memory_text)["qa_score"]
        relevance = self._memory_relevance_similarity(memory_text)
        overflow_penalty = self._memory_overflow_penalty(memory_text)

        score = (
            0.40 * fact_coverage
            + 0.35 * qa_score
            + 0.25 * relevance
            - 0.35 * overflow_penalty
        )
        return max(0.0, min(1.0, score))

    def _potential_score(self, memory_text: str) -> float:
        qa_metrics = self._qa_metrics(memory_text)
        fact_coverage = self._fact_coverage(memory_text)
        relevance = self._memory_relevance_similarity(memory_text)
        overflow_penalty = self._memory_overflow_penalty(memory_text)
        potential = (
            0.35 * fact_coverage
            + 0.35 * qa_metrics["qa_score"]
            + 0.15 * qa_metrics["answerable_rate"]
            + 0.15 * relevance
            - 0.30 * overflow_penalty
        )
        return max(0.0, min(1.0, potential))

    def _rewrite_reward(self, old_memory: str, new_memory: str, message_is_relevant: bool) -> Dict[str, float]:
        old_tokens = self._token_count(old_memory)
        new_tokens = self._token_count(new_memory)

        old_quality = self._quality_score(old_memory)
        new_quality = self._quality_score(new_memory)

        reward = 0.0
        quality_delta = new_quality - old_quality
        if message_is_relevant:
            reward += self.REWRITE_RELEVANT_BASE_REWARD
            reward += 0.25 * quality_delta
        else:
            # Permit beneficial rewrites even on irrelevant turns.
            reward += self.REWRITE_IRRELEVANT_PENALTY
            reward += 0.10 * quality_delta

        growth_penalty = 0.0
        if old_tokens > 0:
            growth_ratio = new_tokens / old_tokens
            if growth_ratio > self.MAX_REWRITE_GROWTH_RATIO:
                over = growth_ratio - self.MAX_REWRITE_GROWTH_RATIO
                growth_penalty = min(self.REWRITE_GROWTH_PENALTY_MAX, 0.15 * over)
                reward -= growth_penalty
        elif new_tokens > self.MEMORY_TOKEN_BUDGET // 3:
            growth_penalty = min(self.REWRITE_GROWTH_PENALTY_MAX, 0.10)
            reward -= growth_penalty

        return {
            "rewrite_reward": reward,
            "growth_penalty": growth_penalty,
            "old_quality": old_quality,
            "new_quality": new_quality,
        }

    def _current_message(self) -> Optional[Dict[str, Any]]:
        if self.total_message_number >= len(self.messages):
            return None
        return self.messages[self.total_message_number]

    def _call_qa_model(self, memory: str, question: str) -> str:
        """
        Use external LLM to answer question based on memory.

        This simulates how a real user would query the memory system.

        Args:
            memory: The compressed memory string
            question: Question to answer

        Returns:
            LLM's predicted answer
        """
        if not memory.strip():
            return ""

        prompt = f"""Based ONLY on the following memory, answer the question concisely.
If the memory doesn't contain the answer, say "Unknown".

Memory:
{memory}

Question: {question}

Answer (be brief):"""

        try:
            import requests

            response = requests.post(
                f"{self.OLLAMA_URL}/api/generate",
                json={
                    "model": self.QA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 50,
                    },
                },
                timeout=10,
            )

            if response.status_code == 200:
                answer = response.json().get("response", "").strip()
                return answer
            else:
                return self._answer_question(memory, question)

        except Exception:
            return self._answer_question(memory, question)

    def _judge_answer_quality(
        self, question: str, predicted: str, ground_truth: str
    ) -> float:
        """
        Use LLM to judge how well predicted answer matches ground truth.

        This is more nuanced than string similarity - it understands semantics,
        paraphrasing, and partial correctness.

        Args:
            question: The question being answered
            predicted: LLM's predicted answer from memory
            ground_truth: Correct answer from dataset

        Returns:
            Score from 0.0 (completely wrong) to 1.0 (perfect)
        """
        if not predicted.strip() or not ground_truth.strip():
            return 0.0

        prompt = f"""You are evaluating QA system answers. Rate how well the Predicted answer matches the Ground Truth for the given Question.

Question: {question}
Ground Truth: {ground_truth}
Predicted: {predicted}

Rating criteria:
- 1.0: Perfect match or equivalent meaning
- 0.7-0.9: Correct but incomplete or slightly off
- 0.4-0.6: Partially correct
- 0.1-0.3: Wrong but related
- 0.0: Completely wrong or unrelated

Provide ONLY a decimal score between 0.0 and 1.0, nothing else.
Score:"""

        try:
            import requests

            response = requests.post(
                f"{self.OLLAMA_URL}/api/generate",
                json={
                    "model": self.JUDGE_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=15,
            )

            if response.status_code == 200:
                text = response.json().get("response", "0.0").strip()
                match = re.search(r"(\d+\.?\d*)", text)
                if match:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))

            return self._hybrid_similarity(predicted, ground_truth)

        except Exception:
            return self._hybrid_similarity(predicted, ground_truth)

    def _llm_qa_score(self, memory_text: str) -> Dict[str, float]:
        """
        Calculate QA performance using actual LLM instead of string matching.

        This replaces the heuristic _qa_metrics with real LLM-generated answers
        judged by another LLM.

        Returns:
            Dictionary with qa_score and other metrics
        """
        if not self.questions:
            return {"qa_score": 0.0, "llm_qa_score": 0.0}

        scores = []
        llm_scores = []

        for q in self.questions:
            question = str(q.get("question", ""))
            ground_truth = str(q.get("answer", "")).strip()

            if not ground_truth:
                continue

            predicted = self._call_qa_model(memory_text, question)

            if self.USE_LLM_JUDGE:
                llm_score = self._judge_answer_quality(question, predicted, ground_truth)
                llm_scores.append(llm_score)

            similarity_score = self._hybrid_similarity(predicted, ground_truth)
            scores.append(similarity_score)

        result = {
            "qa_score": sum(scores) / len(scores) if scores else 0.0,
        }

        if llm_scores:
            result["llm_qa_score"] = sum(llm_scores) / len(llm_scores)

        return result

    def _counterfactual_reward(
        self,
        operation: str,
        old_memory: str,
        new_memory: str,
        message_text: str,
    ) -> float:
        """
        Calculate advantage of chosen action vs alternatives.

        Simulates "what if I had chosen differently?" and rewards
        the agent for choosing actions that improve quality more than alternatives.

        Args:
            operation: Action taken ("append", "noop", "rewrite")
            old_memory: Memory before action
            new_memory: Memory after action
            message_text: Current message being processed

        Returns:
            Counterfactual advantage bonus/penalty
        """
        actual_quality = self._quality_score(new_memory)

        if operation == "append":
            # What if we had done noop instead?
            noop_quality = self._quality_score(old_memory)
            advantage = actual_quality - noop_quality

        elif operation == "noop":
            # What if we had appended instead?
            hypothetical_append = self._normalize_memory(
                f"{old_memory}\n{message_text}" if old_memory else message_text
            )
            append_quality = self._quality_score(hypothetical_append)
            advantage = actual_quality - append_quality

        elif operation == "rewrite":
            # What if we had kept old memory?
            noop_quality = self._quality_score(old_memory)
            advantage = actual_quality - noop_quality

        else:
            return 0.0

        return self.COUNTERFACTUAL_WEIGHT * advantage

    def _terminal_bonus(self) -> float:
        # Use LLM to answer questions if enabled
        if self.USE_LLM_JUDGE:
            # LLM answers questions, similarity judges them (12 API calls)
            llm_metrics = self._llm_qa_score(self.memory_text)
            qa_score = llm_metrics["qa_score"]

            # Still use heuristic for exact_hit and answerable_rate
            baseline_metrics = self._qa_metrics(self.memory_text)
            exact_hit_rate = baseline_metrics["exact_hit_rate"]
            answerable_rate = baseline_metrics["answerable_rate"]
        else:
            # Pure heuristic (no LLM calls)
            qa_metrics = self._qa_metrics(self.memory_text)
            qa_score = qa_metrics["qa_score"]
            exact_hit_rate = qa_metrics["exact_hit_rate"]
            answerable_rate = qa_metrics["answerable_rate"]

        fact_coverage = self._fact_coverage(self.memory_text)
        relevance = self._memory_relevance_similarity(self.memory_text)

        terminal = (
            0.35 * qa_score
            + 0.25 * exact_hit_rate
            + 0.15 * answerable_rate
            + 0.15 * fact_coverage
            + 0.10 * relevance
        )
        return max(0.0, min(1.0, terminal))

    def _task_score(self) -> float:
        quality = self._quality_score(self.memory_text)
        terminal = self._terminal_bonus() if self._done else 0.0
        score = (1.0 - self.TERMINAL_WEIGHT) * quality + self.TERMINAL_WEIGHT * terminal
        return max(0.0, min(1.0, score))

    def _observation(self, reward: float) -> LongHorizonMemoryObservation:
        current_message = self._current_message()
        new_message = "" if current_message is None else str(current_message.get("text", ""))

        metadata = {
            "reset_count": self._reset_count,
            "episode_id": self.episodes[self.episode].get("episode_id", self.episode),
            "task": self.current_difficulty,
            "memory_token_budget": self.MEMORY_TOKEN_BUDGET,
            "memory_token_count": self._token_count(self.memory_text),
            "fact_coverage": self._fact_coverage(self.memory_text),
            "qa_similarity": self._qa_similarity_score(self.memory_text),
            "memory_relevance_similarity": self._memory_relevance_similarity(self.memory_text),
            "potential_score": self._potential_score(self.memory_text),
            "task_score": self._task_score(),
            "last_action_error": self.last_action_error,
            "reward_breakdown": self._last_reward_breakdown,
        }

        return LongHorizonMemoryObservation(
            domain="long_horizon_memory",
            task_name=self.current_difficulty,
            new_message=new_message,
            memory=self.memory_text,
            memory_count=self._token_count(self.memory_text),
            reward=reward,
            done=self._done,
            metadata=metadata,
        )

    def reset(self) -> LongHorizonMemoryObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._set_random_episode()
        return self._observation(reward=0.0)

    def step(self, action: LongHorizonMemoryAction) -> LongHorizonMemoryObservation:  # type: ignore[override]
        self._state.step_count += 1
        self.last_action_error = None

        if self._done:
            self.last_action_error = "episode_already_done"
            self._last_reward_breakdown = {"already_done_penalty": -0.25}
            return self._observation(reward=-0.25)

        current_message = self._current_message()
        message_text = "" if current_message is None else str(current_message.get("text", ""))
        message_is_relevant = bool(current_message.get("isRelevant", False)) if current_message else False

        reward = 0.0
        breakdown: Dict[str, float] = {}
        prev_potential = self._last_potential_score

        # Save old memory for counterfactual calculation
        old_memory = self.memory_text

        operation = action.operation
        if operation == "append":
            if message_is_relevant:
                reward += self.APPEND_RELEVANT_REWARD
                breakdown["append_relevance"] = self.APPEND_RELEVANT_REWARD
            else:
                reward += self.APPEND_IRRELEVANT_PENALTY
                breakdown["append_relevance"] = self.APPEND_IRRELEVANT_PENALTY

            if message_text:
                self.memory_text = self._normalize_memory(
                    f"{self.memory_text}\n{message_text}" if self.memory_text else message_text
                )

        elif operation == "noop":
            if message_is_relevant:
                reward += self.NOOP_RELEVANT_PENALTY
                breakdown["noop_relevance"] = self.NOOP_RELEVANT_PENALTY
            else:
                reward += self.NOOP_IRRELEVANT_REWARD
                breakdown["noop_relevance"] = self.NOOP_IRRELEVANT_REWARD

        elif operation == "rewrite":
            proposed = action.rewrite_memory
            if proposed is None:
                self.last_action_error = "rewrite_memory_required"
                reward -= 0.20
                breakdown["rewrite_invalid"] = -0.20
            else:
                old_memory = self.memory_text
                new_memory = self._normalize_memory(proposed)
                rewrite_details = self._rewrite_reward(
                    old_memory=old_memory,
                    new_memory=new_memory,
                    message_is_relevant=message_is_relevant,
                )
                self.memory_text = new_memory
                rewrite_reward = float(rewrite_details["rewrite_reward"])
                reward += rewrite_reward
                breakdown["rewrite_reward"] = rewrite_reward
                if float(rewrite_details["growth_penalty"]) > 0:
                    breakdown["rewrite_growth_penalty"] = -float(rewrite_details["growth_penalty"])

        else:
            self.last_action_error = "invalid_operation"
            reward -= 0.20
            breakdown["invalid_operation"] = -0.20

        overflow_penalty = self._memory_overflow_penalty(self.memory_text)
        if overflow_penalty > 0:
            reward -= overflow_penalty
            breakdown["memory_overflow_penalty"] = -overflow_penalty

        # Add counterfactual reward for append/noop/rewrite
        if operation in ["append", "noop", "rewrite"]:
            counterfactual = self._counterfactual_reward(
                operation=operation,
                old_memory=old_memory,
                new_memory=self.memory_text,
                message_text=message_text,
            )
            reward += counterfactual
            breakdown["counterfactual_advantage"] = counterfactual

        new_quality = self._quality_score(self.memory_text)
        quality_delta = new_quality - self._last_quality_score
        delta_reward = self.QUALITY_DELTA_WEIGHT * quality_delta
        reward += delta_reward
        breakdown["quality_delta_reward"] = delta_reward
        self._last_quality_score = new_quality

        new_potential = self._potential_score(self.memory_text)
        potential_delta = new_potential - prev_potential
        potential_reward = self.POTENTIAL_SHAPING_WEIGHT * potential_delta
        reward += potential_reward
        breakdown["potential_shaping_reward"] = potential_reward
        self._last_potential_score = new_potential

        self.total_message_number += 1
        if self.total_message_number >= len(self.messages):
            self._done = True

        if self._done:
            terminal_bonus = self._terminal_bonus()
            reward += terminal_bonus
            breakdown["terminal_bonus"] = terminal_bonus

        reward = max(-1.0, min(1.0, reward))
        self._last_reward_breakdown = breakdown
        return self._observation(reward=reward)

    def close(self) -> None:
        return None

    @property
    def state(self) -> State:
        return self._state


if __name__ == "__main__":
    pass
