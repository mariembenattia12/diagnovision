from __future__ import annotations

from typing import Any

from agents.agent1 import DenseNet121Agent1
from agents.agent2 import run_agent2


def node_agent1(state: dict[str, Any]) -> dict[str, Any]:
    agent1: DenseNet121Agent1 = state["agent1"]
    output = agent1.analyze(state["uploaded_file"], state["symptoms"])
    state["agent1_probability"] = output.probability
    state["rapport_agent1"] = output.report
    state["gradcam_result"] = {
        "heatmap": output.heatmap,
        "overlay": output.overlay,
        "zone_activee": output.active_zone,
    }
    return state


def node_fusion(state: dict[str, Any]) -> dict[str, Any]:
    state["fusion_text"] = (
        f"[Rapport Agent1]\n{state['rapport_agent1']}\n\n"
        f"[Symptomes]\n{state['symptoms']}\n\n"
        f"[Signes vitaux anormaux]\n{', '.join(state['vital_signs']) or 'Aucun'}"
    )
    return state


def node_agent2(state: dict[str, Any]) -> dict[str, Any]:
    diagnosis = run_agent2(
        agent1_report=state["rapport_agent1"],
        symptoms=state["symptoms"],
        vital_signs=state["vital_signs"],
        probability=state["agent1_probability"],
    )
    state["diagnosis"] = diagnosis.to_dict()
    state["diagnosis_object"] = diagnosis
    return state


def run_pipeline_graph(
    uploaded_file,
    symptoms: str,
    vital_signs: list[str],
    agent1: DenseNet121Agent1 | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "uploaded_file": uploaded_file,
        "symptoms": symptoms,
        "vital_signs": vital_signs,
        "agent1": agent1 or DenseNet121Agent1(),
    }
    state = node_agent1(state)
    state = node_fusion(state)
    state = node_agent2(state)
    return state
