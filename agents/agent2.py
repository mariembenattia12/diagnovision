from __future__ import annotations

from dataclasses import dataclass, asdict
import subprocess
from typing import Iterable


@dataclass
class DiagnosticOutput:
    label: str
    pathology: str
    confidence: float
    severity: str
    concordance: str
    decision: str
    argumentation: str
    recommendation: str
    warning: str

    def to_dict(self) -> dict:
        return asdict(self)


def _build_argumentation(agent1_report: str, symptoms: str, probability: float) -> str:
    return (
        f"Le rapport Agent 1 indique des anomalies radiographiques compatibles avec une atteinte "
        f"infectieuse (score image {probability:.1%}). Symptomes rapportes: {symptoms.strip() or 'non specifies'}. "
        "La fusion image + clinique renforce l'hypothese de pneumonie communautaire."
    )


def run_agent2(agent1_report: str, symptoms: str, vital_signs: list[str], probability: float) -> DiagnosticOutput:
    import subprocess, json, re

    suspicious_vitals   = len(vital_signs)
    boosted_confidence  = min(0.99, probability + 0.04 * suspicious_vitals)

    merged_text = (
        f"[Rapport Agent1]\n{agent1_report}\n\n"
        f"[Symptomes]\n{symptoms or 'non specifies'}\n\n"
        f"[Signes vitaux anormaux]\n{', '.join(vital_signs) or 'Aucun'}\n\n"
        f"[Score image Agent1]\n{probability:.1%}"
    )

    prompt = f"""Tu es le Dr. IA, medecin specialiste en pneumologie.
Analyse ce texte et reponds UNIQUEMENT en JSON valide, sans texte avant ni apres.

Texte:
{merged_text}

Reponds avec exactement ces champs:
{{"label": "POSITIF ou NEGATIF", "pathology": "nom pathologie", "confidence": 0.00, "severity": "Moderee|Faible|Severe", "concordance": "Elevee|Moyenne|Faible", "decision": "action recommandee", "argumentation": "raisonnement 2 phrases", "recommendation": "prochaine etape", "warning": "limite IA"}}

JSON:"""

    try:
        result = subprocess.run(
    ["ollama", "run", "llama3:8b", "--nowordwrap", prompt],
    capture_output=True, 
    text=True, 
    timeout=60, 
    check=False,
    encoding='utf-8'  # <--- AJOUTEZ CETTE LIGNE
)
        if result.returncode == 0 and result.stdout.strip():
            raw   = result.stdout.strip()
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return DiagnosticOutput(
                    label          = data.get("label", "NEGATIF"),
                    pathology      = data.get("pathology", "Indetermine"),
                    confidence     = float(data.get("confidence", boosted_confidence)),
                    severity       = data.get("severity", "Faible"),
                    concordance    = data.get("concordance", "Moyenne"),
                    decision       = data.get("decision", "Surveillance clinique"),
                    argumentation  = data.get("argumentation", _build_argumentation(agent1_report, symptoms, boosted_confidence)),
                    recommendation = data.get("recommendation", "Reevaluation sous 24-48h"),
                    warning        = data.get("warning", "Aide au diagnostic uniquement — validation medicale obligatoire."),
                )
    except Exception:
        pass  # fallback ci-dessous si Ollama indisponible

    # ── Fallback déterministe si Ollama est indisponible ──────────────────
    if boosted_confidence >= 0.58:
        label = "POSITIF"; pathology = "Pneumonie bacterienne probable"
        severity = "Moderee"; concordance = "Elevee"
        decision = "Hospitaliser / antibiotherapie a discuter"
        recommendation = "Verifier NFS/CRP, gaz du sang si necessaire, surveillance SpO2 et avis pneumologue."
    else:
        label = "NEGATIF"; pathology = "Pas d'atteinte pneumonique franche"
        severity = "Faible"; concordance = "Moyenne"
        decision = "Surveillance clinique"
        recommendation = "Reevaluation clinique sous 24-48h, repetition imagerie si aggravation."

    return DiagnosticOutput(
        label=label, pathology=pathology, confidence=boosted_confidence,
        severity=severity, concordance=concordance, decision=decision,
        argumentation=_build_argumentation(agent1_report, symptoms, boosted_confidence),
        recommendation=recommendation,
        warning="Ceci est une aide au diagnostic. Toute decision medicale doit etre validee par un medecin qualifie.",
    )


def _history_to_text(messages: Iterable[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        speaker = "DrIA" if role == "assistant" else "User"
        lines.append(f"{speaker}: {msg.get('content', '')}")
    return "\n".join(lines)


def _fallback_reply(question: str, diagnosis: DiagnosticOutput) -> str:
    question_l = question.lower()
    if "confiance" in question_l or "score" in question_l:
        return f"Le score de confiance estime est de {diagnosis.confidence:.1%}."
    if "pourquoi" in question_l or "argument" in question_l:
        return diagnosis.argumentation
    if "recommand" in question_l or "suite" in question_l:
        return diagnosis.recommendation
    return (
        f"Diagnostic actuel: {diagnosis.pathology} ({diagnosis.label}, {diagnosis.confidence:.1%}). "
        "Je peux preciser l'argumentation, la severite ou les prochaines etapes."
    )


def run_agent2_with_memory(
    messages: list[dict],
    diagnosis: DiagnosticOutput,
    model_name: str = "llama3:8b",
) -> str:
    if not messages:
        return _fallback_reply("", diagnosis)

    user_question = messages[-1]["content"]
    history = _history_to_text(messages[-12:]) # Augmenté pour une meilleure mémoire

    # Prompt Expert optimisé
    system_context = f"""Tu es le Dr. IA, un expert senior en pneumologie et imagerie médicale.
Ton rôle est d'interpréter les résultats et d'accompagner le clinicien ou le patient avec une expertise de haut niveau.

FICHE CLINIQUE DU PATIENT:
- Diagnostic suspecté : {diagnosis.pathology}
- Statut : {diagnosis.label}
- Certitude algorithmique : {diagnosis.confidence:.1%}
- Niveau de sévérité : {diagnosis.severity}
- Analyse des biomarqueurs/images : {diagnosis.argumentation}
- Protocole suggéré : {diagnosis.recommendation}

DIRECTIVES D'INTELLIGENCE :
1. RAISONNEMENT CLINIQUE : Ne te contente pas de citer les résultats. Si on te pose une question, fais le lien entre les signes cliniques (toux, fièvre) et les observations radiologiques (opacités, foyers).
2. CAPACITÉ D'EXPLICATION : Si l'utilisateur pose une question complexe ou générale, utilise ta base de connaissances médicale pour expliquer les mécanismes physiopathologiques.
3. ADAPTABILITÉ : Si l'utilisateur semble inquiet, adopte un ton rassurant mais restez factuel.
4. NUANCE : Rappelle subtilement que l'IA complète l'examen clinique mais ne remplace pas la palpation ou l'auscultation réelle.
5. CONCISION EXPERTE : Va droit au but, évite les répétitions inutiles du diagnostic si la question est précise.

HISTORIQUE DES ÉCHANGES :
{history}
"""
    
    # Structure de réponse forcée pour éviter les bavardages inutiles d'Ollama
    prompt = f"{system_context}\nQuestion : {user_question}\nRéponse structurée du Dr. IA :"

    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True, 
            text=True, 
            timeout=120,
            check=False,
            encoding='utf-8'
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
            
    except Exception as e:
        print(f"Erreur système Ollama: {e}")
        pass

    return _fallback_reply(user_question, diagnosis)