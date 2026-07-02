from __future__ import annotations

import random
import time
from datetime import datetime
from html import escape
from io import BytesIO

from PIL import Image
import streamlit as st

from agents.agent1 import DenseNet121Agent1
from agents.agent2 import DiagnosticOutput, run_agent2_with_memory
from pipeline.graph import node_agent1, node_agent2, node_fusion
from utils.report import generate_pdf_report


st.set_page_config(layout="wide", page_title="MedAI Vision", page_icon="🫁")


DEFAULT_VISION_REPORT = (
    "Aucune image n'a encore ete analysee. Importez une radiographie thoracique "
    "pour lancer l'Expert Vision."
)
DEFAULT_DIAGNOSIS_TEXT = (
    "Le diagnostic final sera genere apres combinaison du rapport vision et des "
    "symptomes cliniques."
)
AI_WARNING_TEXT = (
    "⚠ Ceci est une aide au diagnostic. Toute decision medicale doit etre validee "
    "par un medecin qualifie."
)


def init_state() -> None:
    defaults = {
        "uploaded_image": None,
        "uploaded_signature": None,
        "gradcam_result": None,
        "rapport_agent1": DEFAULT_VISION_REPORT,
        "diagnosis": None,
        "messages": [],
        "pipeline_status": "idle",
        "pipeline_details": {},
        "case_id": None,
        "agent1_probability": 0.0,
        "last_symptoms": "",
        "last_vitals": [],
        "chat_init_case": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_case_outputs() -> None:
    st.session_state.gradcam_result = None
    st.session_state.rapport_agent1 = DEFAULT_VISION_REPORT
    st.session_state.diagnosis = None
    st.session_state.pipeline_status = "idle"
    st.session_state.pipeline_details = {}
    st.session_state.messages = []
    st.session_state.chat_init_case = None
    st.session_state.agent1_probability = 0.0
    st.session_state.last_symptoms = ""
    st.session_state.last_vitals = []


def html_text(value: str) -> str:
    return escape(value).replace("\n", "<br>")


def make_case_id() -> str:
    return f"#P2M-{random.randint(10000, 99999)}-X"


def get_uploaded_metadata(uploaded_file, image: Image.Image) -> str:
    size_kb = len(uploaded_file.getvalue()) / 1024
    width, height = image.size
    return (
        f"Fichier: {uploaded_file.name} | Taille: {size_kb:.1f} KB | "
        f"Dimensions: {width}x{height} px"
    )


def load_uploaded_image(uploaded_file) -> Image.Image:
    return Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")


def ensure_case_from_upload(uploaded_file) -> None:
    if uploaded_file is None:
        return
    signature = f"{uploaded_file.name}:{len(uploaded_file.getvalue())}"
    if signature != st.session_state.uploaded_signature:
        st.session_state.uploaded_signature = signature
        st.session_state.uploaded_image = load_uploaded_image(uploaded_file)
        st.session_state.case_id = make_case_id()
        reset_case_outputs()


def get_diagnosis_object() -> DiagnosticOutput | None:
    diagnosis = st.session_state.diagnosis
    if isinstance(diagnosis, dict):
        return DiagnosticOutput(**diagnosis)
    return None


def ensure_welcome_message() -> None:
    diagnosis = get_diagnosis_object()
    if diagnosis is None or st.session_state.case_id is None:
        return
    if st.session_state.chat_init_case == st.session_state.case_id:
        return

    welcome = (
        "Bonjour, j'ai analyse le dossier du patient. Le diagnostic suggere "
        f"{diagnosis.pathology}. Avez-vous des questions sur les resultats ou "
        "souhaitez-vous des precisions ?"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome})
    st.session_state.chat_init_case = st.session_state.case_id

@st.cache_resource
def _get_agent1():
    return DenseNet121Agent1()

def run_pipeline(uploaded_file, symptoms: str, vital_signs: list[str]) -> None:
    st.session_state.pipeline_status = "running"
    state = {
    "uploaded_file": uploaded_file,
    "symptoms": symptoms,
    "vital_signs": vital_signs,
    "agent1": _get_agent1(),   # chargé une fois, mis en cache
}

    with st.status("Pipeline d'Execution — P2M", expanded=True) as status:
        st.markdown(
            "<div class='pipeline-step'>👁 Expert Vision (Agent 1) — Analyse radiographique en cours...</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Traitement DenseNet121..."):
            time.sleep(2.3)
            state = node_agent1(state)
        st.success(
            f"Rapport genere — Score: {state['agent1_probability']:.1%}"
        )

        st.markdown(
            "<div class='pipeline-step'>🔀 Fusion des donnees — Rapport + Symptomes cliniques...</div>",
            unsafe_allow_html=True,
        )
        fusion_progress = st.progress(0)
        for value in range(0, 101, 20):
            fusion_progress.progress(value)
            time.sleep(0.11)
        state = node_fusion(state)
        st.success("Texte fusionne pret pour l'Agent 2")

        st.markdown(
            "<div class='pipeline-step'>🧠 Expert Diagnostic (Agent 2) — Raisonnement clinique LLaMA 3 8B...</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Le Dr. IA analyse les donnees..."):
            time.sleep(2.0)
            state = node_agent2(state)
        st.success("Diagnostic structure genere")
        status.update(label="Pipeline termine", state="complete")

    st.session_state.pipeline_status = "done"
    st.session_state.gradcam_result = state["gradcam_result"]
    st.session_state.rapport_agent1 = state["rapport_agent1"]
    st.session_state.diagnosis = state["diagnosis"]
    st.session_state.agent1_probability = state["agent1_probability"]
    st.session_state.last_symptoms = symptoms
    st.session_state.last_vitals = vital_signs
    st.session_state.pipeline_details = {
        "step1": (
            f"Modele: DenseNet121 v1.0\n"
            f"Zone activee: {state['gradcam_result']['zone_activee']}\n"
            f"Score image: {state['agent1_probability']:.1%}\n"
            f"Rapport brut:\n{state['rapport_agent1']}"
        ),
        "step2": (
            "Fusion textuelle preparee pour Agent 2:\n"
            f"{state['fusion_text'][:650]}"
        ),
        "step3": (
            f"Label: {state['diagnosis']['label']}\n"
            f"Pathologie: {state['diagnosis']['pathology']}\n"
            f"Confiance: {state['diagnosis']['confidence']:.1%}\n"
            f"Decision: {state['diagnosis']['decision']}"
        ),
    }
    ensure_welcome_message()


def render_pipeline_details() -> None:
    if st.session_state.pipeline_status == "idle":
        return
    st.markdown("#### Pipeline d'Execution — P2M")
    with st.expander(
        "Etape 1 : Expert Vision (Agent 1) — Analyse radiographique",
        expanded=st.session_state.pipeline_status == "running",
    ):
        st.code(st.session_state.pipeline_details.get("step1", "En attente..."))
    with st.expander(
        "Etape 2 : Fusion des donnees — Rapport + Symptomes",
        expanded=False,
    ):
        st.code(st.session_state.pipeline_details.get("step2", "En attente..."))
    with st.expander(
        "Etape 3 : Expert Diagnostic (Agent 2) — Raisonnement clinique",
        expanded=False,
    ):
        st.code(st.session_state.pipeline_details.get("step3", "En attente..."))


def render_header() -> None:
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    st.markdown(
        f"""
<div class="top-header">
  <div class="top-left">
    <div class="brand">MedAI Vision</div>
    <div class="top-tabs">
      <span class="tab active">Dashboard</span>
      <span class="tab">Patients</span>
      <span class="tab">Archive</span>
      <span class="tab">Research</span>
    </div>
  </div>
  <div class="top-right">
    <div class="search-box">🔎 Search records...</div>
    <div class="avatar-chip">N</div>
    <div class="avatar-chip">U</div>
    <div class="datetime-chip">{now_str}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    # Début du cadre blanc
    st.markdown("<div class='sidebar-wrap'>", unsafe_allow_html=True)
    
    # Conteneur pour regrouper le haut (Logo + Nav)
    st.markdown("<div class='sidebar-top-content'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-logo'>MedAI Vision</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sidebar-subtitle'>Precision Diagnostics</div>",
        unsafe_allow_html=True,
    )
    
    # Navigation par radio
    nav = st.radio(
        "Navigation",
        ["Overview", "Diagnostics", "Vitals", "Reports", "Settings"],
        index=0,
        label_visibility="collapsed",
        key="sidebar_nav",
    )
    st.markdown("</div>", unsafe_allow_html=True) # Fin du top-content

    # Espaceur qui pousse tout le reste vers le bas
    st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)

    # Badge de version (poussé en bas du cadre blanc)
    st.markdown(
        """
        <div class="version-badge">
          <div>Agent 1: DenseNet121 v1.0</div>
          <div>Agent 2: LLaMA 3 8B</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("</div>", unsafe_allow_html=True) # Fin du sidebar-wrap
    return nav

def render_gradcam_section() -> None:
    st.markdown(
        """
<div class="section-row">
  <h3>Grad-CAM Visualization</h3>
  <span class="badge-agent">Agent 1 Output</span>
</div>
""",
        unsafe_allow_html=True,
    )
    gradcam_result = st.session_state.gradcam_result
    image = st.session_state.uploaded_image

    if gradcam_result is None or image is None:
        st.markdown(
            """
<div class="gradcam-placeholder">
  L'analyse Grad-CAM apparaitra ici apres traitement par l'Expert Vision
</div>
""",
            unsafe_allow_html=True,
        )
        return

    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.caption("Radiographie originale")
        st.image(image, width="stretch")
    with col2:
        st.caption("Carte d'activation")
        st.image(gradcam_result["heatmap"], width="stretch")
    with col3:
        st.caption("Superposition")
        st.image(gradcam_result["overlay"], width="stretch")

    st.markdown(
        f"<div class='zone-note'>Zone activee : {html_text(gradcam_result['zone_activee'])}</div>",
        unsafe_allow_html=True,
    )


def render_diagnostic_output() -> None:
    st.markdown("## AI Diagnostic Output")
    diagnosis = st.session_state.diagnosis

    if diagnosis is None:
        st.markdown("**Confidence Score: 0%**")
        st.markdown(
            f"<div class='info-card'>{html_text(DEFAULT_VISION_REPORT)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='info-card'>{html_text(DEFAULT_DIAGNOSIS_TEXT)}</div>",
            unsafe_allow_html=True,
        )
        return

    is_positive = diagnosis["label"] == "POSITIF"
    header_class = "result-header positive" if is_positive else "result-header negative"
    st.markdown(
        f"""
<div class="{header_class}">
  Resultat: {diagnosis['label']}
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Pathologie detectee : {diagnosis['pathology']}**")

    confidence = float(diagnosis["confidence"])
    st.markdown(f"**Confidence Score: {confidence:.1%}**")
    st.progress(confidence)

    m1, m2, m3 = st.columns(3, gap="small")
    m1.metric("Severite", diagnosis["severity"])
    m2.metric("Concordance", diagnosis["concordance"])
    m3.metric("Decision", diagnosis["decision"])

    st.info(diagnosis["argumentation"])
    st.success(diagnosis["recommendation"])
    st.warning(AI_WARNING_TEXT)

    gradcam = st.session_state.gradcam_result or {}
    report_bytes = generate_pdf_report(
        case_id=st.session_state.case_id or "--",
        diagnosis=diagnosis,
        report_agent1=st.session_state.rapport_agent1,
        symptoms=st.session_state.last_symptoms,
        active_zone=gradcam.get("zone_activee", "--"),
    )
    st.download_button(
        "Telecharger le rapport PDF",
        data=report_bytes,
        file_name=f"{(st.session_state.case_id or 'P2M').replace('#', '')}_report.pdf",
        mime="application/pdf",
        width="stretch",
    )


def render_chatbot() -> None:
    st.markdown(
        """
<div class="chat-title-row">
  <h3>Dr. IA — Suivi Clinique</h3>
  <span class="badge-memory">Memoire active</span>
</div>
<div class="chat-subtitle">Posez vos questions de suivi au medecin virtuel</div>
""",
        unsafe_allow_html=True,
    )

    with st.container(height=300):
        for msg in st.session_state.messages:
            content = html_text(msg["content"])
            if msg["role"] == "assistant":
                st.markdown(
                    f"""
<div class="chat-row bot">
  <div class="bot-avatar">🤖</div>
  <div class="chat-message bot">{content}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
<div class="chat-row user">
  <div class="chat-message user">{content}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

    prompt = st.chat_input("Poser une question au Dr. IA...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        diagnosis_obj = get_diagnosis_object()
        if diagnosis_obj is None:
            reply = (
                "Je n'ai pas encore de diagnostic structure pour ce dossier. "
                "Lancez d'abord l'analyse IA."
            )
        else:
            reply = run_agent2_with_memory(st.session_state.messages, diagnosis_obj)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.button("Reinitialiser la conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.chat_init_case = None
        ensure_welcome_message()
        st.rerun()


def render_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }

header, #MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

.stApp {
  background:
    radial-gradient(circle at 8% 8%, rgba(26,115,232,.09), transparent 28%),
    radial-gradient(circle at 92% 10%, rgba(41,181,232,.10), transparent 30%),
    #f2f6fb;
}

.block-container { max-width: 1700px; padding-top: 12px; padding-bottom: 20px; }

.top-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 12px 16px;
  border: 1px solid #dbe5f0;
  border-radius: 14px;
  background: rgba(255,255,255,.84);
  box-shadow: 0 8px 24px rgba(16,24,40,.06);
  margin-bottom: 10px;
}

.top-left { display: flex; align-items: center; gap: 28px; }
.brand { font-size: 24px; font-weight: 800; color: #10233f; }
.top-tabs { display: flex; gap: 16px; color: #61758f; }
.tab { padding: 4px 2px; font-size: 16px; }
.tab.active { color: #1a73e8; border-bottom: 3px solid #1a73e8; }

.top-right { display: flex; align-items: center; gap: 10px; }
.search-box {
  height: 44px;
  min-width: 270px;
  display: grid;
  place-items: center;
  border: 1px solid #dce7f3;
  border-radius: 999px;
  color: #7f90a7;
  background: #f8fbff;
  padding: 0 14px;
}
.avatar-chip {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: 1px solid #d6e1ef;
  display: grid;
  place-items: center;
  font-weight: 700;
  color: #4b627f;
  background: #ffffff;
}
.datetime-chip {
  font-size: 12px;
  color: #5f738e;
  border: 1px solid #dce7f3;
  border-radius: 10px;
  padding: 7px 9px;
  background: #f9fcff;
}

/* Dans votre fonction render_css() */

.sidebar-wrap {
  position: sticky;
  top: 12px;
  border: 1px solid #dce6f3;
  border-radius: 12px;
  background: rgba(255,255,255,.78);
  box-shadow: 0 10px 22px rgba(16,24,40,.05);
  min-height: 88vh; /* Augmente la hauteur du cadre blanc */
  padding: 25px 14px; 
  display: flex;
  flex-direction: column; /* Organise les éléments verticalement */
}

.sidebar-top-content {
  display: flex;
  flex-direction: column;
  gap: 0px; /* Réduit l'espace entre le titre et la nav */
}

/* Force le widget radio à remonter et supprime les marges de Streamlit */
div[data-testid="stRadio"] {
  margin-top: -15px !important; 
}

/* Supprime l'espace vide interne des étiquettes radio */
div[data-testid="stRadio"] > div[role="radiogroup"] {
  padding-top: 0px !important;
}

.sidebar-logo { 
  font-weight: 800; 
  font-size: 22px; 
  color: #122744;
  line-height: 1.2;
}

.sidebar-subtitle {
  margin-top: 2px;
  margin-bottom: 20px; /* Espace avant de commencer la liste radio */
  color: #6e839f;
  font-size: 11px;
  letter-spacing: 1px;
  text-transform: uppercase;
}

div[data-testid="stRadio"] > div[role="radiogroup"] label {
  border: 1px solid transparent;
  border-radius: 10px;
  background: #f3f7fc;
  margin-bottom: 8px;
  padding: 7px 8px;
}

div[data-testid="stRadio"] > div[role="radiogroup"] label:hover {
  border-color: #d0def0;
}

div[data-testid="stRadio"] > div[role="radiogroup"] input:checked + div {
  color: #1a73e8 !important;
  font-weight: 700 !important;
}

.sidebar-spacer { height: 50px; }
.version-badge {
  margin-top: 10px;
  border: 1px solid #d8e4f4;
  border-radius: 10px;
  padding: 10px;
  font-size: 12px;
  color: #4e647f;
  background: #f7faff;
}

.card {
  border: 1px solid #dde7f3;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(16,24,40,.05);
  background: rgba(255,255,255,.82);
  padding: 16px;
  margin-bottom: 12px;
}

.section-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.badge-agent {
  border-radius: 999px;
  background: #e9f3fe;
  color: #1a73e8;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 700;
}

.gradcam-placeholder {
  height: 180px;
  border-radius: 10px;
  border: 1px dashed #c8d7ea;
  display: grid;
  place-items: center;
  text-align: center;
  color: #7388a3;
  background: #f2f6fb;
  padding: 0 14px;
}

.zone-note {
  margin-top: 6px;
  color: #3f5672;
  font-size: 13px;
  font-weight: 600;
}

.pipeline-step {
  font-weight: 600;
  color: #244261;
  margin-top: 4px;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0% { opacity: .55; }
  50% { opacity: 1; }
  100% { opacity: .55; }
}

.info-card {
  border: 1px solid #d9e4f2;
  border-radius: 12px;
  background: #f4f7fb;
  color: #526884;
  padding: 12px;
  margin-bottom: 10px;
}

.result-header {
  border-radius: 10px;
  padding: 10px 12px;
  color: white;
  font-weight: 800;
  margin-bottom: 8px;
}
.result-header.positive { background: #E24B4A; }
.result-header.negative { background: #1D9E75; }

.chat-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 12px;
}
.chat-title-row h3 { margin: 0; }
.badge-memory {
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 11px;
  font-weight: 700;
  background: #ebf7ef;
  color: #1b8f69;
}
.chat-subtitle {
  color: #657b97;
  font-size: 13px;
  margin-bottom: 8px;
}

.chat-row { display: flex; margin-bottom: 8px; }
.chat-row.user { justify-content: flex-end; }
.chat-row.bot { justify-content: flex-start; align-items: flex-end; gap: 6px; }
.bot-avatar { font-size: 18px; line-height: 1; margin-bottom: 2px; }

.chat-message {
  max-width: 90%;
  padding: 8px 10px;
  font-size: 13px;
  line-height: 1.45;
}
.chat-message.user {
  background: #E8F0FE;
  border-radius: 12px 12px 2px 12px;
}
.chat-message.bot {
  background: #F1F3F4;
  border-radius: 2px 12px 12px 12px;
}

div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar { width: 7px; }
div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb {
  background: #c4d2e5;
  border-radius: 8px;
}

div[data-testid="stFileUploader"] section {
  border-radius: 12px;
  border: 1px dashed #b9cbe2;
  background: #f7fafe;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, #1a73e8, #1557b0) !important;
  color: white !important;
  border: none !important;
  height: 48px !important;
  border-radius: 12px !important;
  font-weight: 800 !important;
}

@media (max-width: 1100px) {
  .top-header, .top-left, .top-right, .top-tabs, .section-row, .chat-title-row { display: block; }
  .search-box { margin: 8px 0; min-width: auto; width: 100%; }
}
</style>
""",
        unsafe_allow_html=True,
    )


init_state()
render_css()
render_header()

col_gauche, col_centrale, col_droite = st.columns([1.05, 2, 2], gap="large")

with col_gauche:
    render_sidebar()

with col_centrale:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Visual Analysis")

    uploaded_file = st.file_uploader(
        "Upload X-ray",
        type=["jpg", "jpeg", "png"],
        help="Formats supportes: JPG, JPEG, PNG",
    )
    ensure_case_from_upload(uploaded_file)

    current_case_id = st.session_state.case_id or "#P2M-00000-X"
    st.caption(f"CASE ID: {current_case_id}")

    if uploaded_file and st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, width="stretch")
        st.caption(get_uploaded_metadata(uploaded_file, st.session_state.uploaded_image))
    else:
        st.info("Chargez une radiographie thoracique pour demarrer l'analyse.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    render_gradcam_section()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Donnees Cliniques Patient")
    symptoms = st.text_area(
        "Symptomes cliniques",
        placeholder="Ex: Fievre 39°C, toux grasse, douleur thoracique depuis 2 jours...",
        height=100,
    )
    age_col, sex_col = st.columns(2)
    with age_col:
        st.number_input("Age du patient", min_value=0, max_value=120, value=45, key="age")
    with sex_col:
        st.selectbox("Sexe", ["Non specifie", "Masculin", "Feminin"], key="sex")
    st.text_input(
        "Antecedents medicaux",
        placeholder="Ex: Diabete type 2, tabagisme...",
        key="antecedents",
    )
    vital_signs = st.multiselect(
        "Signes vitaux anormaux",
        ["Fievre > 38.5°C", "SpO2 < 95%", "FC > 100 bpm", "FR > 20/min", "TA elevee"],
    )

    launch = st.button(
        "Lancer l'Analyse IA",
        type="primary",
        width="stretch",
        disabled=uploaded_file is None,
    )
    if launch and uploaded_file is not None:
        run_pipeline(uploaded_file, symptoms.strip(), vital_signs)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.pipeline_status in {"running", "done"}:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        render_pipeline_details()
        st.markdown("</div>", unsafe_allow_html=True)

with col_droite:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    render_diagnostic_output()
    st.markdown("</div>", unsafe_allow_html=True)

    ensure_welcome_message()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    render_chatbot()
    st.markdown("</div>", unsafe_allow_html=True)
