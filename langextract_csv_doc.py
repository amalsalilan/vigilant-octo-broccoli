from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
import os, json, textwrap, re
import langextract as lx
import pandas as pd

app = FastAPI(title="LangExtract (Ollama) Adapter")

# --- Ollama / Model config via env ---
MODEL_URL = os.getenv("OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
MODEL_ID = os.getenv("OLLAMA_MODEL_ID", "gemma3:4b")
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))  # 0 for extraction

# Save artifacts toggle (JSONL + HTML) — ON by default
SAVE_ARTIFACTS = os.getenv("SAVE_ARTIFACTS", "1") == "1"
# Optional base URL if you're fronting this service behind a reverse proxy / domain
# Example: PUBLIC_BASE_URL="https://extractor.mycorp.com"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")

# ---- Instruction CSV path (Windows) ----
INSTRUCTION_CSV_PATH = os.getenv(
    "INSTRUCTION_CSV_PATH",
    r"C:\Users\thefl\Documents\langextract\instruction_sample.csv"
)

# --- LangExtract model config ---
MODEL_CONFIG = lx.factory.ModelConfig(
    model_id=MODEL_ID,
    provider_kwargs={
        "model_url": MODEL_URL,
        "format_type": lx.data.FormatType.JSON,
        "temperature": TEMPERATURE,
        "timeout": 3000,
    },
)

# -------------------------
# FALLBACK PROMPT + EXAMPLES (used if column_name not found)
# -------------------------
FALLBACK_PROMPT = textwrap.dedent("""\
    Extract key entities from bulk telecom and property service agreements.
    Use exact spans from the text (no paraphrasing or normalization).
    Only extract from this set of classes: party, property_name, address_line, city, state, zip, total_units,
    effective_date, term_length, renewal_term, non_renewal_notice, construction_status, days_to_construct,
    go_live_date, service_type, revenue_model, revenue_share_percent, billing_frequency, arpu,
    capex_responsibility, capex_terms, antenna_rooftop_rights, assignment_clause, early_termination_fee,
    governing_law, contact_name, contact_email, contact_phone.
    Return entities in the order they appear in the text.
    Each extraction_text must be one contiguous substring copied exactly from the source, including symbols like $ or %.
    Do not overlap spans or merge multiple facts.
    Attributes must use only controlled values:
      - party.role ∈ {Provider, PropertyOwner, PropertyManager, Subsidiary}
      - service_type.category ∈ {Internet, TV, WiFi, Other}
      - revenue_model.type ∈ {Bulk, RevenueShare, Mixed}
      - capex_responsibility.party ∈ {Provider, Owner, Shared}
    Omit an attribute if it is not explicitly present in the text.
    If a fact is absent in the text, omit its object entirely.
    Return JSON only — no commentary, no markdown.
""")

FALLBACK_EXAMPLES = [
    lx.data.ExampleData(
        text='This Bulk Internet Agreement is entered into on March 1, 2024 between GigaMonster, LLC ("Provider") and ZRS Management ("Property Owner") for the community Ablon at Harbor Village located at 2600 Lakefront Trail, Rockwall, Texas 75032. Bulk rate is $31.35 per unit for Ubiquitous WiFi service.',
        extractions=[
            lx.data.Extraction("party", "GigaMonster, LLC", attributes={"role": "Provider"}),
            lx.data.Extraction("party", "ZRS Management", attributes={"role": "PropertyOwner"}),
            lx.data.Extraction("property_name", "Ablon at Harbor Village"),
            lx.data.Extraction("address_line", "2600 Lakefront Trail"),
            lx.data.Extraction("city", "Rockwall"),
            lx.data.Extraction("state", "Texas"),
            lx.data.Extraction("zip", "75032"),
            lx.data.Extraction("effective_date", "March 1, 2024"),
            lx.data.Extraction("service_type", "Ubiquitous WiFi", attributes={"category": "WiFi"}),
            lx.data.Extraction("arpu", "$31.35"),
        ],
    ),
    lx.data.ExampleData(
        text="The initial term is three (3) years with automatic renewals of one (1) year each. Provider will bill quarterly. Property Owner shall receive fifteen percent (15%) of Net Service Revenues.",
        extractions=[
            lx.data.Extraction("term_length", "three (3) years"),
            lx.data.Extraction("renewal_term", "one (1) year"),
            lx.data.Extraction("billing_frequency", "bill quarterly"),
            lx.data.Extraction("revenue_share_percent", "fifteen percent (15%)"),
            lx.data.Extraction("revenue_model", "Revenue Share", attributes={"type": "RevenueShare"}),
        ],
    ),
    lx.data.ExampleData(
        text="Construction status: In Progress; estimated 45 days to construct with a go-live date of May 15, 2024. CAPEX is Shared: 25% upfront and 75% at launch.",
        extractions=[
            lx.data.Extraction("construction_status", "In Progress"),
            lx.data.Extraction("days_to_construct", "45 days"),
            lx.data.Extraction("go_live_date", "May 15, 2024"),
            lx.data.Extraction("capex_responsibility", "Shared", attributes={"party": "Shared"}),
            lx.data.Extraction("capex_terms", "25% upfront and 75% at launch"),
        ],
    ),
    lx.data.ExampleData(
        text="Either party may terminate upon thirty (30) days’ written notice for material breach. Assignment is permitted without consent in connection with a merger. Provider may place rooftop antennas as required. This Agreement is governed by the laws of the State of California.",
        extractions=[
            lx.data.Extraction("termination_clause", "terminate upon thirty (30) days’ written notice for material breach"),
            lx.data.Extraction("assignment_clause", "Assignment is permitted without consent in connection with a merger"),
            lx.data.Extraction("antenna_rooftop_rights", "Provider may place rooftop antennas as required"),
            lx.data.Extraction("governing_law", "the laws of the State of California"),
        ],
    ),
    lx.data.ExampleData(
        text="Owner contact: Jane Smith, jane.smith@example.com, (555) 123-4567.",
        extractions=[
            lx.data.Extraction("contact_name", "Jane Smith"),
            lx.data.Extraction("contact_email", "jane.smith@example.com"),
            lx.data.Extraction("contact_phone", "(555) 123-4567"),
        ],
    ),
    lx.data.ExampleData(text="The parties intend to negotiate a separate addendum in the future.", extractions=[]),
    lx.data.ExampleData(
        text="Bulk rate is USD $31.35 per unit; revenue share equals fifteen percent (15%). Non-renewal notice: ninety (90) days.",
        extractions=[
            lx.data.Extraction("arpu", "USD $31.35"),
            lx.data.Extraction("revenue_share_percent", "fifteen percent (15%)"),
            lx.data.Extraction("non_renewal_notice", "ninety (90) days"),
        ],
    ),
]

# -------------------------
# PER-FIELD PROMPT TEMPLATE
# -------------------------
PROMPT_TPL = textwrap.dedent("""\
    Extract ONLY the following field from the input text:
    {what}

    Rules:
    - Copy spans EXACTLY from input (no paraphrasing, keep symbols like $ or %).
    - Return JSON only, shaped like:
      {{ "extractions": [ {{ "extraction_class": "<class>", "extraction_text": "<span>", "attributes": {{...}} }} ] }}
    - Omit any object you cannot copy verbatim from the input.
    - No markdown, no extra commentary.
""")

# -------------------------
# Instruction CSV loader/cache
# -------------------------
_INSTR_CACHE = None

def _parse_examples(raw: str):
    s = (raw or "").strip()
    if not s:
        return []
    if s.startswith("[") and "lx.data.ExampleData" in s:
        scope = {"lx": lx}
        try:
            return eval(s, {"__builtins__": {}}, scope)  # trusted CSV only
        except Exception as e:
            raise ValueError(f"Failed to eval examples: {e}")
    try:
        arr = json.loads(s)
        out = []
        for ex in arr:
            exs = []
            for e in ex.get("extractions", []):
                exs.append(
                    lx.data.Extraction(
                        extraction_class=e.get("class") or e.get("extraction_class"),
                        extraction_text=e.get("span") or e.get("extraction_text"),
                        attributes=e.get("attributes") or {},
                    )
                )
            out.append(lx.data.ExampleData(text=ex["text"], extractions=exs))
        return out
    except Exception as e:
        raise ValueError(f"Failed to parse examples JSON: {e}")

def _load_instruction_csv(path: str):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    required_cols = {"column_name", "what_to_extract", "examples"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Instruction CSV missing columns: {missing}")
    instr = {}
    for _, r in df.iterrows():
        col = str(r["column_name"]).strip().lstrip("=")
        what = str(r["what_to_extract"]).strip()
        examples_raw = str(r["examples"])
        examples = _parse_examples(examples_raw)
        instr[col] = {"what_to_extract": what, "examples": examples}
    return instr

def _get_instruction():
    global _INSTR_CACHE
    if _INSTR_CACHE is None:
        _INSTR_CACHE = _load_instruction_csv(INSTRUCTION_CSV_PATH)
    return _INSTR_CACHE

# -------------------------
# Artifact helpers
# -------------------------
def _safe_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]+', '_', (name or '').strip()).strip('_')[:120]

def _ensure_dirs():
    os.makedirs("outputs/json", exist_ok=True)
    os.makedirs("outputs/html", exist_ok=True)

def _artifact_paths(field_name: str, doc_id: Optional[str]):
    base = f"{_safe_name(doc_id)}__{_safe_name(field_name)}" if doc_id else _safe_name(field_name)
    return {
        "base_name": base,
        "jsonl_path": os.path.join("outputs", "json", f"{base}.jsonl"),
        "html_path":  os.path.join("outputs", "html", f"{base}.html"),
        "relative_html_path": f"artifacts/{base}.html",
    }

def _artifact_urls(base_name: str, request: Optional[Request] = None):
    """
    Best-effort public URL for the HTML report.
    Prefers PUBLIC_BASE_URL if provided; otherwise uses request.base_url.
    """
    if not base_name:
        return {}
    rel = f"artifacts/{base_name}.html"
    if PUBLIC_BASE_URL:
        return {"html_url": f"{PUBLIC_BASE_URL.rstrip('/')}/{rel}"}
    if request is not None:
        return {"html_url": str(request.base_url) + rel}
    return {}

def _save_artifacts(field_name: str, doc_id: Optional[str], result_obj: dict, source_text: Optional[str] = None):
    """Save JSONL (+append) and HTML visualization. Filenames include doc_id."""
    if not SAVE_ARTIFACTS:
        return
    try:
        _ensure_dirs()
        paths = _artifact_paths(field_name, doc_id)
        base = paths["base_name"]
        path_json = paths["jsonl_path"]
        path_html = paths["html_path"]

        # Try LangExtract saver first (expects annotated docs format)
        try:
            lx.io.save_annotated_documents([result_obj], output_name=f"{base}.jsonl", output_dir="outputs/json")
        except Exception:
            # Fallback: append one line of JSON with some helpful context
            line = {"doc_id": doc_id, "source_text": source_text, "result": result_obj}
            with open(path_json, "a", encoding="utf-8", errors="replace") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        # HTML visualize (best-effort)
        try:
            html_content = lx.visualize(path_json)
            html_str = getattr(html_content, "data", html_content)
            with open(path_html, "w", encoding="utf-8", errors="replace") as f:
                f.write(html_str)
        except Exception:
            pass
    except Exception:
        # Never fail API due to artifact I/O
        pass

# Mount /artifacts to serve outputs/html
_ensure_dirs()
try:
    app.mount("/artifacts", StaticFiles(directory="outputs/html"), name="artifacts")
except Exception:
    # If directory missing or running in restricted env, ignore
    pass

# --- helper: normalize Ramp-Up into fixed keys (returned in meta.normalized) ---
def _normalize_ramp_up(field_name: str, doc_id: str, result_obj: dict, source_text: str, request: Optional[Request] = None) -> dict:
    """
    Produce:
    {
      "ramp_up_applicable": "yes/no",
      "%_rate": <int or "">,
      "tenure_in_months": <int or "">,
      "extraction_report_location": "outputs/html/<doc>__<field>.html",
      "extraction_report_url": "<public or local url>"
    }
    """
    paths = _artifact_paths(field_name, doc_id)
    urls = _artifact_urls(paths["base_name"], request)

    out = {
        "ramp_up_applicable": "",
        "%_rate": "",
        "tenure_in_months": "",
        "extraction_report_location": paths["html_path"],
        "extraction_report_url": urls.get("html_url", ""),
    }

    ex_list = (result_obj or {}).get("extractions") or []
    attrs = (ex_list[0].get("attributes", {}) if ex_list else {}) or {}

    def _yesno(v):
        s = str(v or "").strip().lower()
        if s in ("yes", "y", "true", "applicable"):
            return "yes"
        if s in ("no", "n", "false", "not applicable", "na"):
            return "no"
        return ""

    def _to_int(v):
        try:
            if isinstance(v, (int, float)):
                return int(float(v))
            s = str(v or "").replace("%", "").strip()
            return int(float(s)) if s else ""
        except Exception:
            return ""

    # Prefer model attributes
    out["ramp_up_applicable"] = _yesno(attrs.get("ramp_up_applicable"))
    out["%_rate"]            = _to_int(attrs.get("%_rate"))
    out["tenure_in_months"]  = _to_int(attrs.get("tenure_in_months"))

    # Fallbacks from text if attributes missing
    blob = " ".join([ex_list[0].get("extraction_text", "")] + [source_text or ""]).lower()
    if out["%_rate"] == "":
        m = re.search(r'(\d+(?:\.\d+)?)\s*%', blob)
        if m:
            out["%_rate"] = _to_int(m.group(1))

    if out["tenure_in_months"] == "":
        m = re.search(r'(\d+)\s*(months|month|mos|mths)\b', blob)
        if m:
            out["tenure_in_months"] = _to_int(m.group(1))

    if out["ramp_up_applicable"] == "":
        if re.search(r'\bno\b|not applicable|does not apply', blob):
            out["ramp_up_applicable"] = "no"
        elif out["%_rate"] != "" or re.search(r'\bramp[-\s]?up\b|\bapplies\b', blob):
            out["ramp_up_applicable"] = "yes"

    return out

# -------------------------
# API models / endpoints
# -------------------------
class ExtractReq(BaseModel):
    text: str
    column_name: Optional[str] = None
    record_id: Optional[str] = None
    doc_id: Optional[str] = None  # NEW: allow tying artifacts to a doc_id

class RowExtractReq(BaseModel):
    doc_id: str
    row: Dict[str, str]
    only_fields: Optional[List[str]] = None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "ollama",
        "model_id": MODEL_ID,
        "model_url": MODEL_URL,
        "temperature": TEMPERATURE,
        "instruction_csv": INSTRUCTION_CSV_PATH,
        "instruction_loaded": bool(_INSTR_CACHE),
        "save_artifacts": SAVE_ARTIFACTS,
        "public_base_url": PUBLIC_BASE_URL,
        "artifacts_mount": "/artifacts",
    }

@app.post("/reload-instruction")
def reload_instruction():
    global _INSTR_CACHE
    _INSTR_CACHE = _load_instruction_csv(INSTRUCTION_CSV_PATH)
    return {"ok": True, "loaded_keys": list(_INSTR_CACHE.keys())}

# --- helper: run extraction for one field using the same logic as /extract-text ---
def _extract_for_column(column_name: str, text: str):
    text = (text or "").strip()
    text = text.replace("\u2011", "-")
    if text.startswith("="):
        text = text[1:].lstrip()
    if len(text) < 10:
        return {"ok": False, "error": "Text too short", "result": None}

    prompt = FALLBACK_PROMPT
    examples = FALLBACK_EXAMPLES
    used_fallback = True

    if column_name:
        instr = _get_instruction()
        lookup_key = str(column_name or "").strip().lstrip("=")
        row = instr.get(lookup_key)
        if row:
            prompt = PROMPT_TPL.format(what=row["what_to_extract"])
            examples = row["examples"]
            used_fallback = False

    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        config=MODEL_CONFIG,
        use_schema_constraints=True,
        extraction_passes=3,  # higher passes retained
        max_char_buffer=10000,
        max_workers=20,       # higher workers retained
    )

    return {
        "ok": True,
        "result": result,
        "meta": {"column_name": column_name, "used_fallback": used_fallback},
    }

@app.post("/extract-text")
def extract_text(req: ExtractReq, request: Request):
    text = (req.text or "").strip()
    text = text.replace("\u2011", "-")
    if text.startswith("="):
        text = text[1:].lstrip()
    if len(text) < 10:
        raise HTTPException(400, "Text too short; provide a paragraph.")

    prompt = FALLBACK_PROMPT
    examples = FALLBACK_EXAMPLES
    used_fallback = True

    if req.column_name:
        instr = _get_instruction()
        lookup_key = str(req.column_name or "").strip().lstrip("=")
        row = instr.get(lookup_key)
        if row:
            prompt = PROMPT_TPL.format(what=row["what_to_extract"])
            examples = row["examples"]
            used_fallback = False

    try:
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            config=MODEL_CONFIG,
            use_schema_constraints=True,
            extraction_passes=1,
            max_char_buffer=1000,
            max_workers=1,
        )

        artifact_info = None
        if SAVE_ARTIFACTS:
            field = req.column_name or "extract"
            _save_artifacts(field_name=field, doc_id=req.doc_id, result_obj=result, source_text=text)
            paths = _artifact_paths(field, req.doc_id)
            urls = _artifact_urls(paths["base_name"], request)
            artifact_info = {**paths, **urls}

        return {
            "ok": True,
            "result": result,
            "meta": {
                "column_name": req.column_name,
                "record_id": req.record_id,
                "doc_id": req.doc_id,
                "used_fallback": used_fallback,
                "artifact": artifact_info,  # single-field convenience
            },
        }
    except Exception as e:
        raise HTTPException(500, f"LangExtract failed: {e}")

@app.post("/extract-row")
def extract_row(req: RowExtractReq, request: Request):
    """
    Document-centric extraction:
      - Accepts a single document row with its doc_id.
      - Processes only fields present in BOTH the instruction set and the row
        (or restricts to req.only_fields if provided).
      - Returns one JSON object keyed by doc_id with per-field LangExtract results.
      - When SAVE_ARTIFACTS=1 (default), writes {DOCID}__{FIELD}.jsonl/.html per field.
      - Adds artifact info in meta.artifacts[<FIELD>] and ramp-up normalization in meta.normalized[<FIELD>].
    """
    if not req.doc_id:
        raise HTTPException(400, "doc_id is required")

    instr = _get_instruction()
    instr_keys = set(instr.keys())
    row_keys = set(req.row.keys())

    if req.only_fields:
        target_fields = [f for f in req.only_fields if f in row_keys and f in instr_keys]
    else:
        target_fields = [f for f in row_keys if f in instr_keys]

    extractions: Dict[str, dict] = {}
    errors: Dict[str, str] = {}
    artifacts_index: Dict[str, dict] = {}
    normalized_index: Dict[str, dict] = {}

    for field in target_fields:
        text = (req.row.get(field) or "").strip()
        text = text.replace("\u2011", "-")
        if text.startswith("="):
            text = text[1:].lstrip()

        if len(text) < 10:
            errors[field] = "Text too short"
            continue

        try:
            one = _extract_for_column(field, text)
            if one.get("ok"):
                res = one["result"]
                extractions[field] = res  # keep original object/shape

                # Save artifacts and index them (without mutating res)
                if SAVE_ARTIFACTS:
                    _save_artifacts(field_name=field, doc_id=req.doc_id, result_obj=res, source_text=text)
                    paths = _artifact_paths(field, req.doc_id)
                    urls = _artifact_urls(paths["base_name"], request)
                    artifacts_index[field] = {**paths, **urls}

                # Normalized object ONLY for Ramp Up Period (Y/N) & %
                if field.strip().lower().startswith("ramp up period"):
                    try:
                        norm = _normalize_ramp_up(
                            field_name=field,
                            doc_id=req.doc_id,
                            result_obj=res,
                            source_text=text,
                            request=request
                        )
                        normalized_index[field] = norm
                    except Exception:
                        pass
            else:
                errors[field] = one.get("error", "Unknown error")
        except Exception as e:
            errors[field] = f"{e}"

    return {
        "ok": True,
        "doc_id": req.doc_id,
        "extractions": extractions,
        "errors": errors,
        "meta": {
            "fields_processed": list(extractions.keys()),
            "fields_skipped": list(errors.keys()),
            "artifacts": artifacts_index,    # <-- per-field artifact info here
            "normalized": normalized_index,  # <-- per-field normalized (ramp-up) here
        },
    }
