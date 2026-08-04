"""
ingest.py
RideReady knowledge-base builder.

Run this ONCE whenever you add or change a manual PDF:

    python ingest.py

What it does (the real RAG ingestion pipeline):
  1. LOAD   - reads every PDF in the manuals/ folder, page by page.
  2. CHUNK  - splits each page's text into ~800-char overlapping pieces,
              stamping each chunk with vehicle (from the filename), the
              real printed manual page number, and the section header.
  3. EMBED  - embeds every chunk once with text-embedding-3-small.
  4. SAVE   - writes chunks + vectors + metadata to manual_index/index.pkl,
              so the app loads the index instantly instead of re-embedding
              hundreds of pages on every launch.

Filename convention (IMPORTANT): name each PDF  YEAR_Make_Model.pdf
  e.g.  2023_Toyota_Camry.pdf   ->  year=2023, make=Toyota, model=Camry
The vehicle metadata is parsed from the filename, which is what keeps
retrieval scoped to the correct vehicle.
"""

import os
import pickle
import re

import numpy as np
import pypdf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
MANUALS_DIR = "manuals"
INDEX_DIR = "manual_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.pkl")

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 100   # characters of overlap between consecutive chunks
EMBED_BATCH = 100     # embed this many chunks per API call (faster + cheaper)


# ---------------------------------------------------------------------------
# CURATED warning-light chunks (2023 Toyota Camry).
#
# These are authored deliberately, one clean chunk per warning light, because
# the manual presents all lights in a single dense table that blind chunking
# splits awkwardly — hurting retrieval on the safety-critical "identify a
# warning light" use case. Each chunk combines:
#   • MEANING + ACTION: quoted/paraphrased from the manual (pp.528-532),
#     so the answer stays grounded and the page citation is real.
#   • SYMBOL description: the standardized ISO 2575 dashboard-symbol
#     appearance (e.g. "red circle with exclamation mark"), added so a
#     driver's visual description ("red exclamation mark") retrieves the
#     right light. (Documented as ISO-standard, not manual-sourced.)
# These are indexed ALONGSIDE the auto-extracted PDF chunks.
# ---------------------------------------------------------------------------
CAMRY = {"year": "2023", "make": "Toyota", "model": "Camry"}
ACCORD = {"year": "2022", "make": "Honda", "model": "Accord"}

def _wl(vehicle, section, page, symbol, color, meaning, action, severity):
    """Build one curated warning-light chunk for a given vehicle."""
    text = (f"Warning light: {section}. "
            f"Symbol: {symbol} Color: {color}. "
            f"Meaning: {meaning} "
            f"Action: {action} "
            f"Severity: {severity}")
    return {**vehicle, "section": section, "page": page, "text": text}

CURATED_CHUNKS = [
    _wl(CAMRY, "Brake System Warning Light", 528,
        "a red circle with an exclamation mark inside it (may be shown as a circle flanked by parentheses/brackets with a '!' in the middle), or the word BRAKE",
        "red",
        "The brake fluid level is low, or the brake system is malfunctioning.",
        "Immediately stop the vehicle in a safe place and contact your Toyota dealer. Continuing to drive the vehicle may be dangerous.",
        "Stop safely & check now"),
    _wl(CAMRY, "Tire Pressure Warning Light (TPMS)", 532,
        "a yellow/amber horseshoe or U-shaped tire cross-section with an exclamation mark in the center",
        "yellow/amber",
        "One or more tires is significantly under-inflated, due to a puncture, natural pressure loss, or a TPMS malfunction.",
        "Check the tire pressures and inflate to the level on the driver's doorjamb label as soon as possible. If a tire is punctured, see manual p.543.",
        "Caution"),
    _wl(CAMRY, "Malfunction Indicator Lamp (Check Engine)", 529,
        "an amber/yellow engine-block outline symbol",
        "amber/yellow",
        "A malfunction in the electronic engine control system, electronic throttle control system, or electronic automatic transmission control system.",
        "Have the vehicle inspected by your Toyota dealer immediately.",
        "Caution"),
    _wl(CAMRY, "SRS Airbag Warning Light", 529,
        "a red seated person symbol with a large circle (airbag) in front of the chest",
        "red",
        "A malfunction in the SRS airbag system, the front passenger occupant classification system, or the seat belt pretensioner system.",
        "Have the vehicle inspected by your Toyota dealer immediately.",
        "Caution"),
    _wl(CAMRY, "ABS Warning Light", 529,
        "an amber/yellow circle with the letters ABS inside it",
        "amber/yellow",
        "A malfunction in the ABS (Anti-lock Brake System) or the brake assist system.",
        "Have the vehicle inspected by your Toyota dealer immediately. If BOTH the ABS and brake system warning lights stay on, stop safely and contact your dealer immediately, as the vehicle may become unstable when braking.",
        "Caution"),
    _wl(CAMRY, "Electric Power Steering (EPS) Warning Light", 530,
        "a red or yellow steering wheel symbol with an exclamation mark beside it",
        "red or yellow",
        "A malfunction in the EPS (Electric Power Steering) system. Yellow means steering assist is restricted; red means assist is lost and the steering wheel becomes very heavy.",
        "Have the vehicle inspected by your Toyota dealer immediately.",
        "Caution"),
    _wl(CAMRY, "Pre-Collision System (PCS) Warning Light", 530,
        "an amber/yellow symbol of a car with a sensor/impact line in front, sometimes with an exclamation mark",
        "amber/yellow",
        "If a buzzer sounds, a malfunction has occurred in the PCS (Pre-Collision System). If no buzzer, the PCS is temporarily unavailable.",
        "If a buzzer sounds, have the vehicle inspected by your Toyota dealer immediately. Otherwise, follow the multi-information display instructions.",
        "Caution"),
    _wl(CAMRY, "Slip Indicator (VSC/TRAC)", 531,
        "a car symbol with two curved skid lines beneath it",
        "amber/yellow",
        "A malfunction in the VSC (Vehicle Stability Control), TRAC (Traction Control), or ABS. The light flashes while VSC or TRAC is actively operating.",
        "If it flashes during driving, the system is operating normally. If it stays on, have the vehicle inspected by your Toyota dealer immediately.",
        "Caution"),
    _wl(CAMRY, "Low Fuel Level Warning Light", 531,
        "an amber/yellow fuel-pump symbol",
        "amber/yellow",
        "The remaining fuel level is low.",
        "Refuel as soon as possible.",
        "Safe to drive"),
    _wl(CAMRY, "High Coolant Temperature Warning Light", 528,
        "a red thermometer symbol sitting in wavy liquid lines",
        "red",
        "The engine coolant temperature is too high.",
        "Immediately stop the vehicle in a safe place. See manual p.567 for handling method.",
        "Stop safely & check now"),
    _wl(CAMRY, "Charging System Warning Light", 528,
        "a red battery symbol with plus and minus terminals",
        "red",
        "A malfunction in the vehicle's charging system.",
        "Immediately stop the vehicle in a safe place and contact your Toyota dealer.",
        "Stop safely & check now"),
    _wl(CAMRY, "Low Engine Oil Pressure Warning Light", 528,
        "a red oil-can symbol with a drip",
        "red",
        "The engine oil pressure is too low.",
        "Immediately stop the vehicle in a safe place and contact your Toyota dealer.",
        "Stop safely & check now"),
    _wl(CAMRY, "Master Warning Light", 538,
        "a red or amber triangle with an exclamation mark inside",
        "red or amber",
        "A general master warning that a message is being shown on the multi-information display; it accompanies various system warnings.",
        "Follow the instructions in the message on the multi-information display. If the message reappears after corrective action, contact your Toyota dealer.",
        "Caution"),

    # --- 2022 Honda Accord warning lights (meaning/action from the Accord
    #     Owner's Manual indicator section pp.84-98; symbols are ISO-standard) ---
    _wl(ACCORD, "Brake System Indicator (Red)", 85,
        "a red circle with an exclamation mark inside it (may be shown as a circle with '(!)' or the word BRAKE)",
        "red",
        "Comes on when the parking brake is applied; when the brake fluid level is low; or if there is a problem with the brake system.",
        "If it stays on after releasing the parking brake, the brake fluid may be low or the brake system faulty. Stop in a safe place and have the vehicle checked by a dealer immediately.",
        "Stop safely & check now"),
    _wl(ACCORD, "Parking Brake and Brake System Indicator (Amber)", 86,
        "an amber circle with an exclamation mark, or brackets around a '!'",
        "amber",
        "Comes on if there is a problem with a braking-related system other than the conventional brake system (e.g. electric parking brake or automatic brake hold).",
        "Avoid high speeds and sudden braking, and take the vehicle to a dealer immediately.",
        "Caution"),
    _wl(ACCORD, "Anti-lock Brake System (ABS) Indicator", 95,
        "an amber circle with the letters ABS inside it",
        "amber",
        "Comes on if there is a problem with the ABS. The vehicle still has normal braking ability but no anti-lock function.",
        "If it stays on constantly, have the vehicle checked by a dealer.",
        "Caution"),
    _wl(ACCORD, "Supplemental Restraint System (SRS) Indicator", 95,
        "a red seated person symbol with a circle (airbag) in front of the chest",
        "red",
        "Comes on if a problem is detected with the supplemental restraint (airbag) system, knee/side/side-curtain airbags, or the seat belt tensioner.",
        "If it stays on constantly, have the vehicle checked by a dealer.",
        "Caution"),
    _wl(ACCORD, "Low Tire Pressure / TPMS Indicator", 98,
        "an amber horseshoe / U-shaped tire cross-section with an exclamation mark in the center",
        "amber",
        "Comes on and stays on when one or more tire pressures are significantly low, or the system needs calibration. Blinks then stays on if there is a TPMS problem or a compact spare is installed.",
        "If it comes on while driving, stop in a safe place, check tire pressures, and inflate as needed. If it blinks and remains on, have the vehicle checked by a dealer.",
        "Caution"),
    _wl(ACCORD, "Electric Power Steering (EPS) System Indicator", 97,
        "a red or amber steering wheel symbol with an exclamation mark beside it",
        "red or amber",
        "Comes on if there is a problem with the EPS (Electric Power Steering) system.",
        "If it stays on constantly, have the vehicle checked by a dealer.",
        "Caution"),
    _wl(ACCORD, "Charging System Indicator", 90,
        "a red battery symbol with plus and minus terminals",
        "red",
        "Comes on when there is a problem with the charging system.",
        "Stop in a safe place and contact a dealer immediately.",
        "Stop safely & check now"),
    _wl(ACCORD, "Malfunction Indicator Lamp (Check Engine)", 89,
        "an amber engine-block outline symbol",
        "amber",
        "Comes on if there is a problem with the emissions control systems. Blinks when an engine cylinder misfire is detected.",
        "Have the vehicle checked by a dealer. If it is blinking, reduce speed and have it checked as soon as possible.",
        "Caution"),
    _wl(ACCORD, "Low Fuel Indicator", 95,
        "an amber fuel-pump symbol",
        "amber",
        "Comes on when the fuel reserve is low (about 2.2 U.S. gal / 8.4 L remaining). Blinks if there is a problem with the fuel gauge.",
        "Refuel as soon as possible. If it is blinking, have the vehicle checked by a dealer.",
        "Safe to drive"),
    _wl(ACCORD, "Vehicle Stability Assist (VSA) OFF Indicator", 97,
        "an amber car symbol with skid marks and the word OFF, or a triangle with a car and skid lines",
        "amber",
        "Comes on when VSA is partially disabled, or if VSA was deactivated temporarily after the battery was disconnected and reconnected.",
        "Drive a short distance above 12 mph (20 km/h); the indicator should go off. If it does not, have the vehicle checked by a dealer.",
        "Caution"),
]


def parse_filename(fname):
    """'2023_Toyota_Camry.pdf' -> ('2023', 'Toyota', 'Camry')."""
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Filename '{fname}' must be YEAR_Make_Model.pdf "
            f"(e.g. 2023_Toyota_Camry.pdf)."
        )
    year, make = parts[0], parts[1]
    model = " ".join(parts[2:])  # handles models like 'Model 3' -> Model_3
    return year, make, model


def parse_page(txt, pdf_index):
    """
    Pull the printed page number and section header from a page's text.

    Different manufacturers format page headers differently, so this tries
    several known patterns in order (layout-adaptive), then falls back to the
    PDF index + 1 when none match:

      Toyota:  '312 4-5. Using the driving support systems ...'
               -> page 312, section '4-5. Using the driving support systems'
      Honda:   '99 uuIndicatorsu Continued Instrument Panel ...'
               '448 uuWhen DrivinguTire Pressure ...'
               'Continued 499 uuHonda Sensing®u...'
               -> page 99/448/499, section 'Indicators' / 'When Driving' / ...
    """
    flat = " ".join(txt.split())

    # --- Toyota-style: "<page> <chapter-section>. <title> ..."
    m = re.match(r"\s*(\d{1,4})\s+(\d+-\d+\.?\s+[^|]+?)(?:CAMRY_U|\||[A-Z]{4,}_U)", flat)
    if m:
        return int(m.group(1)), m.group(2).strip()

    # --- Honda-style: optional "Continued", a page number, then "uuSectionu..."
    m = re.match(r"\s*(?:Continued\s+)?(\d{1,4})\s+uu([^u]+?)u", flat)
    if m:
        return int(m.group(1)), m.group(2).strip()

    # --- Honda variant: section first, page number later e.g. "uuParking...u 526 Driving"
    m = re.match(r"\s*uu([^u]+?)u.*?\s(\d{2,4})\s", flat)
    if m:
        return int(m.group(2)), m.group(1).strip()

    return pdf_index + 1, "General"


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into fixed-size overlapping chunks. Layout-agnostic."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks


def embed_batch(texts):
    """Embed a list of texts in one API call; returns list of np arrays."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(d.embedding) for d in resp.data]


def build_index():
    if not os.path.isdir(MANUALS_DIR):
        raise SystemExit(f"No '{MANUALS_DIR}/' folder found. Create it and add PDFs.")

    pdfs = sorted(f for f in os.listdir(MANUALS_DIR) if f.lower().endswith(".pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in '{MANUALS_DIR}/'. Add YEAR_Make_Model.pdf files.")

    all_chunks = []  # each: {year, make, model, section, page, text}

    for fname in pdfs:
        year, make, model = parse_filename(fname)
        path = os.path.join(MANUALS_DIR, fname)
        reader = pypdf.PdfReader(path)
        print(f"Loading {fname}  ({len(reader.pages)} pages)  ->  {year} {make} {model}")

        for pdf_index, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            if not txt.strip():
                continue
            printed_page, section = parse_page(txt, pdf_index)
            for piece in chunk_text(txt):
                all_chunks.append({
                    "year": year, "make": make, "model": model,
                    "section": section, "page": printed_page, "text": piece,
                })

    # Add the deliberately-authored warning-light chunks alongside the
    # auto-extracted PDF chunks (see CURATED_CHUNKS note above).
    all_chunks.extend(CURATED_CHUNKS)
    print(f"  + {len(CURATED_CHUNKS)} curated warning-light chunks")

    print(f"\nTotal chunks: {len(all_chunks)}. Embedding with {EMBED_MODEL} ...")

    vectors = []
    for start in range(0, len(all_chunks), EMBED_BATCH):
        batch = [c["text"] for c in all_chunks[start:start + EMBED_BATCH]]
        vectors.extend(embed_batch(batch))
        print(f"  embedded {min(start + EMBED_BATCH, len(all_chunks))}/{len(all_chunks)}")

    os.makedirs(INDEX_DIR, exist_ok=True)
    import gzip
    with gzip.open(INDEX_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "vectors": np.array(vectors)}, f)

    print(f"\nSaved index -> {INDEX_PATH}")
    print(f"  {len(all_chunks)} chunks from {len(pdfs)} manual(s).")
    vehicles = sorted({f'{c["year"]} {c["make"]} {c["model"]}' for c in all_chunks})
    print("  Vehicles indexed:", ", ".join(vehicles))


if __name__ == "__main__":
    build_index()