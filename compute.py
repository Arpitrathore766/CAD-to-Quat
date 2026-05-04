import re
import json
import numpy as np


GEMINI_PROMPT = """You will be provided with a block of text containing EDGE data. Each point has a Guiding_Air_Vector (the physical open space) and 3 Raw_Plates (CAD normals). You must apply the following exact mathematical logic to align the plates and calculate the torch approach vector:

Alignment (Dot Product): For each of the 3 Raw_Plates, calculate the dot product against the Guiding_Air_Vector.
- If dot_product < 0, the plate is facing inward. Multiply it by -1 (flip it).
- If dot_product >= 0, keep the plate as is.
Store these 3 vectors as corrected_normals.

Bisection: Sum the 3 corrected_normals together.

Inward Approach: Normalize the summed vector, and then multiply it by -1 so that it points INWARD toward the weld joint. This is the approach_vector.

Clean up: Round all output floats to 4 decimal places. If a value is effectively zero (e.g., < 1e-5), force it to 0.0 to avoid -0.0 in the JSON.

Strict Execution Rules:
- You MUST write and execute a Python script to parse the text and perform this math. Do not attempt to calculate these floating-point vectors manually.
- Output ONLY the final JSON inside a standard ```json code block. Do not include introductory or concluding conversational text.

Desired JSON Structure: {"edge-ID": {"start": [x,y,z], "end": [x,y,z]}, "edge-ID2": ...}

EDGE DATA:
"""


def compute_direct(edge_text):
    """Compute approach vectors with pure Python math, no API needed."""
    edge_data = {}

    pattern = re.compile(
        r"EDGE: (\S+) \| (START|END)\s+"
        r"Travel_Dir: (\[.*?\])\s+"
        r"Guiding_Air_Vector: (\[.*?\])\s+"
        r"Raw_Plates: (\[.*?\])",
        re.DOTALL,
    )

    for m in pattern.finditer(edge_text):
        edge_name = m.group(1)
        point_type = m.group(2)
        air_dir = np.array(json.loads(m.group(4)))
        raw_plates = json.loads(m.group(5))

        if not raw_plates:
            continue

        corrected = []
        for plate in raw_plates:
            n = np.array(plate)
            corrected.append(n if np.dot(n, air_dir) >= 0 else -n)

        summed = np.sum(corrected, axis=0)
        norm = np.linalg.norm(summed)
        approach = -(summed / norm) if norm > 1e-6 else np.array([0.0, 0.0, -1.0])

        vec = [0.0 if abs(v) < 1e-5 else round(float(v), 4) for v in approach]

        if edge_name not in edge_data:
            edge_data[edge_name] = {}
        edge_data[edge_name]["start" if point_type == "START" else "end"] = vec

    return edge_data


def compute_via_gemini(edge_text, api_key):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash", tools="code_execution")
    response = model.generate_content(GEMINI_PROMPT + edge_text)

    # Extract JSON from the response text
    full_text = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text"):
            full_text += part.text

    match = re.search(r"```json\s*(.*?)\s*```", full_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON block found in Gemini response:\n{full_text}")

    return json.loads(match.group(1))
