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


def _apply_math(air_dir, raw_plates):
    corrected = [n if np.dot(n := np.array(p), air_dir) >= 0 else -n for p in raw_plates]
    summed = np.sum(corrected, axis=0)
    norm = np.linalg.norm(summed)
    approach = -(summed / norm) if norm > 1e-6 else np.array([0.0, 0.0, -1.0])
    return [0.0 if abs(v) < 1e-5 else round(float(v), 4) for v in approach]


def compute_direct(edge_text):
    """Compute approach vectors with pure Python math, no API needed."""
    edge_data = {}
    current_edge = current_point = air_dir = raw_plates = None

    for line in edge_text.splitlines():
        line = line.strip()
        if line.startswith("EDGE:"):
            if current_edge and air_dir is not None and raw_plates:
                edge_data.setdefault(current_edge, {})
                edge_data[current_edge]["start" if current_point == "START" else "end"] = _apply_math(air_dir, raw_plates)
            parts = line.split("|")
            current_edge = parts[0].replace("EDGE:", "").strip()
            current_point = parts[1].strip()
            air_dir = raw_plates = None
        elif line.startswith("Guiding_Air_Vector:"):
            air_dir = np.array(json.loads(line.split(":", 1)[1].strip()))
        elif line.startswith("Raw_Plates:"):
            raw_plates = json.loads(line.split(":", 1)[1].strip())

    # flush last block
    if current_edge and air_dir is not None and raw_plates:
        edge_data.setdefault(current_edge, {})
        edge_data[current_edge]["start" if current_point == "START" else "end"] = _apply_math(air_dir, raw_plates)

    return edge_data


def compute_via_gemini(edge_text, api_key):
    """
    Gemini parses the edge text and calls our compute_approach_vector tool
    for each edge point. We do the math — Gemini does the parsing.
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    results = {}

    def compute_approach_vector(
        edge_id: str,
        point: str,
        air_dir: list[float],
        raw_plates: list[list[float]],
    ) -> str:
        """
        Compute the torch approach vector for one weld edge point.

        Args:
            edge_id: Edge identifier e.g. "edge-30"
            point: "start" or "end"
            air_dir: Guiding air direction vector [x, y, z]
            raw_plates: List of plate normal vectors [[x,y,z], ...]
        """
        vec = _apply_math(np.array(air_dir), raw_plates)
        results.setdefault(edge_id, {})[point] = vec
        return f"Stored {edge_id}/{point}: {vec}"

    model = genai.GenerativeModel("gemini-2.0-flash", tools=[compute_approach_vector])
    chat = model.start_chat(enable_automatic_function_calling=True)
    chat.send_message(
        "Parse the EDGE data below. For every EDGE block call compute_approach_vector "
        "with the edge name, point type (start/end lowercase), Guiding_Air_Vector, and Raw_Plates.\n\n"
        + edge_text
    )

    return results
