import re
import json
from google import genai
from google.genai import types


GEMINI_PROMPT = """You are a geometric reasoning assistant for robotic welding.

CONTEXT:
A welding torch must approach a weld joint from outside the part and point into the joint to deposit the weld bead. Below is data extracted from a CAD mesh, one block per EDGE point (START or END of a weld seam):

  - Travel_Dir: unit vector tangent to the weld seam (direction the torch moves while welding).
  - Guiding_Air_Vector: unit vector pointing from the weld point AWAY from the part into open air. It tells you which side of the geometry is empty (where the torch lives).
  - Raw_Plates: surface normal vectors of the CAD faces (plates) that meet at the weld joint. They come straight from mesh triangle normals, so their sign is arbitrary — a plate normal may point either into the joint or away from it. There are typically 2 or 3 plates meeting at a joint.

YOUR TASK:
For every EDGE point, derive a single unit vector — the **torch approach vector** — that represents the ideal direction for the torch to approach the weld point. Geometrically it should:
  - Point INTO the joint (so the torch arrives at the weld from the open-air side).
  - Be the geometric bisector of the plates meeting at the joint, so the torch is symmetric with respect to all welded surfaces.
  - Be consistent with Guiding_Air_Vector (which side is open).

You must figure out the math yourself. Reason about the geometry, then use the code execution tool to compute the result. Do not invent inputs or skip data.

OUTPUT RULES:
  - Round components to 4 decimal places. If |v| < 1e-5, write exactly 0.0 (no -0.0).
  - Output ONLY the final JSON inside a ```json code block. No explanation, no commentary.
  - JSON structure: {"edge-ID": {"start": [x,y,z], "end": [x,y,z]}, ...}

EDGE DATA:
"""


def compute_via_gemini(edge_text, api_key):
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=GEMINI_PROMPT + edge_text,
        config=types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        ),
    )

    full_text = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            full_text += part.text

    match = re.search(r"```json\s*(.*?)\s*```", full_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON block found in Gemini response:\n{full_text}")

    return json.loads(match.group(1))
