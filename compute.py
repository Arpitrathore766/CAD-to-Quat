import re
import json
from google import genai
from google.genai import types


GEMINI_PROMPT = """You are given EDGE DATA below. Write and EXECUTE a Python script (using the code execution tool) that does the following for every EDGE block:

ALGORITHM (must be implemented in code, not done mentally):
1. Parse edge name and point type (START or END) from the EDGE line.
2. Parse Guiding_Air_Vector and Raw_Plates.
3. For each plate in Raw_Plates:
   - Compute dot product with Guiding_Air_Vector.
   - If dot < 0: multiply plate by -1 (flip it). Otherwise keep it.
4. Sum the 3 (possibly flipped) plates into one vector.
5. Normalize that vector, then negate it. This is the approach_vector.
6. Round each component to 4 decimal places. Set any component where abs(v) < 1e-5 to exactly 0.0.

OUTPUT RULES:
- Run the code. Do not compute by hand.
- After the code runs, output ONLY the resulting JSON inside a ```json block.
- No explanation, no commentary, nothing else.
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
