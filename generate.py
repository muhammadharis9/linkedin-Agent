import os
import json
import re
from datetime import datetime
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# ---------- Setup ----------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("ERROR: GEMINI_API_KEY not found in .env")

genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.5-flash"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- Load inputs ----------
def read_file(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"ERROR: {path} not found")
    return p.read_text(encoding="utf-8").strip()

past_posts = read_file("posts.md")
style_rules = read_file("style_rules.txt")
notes = read_file("notes.txt")

if not notes:
    raise SystemExit("ERROR: notes.txt is empty. Add your rough notes first.")

# ---------- System prompt ----------
SYSTEM_PROMPT = f"""You write LinkedIn posts in the exact voice of the author (Haris).

Study the past posts carefully. Match the rhythm, structure, vocabulary, 
and signature patterns precisely. The output must read as if Haris wrote it.

STYLE RULES (non-negotiable):
{style_rules}

PAST POSTS (voice reference):
---
{past_posts}
---

Output ONLY the post text. No preamble like "Here is your post:". 
No explanation after. No markdown formatting unless the past posts use it.
"""

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=SYSTEM_PROMPT,
    generation_config={"temperature": 1.0, "max_output_tokens": 2000},
)

# ---------- Generation functions ----------
def generate_post(notes_text, part_info=None):
    """Generate a single post. If part_info given, generate as part of a series."""
    if part_info:
        user_msg = (
            f"Write part {part_info['k']} of {part_info['n']} in a LinkedIn series.\n\n"
            f"This part should focus on: {part_info['focus']}\n\n"
            f"Full notes for context (only cover the focus above in this post):\n\n{notes_text}"
        )
    else:
        user_msg = (
            f"Write a LinkedIn post from these rough notes. "
            f"If the notes contain too much content for one 350-word post, "
            f"output ONLY this JSON instead (no other text): "
            f'{{"action": "split", "parts": N, "reason": "..."}}\n\n'
            f"Notes:\n\n{notes_text}"
        )
    response = model.generate_content(user_msg)
    return response.text.strip()

def is_split_signal(text):
    """Check if model returned a JSON split signal instead of a post."""
    cleaned = text.strip()
    # strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    if not cleaned.startswith("{"):
        return None
    try:
        data = json.loads(cleaned)
        if data.get("action") == "split" and "parts" in data:
            return data
    except json.JSONDecodeError:
        return None
    return None

def get_split_outline(notes_text, n_parts):
    """Ask the model to outline what each part of the series should cover."""
    outline_prompt = (
        f"Break these notes into {n_parts} LinkedIn posts that work as a series. "
        f"Each post should have a clear, distinct focus. "
        f"Output ONLY this JSON, no other text:\n"
        f'{{"parts": [{{"k": 1, "focus": "..."}}, {{"k": 2, "focus": "..."}}]}}\n\n'
        f"Notes:\n\n{notes_text}"
    )
    # Use a fresh call without the heavy system prompt for cleaner JSON
    planner = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={"temperature": 0.3},
    )
    response = planner.generate_content(outline_prompt)
    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)

# ---------- Main flow ----------
print(f"Generating with {MODEL_NAME}...\n")

first_output = generate_post(notes)
split = is_split_signal(first_output)

date_tag = datetime.now().strftime("%Y%m%d")

if split is None:
    # Single post
    out_path = OUTPUT_DIR / f"post_{date_tag}.md"
    out_path.write_text(first_output, encoding="utf-8")
    word_count = len(first_output.split())
    print(f"Single post generated ({word_count} words)")
    print(f"Saved to: {out_path}\n")
    print("=" * 60)
    print(first_output)
    print("=" * 60)
else:
    # Multi-part series
    n = split["parts"]
    print(f"Notes too long for one post. Splitting into {n} parts.")
    print(f"Reason: {split.get('reason', 'N/A')}\n")
    
    outline = get_split_outline(notes, n)
    series_dir = OUTPUT_DIR / f"series_{date_tag}"
    series_dir.mkdir(exist_ok=True)
    
    for part in outline["parts"]:
        k = part["k"]
        focus = part["focus"]
        print(f"Generating part {k}/{n} - focus: {focus}")
        post_text = generate_post(notes, {"k": k, "n": n, "focus": focus})
        out_path = series_dir / f"post_day{k}.md"
        out_path.write_text(post_text, encoding="utf-8")
        word_count = len(post_text.split())
        print(f"  Saved ({word_count} words): {out_path}\n")
    
    print(f"Series complete. {n} posts saved in: {series_dir}")
    print("Post one per day, starting with post_day1.md")