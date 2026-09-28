import os
import json
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

from google import genai


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = "submission.xls"
OUTPUT_FILE = "top20_llm_reasoning.csv"

MODEL = "gemini-2.5-flash"

TOP_N = 20
BATCH_SIZE = 10


# ============================================================
# 2. LOAD GEMINI API KEY
# ============================================================

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env or environment variables"
    )

client = genai.Client(api_key=api_key)


# ============================================================
# 3. RESOLVE INPUT FILE
# ============================================================

def resolve_input_file():
    """Resolve the submitted data file from common spreadsheet/csv names."""
    candidates = [
        INPUT_FILE,
        "submission.csv",
        "submission.xlsx",
        "submission.xlsm",
        "submission.tsv",
    ]

    for candidate in candidates:
        path = BASE_DIR / candidate
        if path.exists():
            return path

    fallback = BASE_DIR / INPUT_FILE
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "No submission file found. Expected one of: "
        + ", ".join(candidates)
    )


# ============================================================
# 4. LOAD SUBMISSION DATA
# ============================================================

def read_submission_file(path):
    """Read CSV or Excel input robustly, including CSVs misnamed as .xls."""
    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(path)

    try:
        return pd.read_excel(path)
    except ValueError:
        # Some files are actually CSVs but saved with an Excel-style extension.
        try:
            return pd.read_csv(path)
        except Exception:
            raise


INPUT_PATH = resolve_input_file()


# ============================================================
# 5. LOAD RAW SUBMISSION FILE
# ============================================================

print("Loading raw submission file...")

df = read_submission_file(INPUT_PATH)

print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "candidate_id",
    "rank",
    "score",
    "reasoning"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 7. CLEAN RANK
# ============================================================

df["rank"] = pd.to_numeric(
    df["rank"],
    errors="coerce"
)

df = df.dropna(
    subset=["rank"]
)

df["rank"] = df["rank"].astype(int)


# ============================================================
# 8. SELECT TOP 20
# ============================================================

top20 = (
    df.sort_values("rank")
      .head(TOP_N)
      .copy()
)

print("\n========================================")
print("TOP 20 CANDIDATES")
print("========================================")

print(
    top20[
        ["candidate_id", "rank", "score"]
    ].to_string(index=False)
)


# ============================================================
# 9. JOB DESCRIPTION
# ============================================================

with open(BASE_DIR / "jd.txt", "r", encoding="utf-8") as f:
    jd = f.read()


# ============================================================
# 10. FUNCTION TO GENERATE BATCH REASONING
# ============================================================

def generate_reasoning(candidate_batch):

    candidate_text = json.dumps(
        candidate_batch,
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""
You are an AI-assisted candidate-fit explanation system.

The candidates below have already been ranked by a separate
candidate-ranking system.

Your task is ONLY to explain the relationship between each
candidate's provided evidence and the job description.

You are NOT responsible for ranking candidates.

Do NOT:
- change the candidate's rank
- change the candidate's score
- recalculate the ranking
- recommend hiring
- reject a candidate
- invent candidate information

============================================================
JOB DESCRIPTION
============================================================

{jd}

============================================================
CANDIDATE EVIDENCE
============================================================

{candidate_text}

============================================================
STRICT EVIDENCE RULES
============================================================

Use ONLY the information contained in each candidate's
"reasoning" field.

Never invent:

- skills
- technologies
- projects
- employers
- responsibilities
- years of experience
- production experience
- deployment
- users
- scale
- ranking metrics
- vector databases
- embeddings experience
- LLM experience

For example:

If the evidence says "Python", you may say the candidate
demonstrates Python.

But you must NOT say that the candidate used Python in
production unless the evidence explicitly establishes that.

If the evidence does not establish a requirement, do not
claim that the candidate has it.

You may state that:

"The provided evidence does not establish experience with X."

============================================================
REASONING REQUIREMENTS
============================================================

For every candidate:

- Write 4–6 professional sentences.
- Start with the strongest demonstrated areas of alignment.
- Mention concrete skills, technologies, experience, or titles
  when they are present in the evidence.
- Relate those facts to the Senior AI Engineer role.
- Mention important gaps only when useful.
- Keep the explanation factual and concise.

Do NOT use:

"perfect candidate"
"best candidate"
"excellent candidate"
"should definitely be hired"
"guaranteed fit"

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "candidates": [
        {{
            "candidate_id": "candidate_id_here",
            "llm_reasoning": "Professional evidence-grounded explanation."
        }}
    ]
}}

Return exactly one object for every candidate supplied.
"""


    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json"
        }
    )

    return json.loads(response.text)


# ============================================================
# 11. PROCESS TOP 20 IN BATCHES
# ============================================================

all_reasoning = {}

total_batches = (
    len(top20) + BATCH_SIZE - 1
) // BATCH_SIZE


print("\n========================================")
print("STARTING GEMINI PROCESSING")
print("========================================")

for batch_number, start in enumerate(
    range(0, len(top20), BATCH_SIZE),
    start=1
):

    end = min(
        start + BATCH_SIZE,
        len(top20)
    )

    batch_df = top20.iloc[start:end]

    candidate_batch = []

    for _, row in batch_df.iterrows():

        candidate_batch.append(
            {
                "candidate_id": str(
                    row["candidate_id"]
                ),
                "rank": int(
                    row["rank"]
                ),
                "score": float(
                    row["score"]
                ),
                "reasoning": str(
                    row["reasoning"]
                )
            }
        )

    print(
        f"\nBatch {batch_number}/{total_batches}"
    )

    print(
        f"Processing candidates "
        f"{start + 1}–{end}"
    )

    try:

        result = generate_reasoning(
            candidate_batch
        )

        candidates_result = result.get(
            "candidates",
            []
        )

        for item in candidates_result:

            candidate_id = str(
                item["candidate_id"]
            )

            llm_reasoning = str(
                item["llm_reasoning"]
            ).strip()

            if llm_reasoning:

                all_reasoning[
                    candidate_id
                ] = llm_reasoning

        print(
            f"Batch completed."
        )

        print(
            f"Explanations generated: "
            f"{len(all_reasoning)}"
        )

    except Exception as error:

        print(
            "\nERROR while processing batch:"
        )

        print(error)

        print(
            "\nStopping the program."
        )

        break


# ============================================================
# 12. CREATE OUTPUT
# ============================================================

print("\n========================================")
print("CREATING OUTPUT FILE")
print("========================================")


output_rows = []

for _, row in top20.iterrows():

    candidate_id = str(
        row["candidate_id"]
    )

    output_rows.append(
        {
            "candidate_id": candidate_id,
            "rank": int(row["rank"]),
            "score": row["score"],
            "llm_reasoning": all_reasoning.get(
                candidate_id,
                ""
            )
        }
    )


output_df = pd.DataFrame(
    output_rows
)


# ============================================================
# 13. SAVE OUTPUT
# ============================================================

output_path = BASE_DIR / OUTPUT_FILE
output_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. FINAL STATUS
# ============================================================

generated = (
    output_df["llm_reasoning"]
    .astype(str)
    .str.strip()
    .ne("")
    .sum()
)

missing = TOP_N - generated


print("\n========================================")
print("FINAL STATUS")
print("========================================")

print(
    f"Raw candidates loaded : {len(df)}"
)

print(
    f"Top candidates used   : {len(top20)}"
)

print(
    f"Reasoning generated   : {generated}"
)

print(
    f"Reasoning missing     : {missing}"
)

print(
    f"\nOutput file: {output_path}"
)

if generated == TOP_N:

    print(
        "\nSUCCESS: LLM reasoning generated "
        "for all Top 20 candidates."
    )

else:

    print(
        "\nWARNING: Some Top 20 candidates "
        "do not have reasoning."
    )