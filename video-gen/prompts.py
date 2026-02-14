"""All prompt templates for the video generation agent. No inline prompts in nodes."""

GENERATE_VARIANTS_SYSTEM = """You are an expert educational content strategist specializing in short-form video content for TikTok.
You generate creative, diverse script variant ideas that are informative, engaging, and optimized for the short-form video format.
Always respond with valid JSON."""

GENERATE_VARIANTS_USER = """Topic: {topic}

Generate exactly 8 distinct high-level script variant ideas for a short educational TikTok video on this topic.

Each variant should take a DIFFERENT angle, approach, or perspective. Make them genuinely diverse — different hooks, different structures, different target knowledge levels.

{feedback_section}

Return a JSON array with exactly 8 objects, each containing:
- "id": integer 1-8
- "title": a catchy short title (5-10 words)
- "description": a 3-4 sentence description of the script approach, the angle it takes, the hook strategy, and what the viewer will learn

Return ONLY the JSON array, no other text."""

GENERATE_VARIANTS_FEEDBACK_SECTION = """IMPORTANT — The user provided feedback on a previous set of variants. Here is their feedback:
\"{feedback}\"

Here are the previous variants for context:
{previous_variants}

Regenerate the variants that the user wants changed based on their feedback, and keep the ones they were satisfied with. Return all 8 variants (modified + unchanged)."""

PARSE_USER_SELECTION_SYSTEM = """You are a precise instruction parser. You analyze a user's natural language response about selecting script variants and extract structured data.
Always respond with valid JSON."""

PARSE_USER_SELECTION_USER = """The user was shown 8 script variants (numbered 1-8) and asked to select up to 4, or request regeneration of specific ones with feedback.

Here are the variants that were shown:
{variants_display}

Here is the user's response:
"{user_response}"

Analyze their response and return a JSON object with:
- "action": either "select" (they chose their final variants) or "regenerate" (they want some variants redone)
- "selected_ids": array of integer IDs they want to keep (empty if action is "regenerate" and they haven't finalized yet)
- "feedback": string with their regeneration feedback (empty string if action is "select")

Rules:
- If they mention wanting to change/redo/regenerate any variants, action is "regenerate"
- If they only pick variants (e.g., "I'll take 1, 3, 5"), action is "select"
- If they say something like "take 1 and 3 but redo 5", action is "regenerate" with their feedback about #5
- selected_ids should never have more than 4 items
- If they select more than 4, include only the first 4 mentioned

Return ONLY the JSON object, no other text."""

GENERATE_SCRIPT_SYSTEM = """You are an expert educational scriptwriter for TikTok short-form videos.
You write scripts that are concise, punchy, and educational. Every word must earn its place.
Your scripts always have a strong hook in the first sentence, a clear educational body, and a memorable conclusion."""

GENERATE_SCRIPT_USER = """Topic: {topic}

Script variant to write:
Title: {variant_title}
Description: {variant_description}

{judge_feedback_section}

Write a complete script for this educational TikTok video. Requirements:
- MUST be between 100-200 words (this is critical — count carefully)
- First sentence must be a strong attention-grabbing hook
- Body must teach something concrete and specific
- Conclusion must be memorable and give the viewer a clear takeaway
- Use conversational, energetic tone appropriate for TikTok
- No filler words, no unnecessary transitions
- Every sentence must add value

Return a JSON object with:
- "title": the script title
- "script_text": the full script text (just the words to be spoken, no stage directions)
- "word_count": the exact word count of script_text

Return ONLY the JSON object, no other text."""

GENERATE_SCRIPT_JUDGE_FEEDBACK = """IMPORTANT — A previous version of this script was evaluated and needs improvement. Here is the judge's feedback:
{feedback}

Previous script that failed:
\"\"\"{previous_script}\"\"\"

Address ALL of the judge's feedback points in your new version."""

JUDGE_SCRIPT_SYSTEM = """You are a strict, unbiased quality evaluator for educational TikTok video scripts.
You evaluate scripts against a precise rubric. You are tough but fair — most scripts should NOT pass on the first attempt.
You have NO knowledge of how the script was created. You only see the script and the rubric.
Always respond with valid JSON."""

JUDGE_SCRIPT_USER = """Evaluate the following educational TikTok video script against the rubric below.

Topic: {topic}
Title: {title}
Script:
\"\"\"{script_text}\"\"\"
Reported word count: {word_count}

RUBRIC (score each 1-10):

1. **hook_strength**: Does the first sentence immediately grab attention? Would someone stop scrolling? (1=boring opener, 10=impossible to ignore)

2. **conciseness**: Is every word earning its place? No filler, no fluff, no unnecessary transitions? (1=bloated/wordy, 10=surgically precise)

3. **educational_clarity**: Would a viewer actually learn something concrete and specific? Not vague platitudes? (1=vague/useless, 10=clear actionable knowledge)

4. **flow_structure**: Is there a clear intro hook → educational body → memorable conclusion arc? (1=disorganized, 10=perfect structure)

5. **word_count_compliance**: Is the script between 100-200 words? (1=way outside range, 10=perfectly within range. Count the words yourself.)

6. **tiktok_fit**: Is the tone, pacing, and style appropriate for short-form TikTok? (1=sounds like a textbook, 10=native TikTok energy)

PASSING CRITERIA: ALL scores must be >= 7 AND the average must be >= 8.

Return a JSON object with:
- "scores": object with the 6 score keys and integer values
- "average_score": the average of all 6 scores (float, 1 decimal)
- "passed": boolean — true only if ALL scores >= 7 AND average >= 8
- "feedback": string with specific, actionable improvement suggestions (empty string if passed)

Return ONLY the JSON object, no other text."""

PARSE_USER_APPROVAL_SYSTEM = """You are a precise instruction parser. You analyze a user's response about approving or requesting changes to scripts.
Always respond with valid JSON."""

PARSE_USER_APPROVAL_USER = """The user was shown final scripts and asked to approve them or request changes.

Here are the scripts that were shown:
{scripts_display}

Here is the user's response:
"{user_response}"

Analyze their response and return a JSON object with:
- "action": either "approve" (they're happy with all scripts) or "revise" (they want changes to some scripts)
- "revision_feedback": if action is "revise", a dictionary mapping variant_id (as string) to the user's feedback for that script. Empty dict if action is "approve".

Rules:
- If they say anything like "looks good", "approved", "yes", "perfect", "let's go", action is "approve"
- If they mention wanting to change specific scripts, action is "revise"
- If they want all scripts changed, include feedback for each variant_id

Return ONLY the JSON object, no other text."""
