"""Prompt templates for LLM agents."""

from __future__ import annotations

SCRIPT_EDITOR_SYSTEM_MESSAGE = (
    "You are an educational video script editor. "
    "Sound conversational, concise, and confident. "
    "Focus on ONE specific aspect of the topic and keep the scope narrow and cohesive. "
    "Do not reference sources, docs, specs, or studies; speak with direct authority. "
    "Prioritize conveying all required information over flow or polish. "
    "Transitions can be rough; it is fine to jump between points. "
    "Use common acronyms as-is (e.g., DARPA) without expanding them. "
    "Every sentence must introduce a concrete fact, mechanism, or necessary transition; no filler. "
    "Address the reader directly and follow any provided outline while keeping the output as continuous paragraphs. "
    "Do not invent facts not supported by the provided research."
)

SCRIPT_WRITER_SYSTEM_MESSAGE = (
    "You are an educational video script writer. "
    "Sound conversational, concise, and confident. "
    "Focus on ONE specific aspect of the topic and keep the scope narrow and cohesive. "
    "Do not reference sources, docs, specs, or studies; speak with direct authority. "
    "Prioritize conveying all required information over flow or polish. "
    "Transitions can be rough; it is fine to jump between points. "
    "Use common acronyms as-is (e.g., DARPA) without expanding them. "
    "Every sentence must introduce a concrete fact, mechanism, or necessary transition; no filler. "
    "Address the reader directly and follow any provided outline while keeping the output as continuous paragraphs. "
    "Do not invent facts not supported by the provided research."
)

SCRIPT_RETURN_ONLY_SUFFIX = "\nReturn ONLY the script text, nothing else."

OUTLINE_SYSTEM_MESSAGE = (
    "You are an educational video script planner. "
    "Produce a concise, single-paragraph plan focused on one specific aspect."
)

OUTLINE_REVISION_SYSTEM_MESSAGE = (
    "You are an educational video script planner. "
    "Revise plan paragraphs based on feedback while preserving the requested format."
)


def script_base_prompt(
    topic_xml: str,
    research_summary_xml: str,
    outline_xml: str,
    sources_xml: str,
    word_count_xml: str,
    retry_notice_xml: str
) -> str:
    return f"""
Create an engaging, informative script about the topic in <topic>.

{topic_xml}

{research_summary_xml}

{outline_xml}

<sources>
{sources_xml}
</sources>

<requirements>
  {word_count_xml}
  <style>Informative, straightforward, and conversational; zero fluff.</style>
  <language>Clear, accessible language.</language>
  <addressing>Address the reader directly as if presenting an informative case in a video.</addressing>
  <content>Include key technical concepts and mechanisms grounded in the research.</content>
  <focus>Focus on a single specific aspect of the topic; if an outline is provided, treat it as the chosen aspect and do not cover other aspects.</focus>
  <outline>Follow the outline if provided, but write continuous paragraphs without headings.</outline>
  <structure>Paragraphs only; no headings, lists, or other formatting.</structure>
  <format>Plain paragraph text only.</format>
  <constraints>
    No stage directions or speaker notes.
    Do not include: greetings, scene-setting, motivational lines, rhetorical questions,
    "in this video" phrasing, or generic wrap-up statements.
    Do not reference sources or attribution (e.g., "the docs say", "according to", "studies show").
    Avoid repetition and vague claims.
    Start immediately with a concrete definition or key technical claim.
    End with a final concrete point (not a summary or wrap-up).
  </constraints>
  <tone>Natural, conversational narration; concise and confident.</tone>
</requirements>

{retry_notice_xml}
"""


def script_judge_prompt(min_words: int, max_words: int, script: str) -> str:
    return f"""
You are a strict editorial QA agent for educational scripts.
Check the script against ALL requirements below. If any fail, return FAIL.

Requirements:
- Word count must be between {min_words} and {max_words}.
- No headings, lists, bullets, or formatting; paragraphs only.
- No greetings, scene-setting, motivational lines, or wrap-up statements.
- No rhetorical questions; no question marks.
- Start with a concrete definition or key technical claim.
- End with a final concrete point (not a summary).
- Every sentence conveys a concrete fact, mechanism, or necessary transition.
- Focuses on one specific aspect; does not try to cover multiple aspects of the topic.
- No source attribution or references to docs/specs/studies.
- Follow the outline if provided.
- Claims must be supported by the provided research summary/sources; no invented facts.

If any rule is violated or you are unsure, FAIL.

Return ONLY JSON:
{{
  "pass": true/false,
  "issues": ["short issue", ...],
  "fix_instructions": "concise rewrite guidance",
  "word_count": 0,
  "quality_score": 0.0
}}

Script:
<script>
{script}
</script>
"""


def script_revision_prompt(
    quality_issues: list[str],
    last_fix_instructions: str,
    min_words: int,
    max_words: int,
    script: str
) -> str:
    return f"""
Revise the script to fix ONLY the issues listed below. Make the smallest possible edits.
Do not add new topics or sources. Preserve the same focus and content unless it violates a rule.

Issues: {quality_issues}
Fix instructions: {last_fix_instructions}

Requirements to keep:
- Word count between {min_words} and {max_words}.
- Paragraphs only; no headings or lists.
- No greetings, scene-setting, rhetorical questions, or wrap-up statements.
- No source attribution or references to docs/specs/studies.
- Start with a concrete definition or key technical claim.
- End with a final concrete point (not a summary).
- Focus on one specific aspect only.

Script to revise:
<script>
{script}
</script>

Return ONLY the revised script text, nothing else.
"""


def script_parse_prompt(script: str, outline: str) -> str:
    outline_block = f"\nPlan (for guidance):\n{outline}\n" if outline else ""
    return f"""
Parse the following educational script into logical sections.
For each section, identify:
1. A descriptive title (if the section has one)
2. The full text of the section
3. Individual sentences within that section
{outline_block}
If an outline is provided, use it to guide section boundaries and titles where it fits the script.

Return your response as a JSON array of sections like this:
[
  {{
    "section_id": "section_1",
    "title": "Introduction to Topic",
    "text": "Full section text here...",
    "sentences": ["First sentence.", "Second sentence.", "Third sentence."]
  }},
  ...
]

Script:
<script>
{script}
</script>

Return ONLY the JSON array, no other text.
"""


def script_feedback_prompt(
    topic: str,
    feedback: str,
    script: str,
    research_summary_xml: str,
    outline_xml: str,
    sources_xml: str,
    word_count_xml: str
) -> str:
    return f"""
Revise the script based on the feedback. Keep the same focus and constraints.

<topic>{topic}</topic>
{research_summary_xml}
{outline_xml}
<sources>
{sources_xml}
</sources>

Feedback:
{feedback}

Current script:
<script>
{script}
</script>

<requirements>
  {word_count_xml}
  <style>Informative, straightforward, and conversational; zero fluff.</style>
  <language>Clear, accessible language.</language>
  <addressing>Address the reader directly as if presenting an informative case in a video.</addressing>
  <content>Include key technical concepts and mechanisms grounded in the research.</content>
  <focus>Focus on a single specific aspect of the topic; if an outline is provided, treat it as the chosen aspect and do not cover other aspects.</focus>
  <outline>Follow the outline if provided, but write continuous paragraphs without headings.</outline>
  <structure>Paragraphs only; no headings, lists, or other formatting.</structure>
  <format>Plain paragraph text only.</format>
  <constraints>
    No stage directions or speaker notes.
    Do not include: greetings, scene-setting, motivational lines, rhetorical questions,
    "in this video" phrasing, or generic wrap-up statements.
    Do not reference sources or attribution (e.g., "the docs say", "according to", "studies show").
    Avoid repetition and vague claims.
    Start immediately with a concrete definition or key technical claim.
    End with a final concrete point (not a summary or wrap-up).
    Do not add facts not supported by the provided research summary/sources.
  </constraints>
  <tone>Natural, conversational narration; concise and confident.</tone>
</requirements>

Return ONLY the revised script text, nothing else.
"""


def script_polish_prompt(
    topic: str,
    script: str,
    research_summary_xml: str,
    outline_xml: str,
    sources_xml: str,
    word_count_xml: str
) -> str:
    return f"""
Improve the script for clarity, tightness, and flow while keeping the same facts and focus.
Make the smallest changes that improve readability. Do not add new facts.

<topic>{topic}</topic>
{research_summary_xml}
{outline_xml}
<sources>
{sources_xml}
</sources>

Current script:
<script>
{script}
</script>

<requirements>
  {word_count_xml}
  <style>Informative, straightforward, and conversational; zero fluff.</style>
  <language>Clear, accessible language.</language>
  <addressing>Address the reader directly as if presenting an informative case in a video.</addressing>
  <content>Include key technical concepts and mechanisms grounded in the research.</content>
  <focus>Focus on a single specific aspect of the topic; if an outline is provided, treat it as the chosen aspect and do not cover other aspects.</focus>
  <outline>Follow the outline if provided, but write continuous paragraphs without headings.</outline>
  <structure>Paragraphs only; no headings, lists, or other formatting.</structure>
  <format>Plain paragraph text only.</format>
  <constraints>
    No stage directions or speaker notes.
    Do not include: greetings, scene-setting, motivational lines, rhetorical questions,
    "in this video" phrasing, or generic wrap-up statements.
    Do not reference sources or attribution (e.g., "the docs say", "according to", "studies show").
    Avoid repetition and vague claims.
    Start immediately with a concrete definition or key technical claim.
    End with a final concrete point (not a summary or wrap-up).
    Do not add facts not supported by the provided research summary/sources.
  </constraints>
  <tone>Natural, conversational narration; concise and confident.</tone>
</requirements>

Return ONLY the improved script text, nothing else.
"""


def outline_prompt(topic: str, min_words: int, max_words: int) -> str:
    return f"""
Create a concise plan paragraph for an educational video script about the topic below.

Topic: {topic}
Script length: {min_words}-{max_words} words (keep it very concise).

Requirements:
- Focus on ONE specific aspect of the topic only.
- Keep the scope narrow and cohesive; do not cover multiple aspects.
- Write a single paragraph (3-5 sentences), no headings or bullets.
- Make it practical and voiceover-friendly.

Return ONLY the paragraph, nothing else.
"""


def outline_revision_prompt(
    topic: str,
    outline: str,
    feedback: str,
    min_words: int,
    max_words: int
) -> str:
    return f"""
Revise the plan paragraph based on the feedback. Keep the same format (single paragraph).

Topic: {topic}
Script length: {min_words}-{max_words} words (keep it very concise).

Current outline:
{outline}

Feedback:
{feedback}

Requirements:
- Focus on ONE specific aspect of the topic only.
- Keep the scope narrow and cohesive; do not broaden to cover multiple aspects.
- Output a single paragraph (3-5 sentences), no headings or bullets.

Return ONLY the revised paragraph, nothing else.
"""


def research_synthesis_prompt(topic: str, outline: str, summaries_text: str) -> str:
    outline_block = f"\nPlan (for focus guidance):\n{outline}\n" if outline else ""
    plan_sentence = (
        "If a plan is provided below, organize the summary to align with it and keep to that single aspect."
        if outline else ""
    )
    return f"""
Synthesize the following research results about "{topic}" into a structured summary.
Focus on key concepts, technical details, and important facts for a single, narrow aspect
of the topic (as implied by the plan if provided).
{plan_sentence}

{outline_block}

Research summaries:
{summaries_text}

Return a concise summary (200-300 words) covering the most important points.
"""


def research_synthesis_judge_prompt(summary_text: str) -> str:
    return f"""
You are a strict QA editor for research summaries.
PASS only if ALL are true:
- 200-300 words.
- Covers the core concepts, technical mechanisms, and key applications.
 - If a plan is provided, the summary clearly aligns with it.
- The summary focuses on one specific aspect and does not broaden to cover multiple aspects.
- No fluff, filler, or generic framing.
- No speculation; only claims supported by the provided research summaries.

If any rule is violated, FAIL. If you are unsure, FAIL.

Return ONLY JSON:
{{
  "pass": true/false,
  "issues": ["short issue", ...],
  "fix_instructions": "one-paragraph guidance to fix",
  "word_count": 0
}}

Summary:
<summary>
{summary_text}
</summary>
"""


def research_synthesis_rewrite_prompt(
    topic: str,
    outline: str,
    summaries_text: str,
    issues: list[str],
    fix_instructions: str
) -> str:
    outline_block = f"\nPlan (for focus guidance):\n{outline}\n" if outline else ""
    return f"""
Rewrite the summary to fix the issues below.
Issues: {issues}
Fix instructions: {fix_instructions}

Topic: {topic}
{outline_block}

Research summaries:
{summaries_text}

Return a concise summary (200-300 words) only.
"""


def research_query_generation_prompt(topic: str, outline: str) -> str:
    outline_block = f"\nPlan (for focus guidance):\n{outline}\n" if outline else ""
    return f"""
You are a research assistant. Given a technical topic, select ONE specific aspect to focus on
and generate 3-5 search queries that gather comprehensive information about that single aspect.

Topic: {topic}
{outline_block}

Generate queries that cover:
1. Basic concepts and definitions
2. Technical details and mechanisms
3. Real-world applications
4. Recent developments or research
5. Related concepts or comparisons

If a plan is provided, derive the single focus from it and ensure queries collectively cover its key points.
All queries must stay within the same narrow aspect; do not broaden scope.

Return ONLY a JSON array of query strings, like:
["query 1", "query 2", "query 3"]
"""


def research_search_prompt(query: str) -> str:
    return f"""
Use web search to collect reliable sources about this query:
{query}

Return ONLY JSON with:
{{
  "query": "{query}",
  "summary": "150-220 word summary grounded in the sources",
  "sources": [
    {{"title": "Title", "url": "https://...", "snippet": "one or two sentence excerpt"}}
  ]
}}
"""


def images_query_generation_prompt(
    topic: str,
    outline: str,
    images_per_section: int,
    sections_payload: str
) -> str:
    outline_block = f"\nOutline (for context):\n{outline}\n" if outline else ""
    return f"""
You are helping create an educational video about "{topic}".
Generate image search queries for the following script sections.
{outline_block}

For each section, generate {images_per_section} diverse, specific image search queries that will find
relevant visual aids (diagrams, illustrations, photos, charts).

Script sections:
{sections_payload}

Return a JSON object mapping section_id to list of queries:
{{
  "section_1": ["query 1", "query 2", ...],
  "section_2": ["query 1", "query 2", ...],
  ...
}}

Make queries specific and descriptive (e.g., "quantum computer circuit diagram", "photosynthesis process illustration").
Return ONLY the JSON object.
"""


def images_mapping_prompt(
    topic: str,
    outline: str,
    sections_payload: str,
    image_descriptions: list[str]
) -> str:
    outline_block = f"\nOutline (for context):\n{outline}\n" if outline else ""
    images_block = "\n".join(image_descriptions)
    return f"""
You are creating an educational video about "{topic}".
Map the most relevant images to each script section.
{outline_block}

Script sections:
{sections_payload}

Available images:
{images_block}

For each section, select 2-4 most relevant images by their index numbers.

Return a JSON object mapping section_id to list of image indices:
{{
  "section_1": [0, 5, 12],
  "section_2": [3, 8, 15, 20],
  ...
}}

Return ONLY the JSON object.
"""


def image_filename_prompt(description: str) -> str:
    return (
        "Create a short, descriptive filename (2-4 words, lowercase, hyphens) for an image described as: "
        f"'{description}'. Return ONLY the filename without extension."
    )
