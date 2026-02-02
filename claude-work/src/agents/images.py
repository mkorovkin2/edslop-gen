"""Image collection, mapping, and download agents."""

import os
import json
import logging
import aiohttp
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
from ..utils.tavily_client import TavilyClient
from ..utils.openai_client import OpenAIClient
from ..models import WorkflowState

logger = logging.getLogger(__name__)


async def collect_images_node(
    state: WorkflowState,
    tavily_client: TavilyClient,
    openai_client: OpenAIClient
) -> Dict[str, Any]:
    """
    Collect images from Tavily using LLM-generated queries.

    Args:
        state: Current workflow state
        tavily_client: Tavily API client
        openai_client: OpenAI API client

    Returns:
        Dict with updated images, metadata, and api_call_counts
    """
    # Validate input
    if not state.get('script_sections'):
        raise ValueError("No script sections available for image collection")

    sections = state['script_sections']
    outline = state.get('script_outline', '').strip()
    images_per_section = int(os.getenv('IMAGES_PER_SECTION', '5'))
    logger.info(
        "Images: generating search queries for %d sections (%d per section)",
        len(sections),
        images_per_section
    )

    # Step 1: Batch generate all image search queries using LLM
    outline_block = f"\nOutline (for context):\n{outline}\n" if outline else ""
    query_generation_prompt = f"""
You are helping create an educational video about "{state['topic']}".
Generate image search queries for the following script sections.
{outline_block}

For each section, generate {images_per_section} diverse, specific image search queries that will find
relevant visual aids (diagrams, illustrations, photos, charts).

Script sections:
{json.dumps([{"section_id": s['section_id'], "title": s.get('title', ''), "text": s['text']} for s in sections], indent=2)}

Return a JSON object mapping section_id to list of queries:
{{
  "section_1": ["query 1", "query 2", ...],
  "section_2": ["query 1", "query 2", ...],
  ...
}}

Make queries specific and descriptive (e.g., "quantum computer circuit diagram", "photosynthesis process illustration").
Return ONLY the JSON object.
"""

    queries_json = await openai_client.generate(
        query_generation_prompt,
        max_tokens=1500,
        temperature=0.7
    )

    # Parse queries
    try:
        queries_by_section = json.loads(queries_json.strip())
        if not isinstance(queries_by_section, dict):
            raise ValueError("Invalid queries format")
    except Exception as e:
        logger.warning("Images: failed to parse queries JSON, using fallback. Error: %s", e)
        # Fallback: simple queries
        queries_by_section = {
            s['section_id']: [f"{state['topic']} {s.get('title', '')}", f"{state['topic']} diagram"]
            for s in sections
        }

    total_queries = 0
    for q in queries_by_section.values():
        if isinstance(q, list):
            total_queries += len(q)
    logger.info("Images: using %d queries", total_queries)
    logger.debug("Images: query sections=%s", list(queries_by_section.keys()))

    # Step 2: Execute Tavily image searches
    all_images = []
    all_queries = []

    for section_id, queries in queries_by_section.items():
        for query in queries:
            all_queries.append(query)
            try:
                logger.debug("Images: searching query=%r (section=%s)", query, section_id)
                images = await tavily_client.search_images(query, max_results=10)
                normalized_images = []
                for img in images:
                    if isinstance(img, str):
                        img = {"url": img, "description": ""}
                    if not isinstance(img, dict):
                        continue
                    if not img.get("url"):
                        continue
                    img["description"] = img.get("description", "")
                    img["query_used"] = query
                    img["section_id"] = section_id
                    normalized_images.append(img)
                all_images.extend(normalized_images)
                logger.debug(
                    "Images: %d results for query=%r",
                    len(normalized_images),
                    query
                )
            except Exception as e:
                logger.warning("Images: search failed for %r: %s", query, e)
                continue

    if len(all_images) == 0:
        logger.error("Images: no images collected")
        raise ValueError("No images collected. Image search may have failed.")

    # Get retry count
    retry_count = state.get('retry_counts', {}).get('collect_images', 0)

    logger.info(
        "Images: collected %d images across %d queries",
        len(all_images),
        len(all_queries)
    )
    return {
        "images": all_images,
        "metadata": {
            **state.get("metadata", {}),
            "image_collection_completed_at": datetime.now().isoformat(),
            "image_queries_used": len(all_queries),
            "images_collected": len(all_images)
        },
        "retry_counts": {
            **state.get("retry_counts", {}),
            "collect_images": retry_count + 1
        },
        "api_call_counts": {
            **state.get("api_call_counts", {}),
            "tavily": state.get("api_call_counts", {}).get("tavily", 0) + len(all_queries),
            "openai": state.get("api_call_counts", {}).get("openai", 0) + 1
        }
    }


async def map_images_node(
    state: WorkflowState,
    openai_client: OpenAIClient
) -> Dict[str, Any]:
    """
    Use LLM to map images to script sections.

    Args:
        state: Current workflow state
        openai_client: OpenAI API client

    Returns:
        Dict with updated images_mapping and metadata
    """
    # Validate input
    if not state.get('images'):
        raise ValueError("No images available for mapping")

    if not state.get('script_sections'):
        raise ValueError("No script sections available for mapping")

    sections = state['script_sections']
    images = state['images']
    outline = state.get('script_outline', '').strip()
    logger.info(
        "Images: mapping %d images to %d sections",
        len(images),
        len(sections)
    )

    # Build image descriptions
    image_descriptions = [
        f"Image {i}: {img.get('description', 'No description')} (from query: {img.get('query_used', 'unknown')})"
        for i, img in enumerate(images)
    ]

    outline_block = f"\nOutline (for context):\n{outline}\n" if outline else ""
    mapping_prompt = f"""
You are creating an educational video about "{state['topic']}".
Map the most relevant images to each script section.
{outline_block}

Script sections:
{json.dumps([{"section_id": s['section_id'], "title": s.get('title', ''), "text": s['text']} for s in sections], indent=2)}

Available images:
{chr(10).join(image_descriptions)}

For each section, select 2-4 most relevant images by their index numbers.

Return a JSON object mapping section_id to list of image indices:
{{
  "section_1": [0, 5, 12],
  "section_2": [3, 8, 15, 20],
  ...
}}

Return ONLY the JSON object.
"""

    mapping_json = await openai_client.generate(
        mapping_prompt,
        max_tokens=1000,
        temperature=0.3
    )

    # Parse mapping
    try:
        images_mapping = json.loads(mapping_json.strip())
        if not isinstance(images_mapping, dict):
            raise ValueError("Invalid mapping format")
    except Exception as e:
        logger.warning("Images: failed to parse mapping JSON, using fallback. Error: %s", e)
        # Fallback: distribute images evenly across sections
        images_per_section = len(images) // len(sections)
        images_mapping = {}
        for i, section in enumerate(sections):
            start_idx = i * images_per_section
            end_idx = start_idx + images_per_section
            images_mapping[section['section_id']] = list(range(start_idx, min(end_idx, len(images))))

    logger.info("Images: mapping complete (%d sections mapped)", len(images_mapping))
    return {
        "images_mapping": images_mapping,
        "metadata": {
            **state.get("metadata", {}),
            "image_mapping_completed_at": datetime.now().isoformat()
        },
        "api_call_counts": {
            **state.get("api_call_counts", {}),
            "openai": state.get("api_call_counts", {}).get("openai", 0) + 1
        }
    }


async def download_images_node(
    state: WorkflowState,
    openai_client: OpenAIClient,
    output_dir: str = "output"
) -> Dict[str, Any]:
    """
    Download mapped images and save to disk.

    Args:
        state: Current workflow state
        openai_client: OpenAI API client for filename generation
        output_dir: Base output directory

    Returns:
        Dict with updated images list and metadata
    """
    # Validate input
    if not state.get('images_mapping'):
        raise ValueError("No image mapping available")

    images = state['images']
    images_mapping = state['images_mapping']
    run_id = state['run_id']

    # Get all unique image indices to download
    image_indices = set()
    for indices in images_mapping.values():
        image_indices.update(indices)

    logger.info("Images: downloading %d unique images", len(image_indices))

    # Create output directory
    images_dir = Path(output_dir) / run_id / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Download images concurrently
    async def download_image(idx: int, image_meta: Dict) -> tuple:
        """Download single image and generate filename."""
        url = image_meta['url']
        description = image_meta.get('description', 'image')

        try:
            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        image_data = await response.read()

                        # Generate descriptive filename using LLM
                        filename_prompt = f"Create a short, descriptive filename (2-4 words, lowercase, hyphens) for an image described as: '{description}'. Return ONLY the filename without extension."
                        filename_base = await openai_client.generate(filename_prompt, max_tokens=50, temperature=0.3)
                        filename_base = filename_base.strip().replace(' ', '-').replace('_', '-')[:50]

                        # Determine extension from URL or content-type
                        ext = '.jpg'
                        if url.endswith('.png'):
                            ext = '.png'
                        elif url.endswith('.gif'):
                            ext = '.gif'
                        elif response.content_type:
                            if 'png' in response.content_type:
                                ext = '.png'
                            elif 'gif' in response.content_type:
                                ext = '.gif'

                        filename = f"{idx:03d}_{filename_base}{ext}"

                        # Save image
                        image_path = images_dir / filename
                        async with aiofiles.open(image_path, 'wb') as f:
                            await f.write(image_data)

                        return (idx, str(image_path.relative_to(Path(output_dir) / run_id)), None)
                    else:
                        return (idx, None, f"HTTP {response.status}")
        except Exception as e:
            return (idx, None, str(e))

    # Download all images concurrently
    import aiofiles
    tasks = []
    for idx in sorted(image_indices):
        if idx >= len(images):
            continue
        image_meta = images[idx]
        if not isinstance(image_meta, dict):
            continue
        if not image_meta.get("url"):
            continue
        tasks.append(download_image(idx, image_meta))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Update image metadata with local paths
    downloaded = 0
    failed = 0
    failed_urls = []

    for result in results:
        if isinstance(result, Exception):
            failed += 1
            continue

        idx, local_path, error = result
        if local_path:
            images[idx]['local_path'] = local_path
            downloaded += 1
        else:
            failed += 1
            url = ""
            if idx < len(images) and isinstance(images[idx], dict):
                url = images[idx].get("url", "")
            failed_urls.append(f"{url} ({error})")

    logger.info("Images: downloaded %d, failed %d", downloaded, failed)
    return {
        "images": images,  # Updated with local_path
        "metadata": {
            **state.get("metadata", {}),
            "images_downloaded_at": datetime.now().isoformat(),
            "images_downloaded": downloaded,
            "images_failed": failed,
            "images_failed_urls": failed_urls[:10]  # Limit to first 10
        },
        "api_call_counts": {
            **state.get("api_call_counts", {}),
            "openai": state.get("api_call_counts", {}).get("openai", 0) + len(tasks)  # Filename generation calls
        }
    }


def validate_image_count(state: WorkflowState) -> str:
    """
    Conditional edge function to validate image collection.

    Args:
        state: Current workflow state

    Returns:
        "sufficient" if enough images, "retry" if insufficient (and retries remaining), "fallback" if exhausted
    """
    min_images = int(os.getenv('IMAGES_MIN_TOTAL', '10'))
    image_count = len(state.get('images', []))
    retry_count = state.get('retry_counts', {}).get('collect_images', 0)

    if image_count >= min_images:
        return "sufficient"
    elif retry_count >= 2:
        logger.warning(
            "Images: only %d collected (target: %d) after %d retries; continuing.",
            image_count,
            min_images,
            retry_count
        )
        return "fallback"
    else:
        logger.info(
            "Images: insufficient images (%d/%d); retrying with different queries...",
            image_count,
            min_images
        )
        return "retry"
