"""Node: Generate video segments via Sora API — create, poll, download MP4s."""

import logging
import os
import time

import httpx

from persistence import save_thoughts
from llm import get_sora_client, retry_with_backoff
from state import AgentState


def _download_video(video_id: str, output_path: str, api_key: str):
    """Download completed video MP4 via raw HTTP (SDK doesn't expose this endpoint)."""
    url = f"https://api.openai.com/v1/videos/{video_id}/content"
    with httpx.stream("GET", url, headers={"Authorization": f"Bearer {api_key}"}, timeout=300.0) as resp:
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)


def generate_videos(state: AgentState) -> dict:
    """Call Sora API to generate video segments for each variant's breakdown."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: generate_videos ===")

    breakdowns = state["video_breakdown"]
    existing_paths = state.get("video_paths", [])

    client = get_sora_client()

    all_video_paths = list(existing_paths)  # Preserve any already-generated paths on resume

    print("\n" + "=" * 60)
    print("GENERATING VIDEOS (Sora API)")
    print("=" * 60)

    for bd in breakdowns:
        vid = bd["variant_id"]
        title = bd.get("variant_title", f"Variant {vid}")
        segments = bd.get("segments", [])

        # Create output directory for this variant
        video_dir = os.path.join("output", sid, "videos", f"variant_{vid}")
        os.makedirs(video_dir, exist_ok=True)

        print(f"\n  Variant {vid}: {title}")
        print(f"  Generating {len(segments)} video segments...")

        for seg in segments:
            seg_id = seg.get("segment_id", 0)
            sora_prompt = seg.get("sora_prompt", "")
            duration = seg.get("duration", "4")
            size = seg.get("size", "720x1280")
            model = seg.get("model", os.getenv("SORA_MODEL", "sora-2"))
            filename = seg.get("filename", f"part_{seg_id}.mp4")

            output_path = os.path.join(video_dir, filename)

            # Skip if already generated (resume support)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                log.info(f"Segment {seg_id} already exists at {output_path}, skipping")
                print(f"    Segment {seg_id}: already exists, skipping")
                if output_path not in all_video_paths:
                    all_video_paths.append(output_path)
                continue

            log.info(f"Creating Sora video: variant {vid}, segment {seg_id}")
            print(f"    Segment {seg_id}: submitting to Sora ({duration}s, {size})...")

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Create video generation job with rate limit retry
                    # Use default args to capture current loop values
                    video = retry_with_backoff(
                        lambda _m=model, _p=sora_prompt, _sz=size, _d=duration: client.videos.create(
                            model=_m,
                            prompt=_p,
                            size=_sz,
                            seconds=_d,
                        ),
                        max_retries=5,
                        base_delay=5.0,
                        max_delay=300.0,
                    )
                    video_id = video.id
                    log.info(f"Sora job created: {video_id} (attempt {attempt + 1})")

                    # Poll until completed or failed (with rate limit handling on poll)
                    poll_interval = 5
                    max_polls = 360  # 30 min max wait
                    for poll_num in range(max_polls):
                        status_resp = retry_with_backoff(
                            lambda _vid=video_id: client.videos.retrieve(_vid),
                            max_retries=5,
                            base_delay=5.0,
                        )
                        status = status_resp.status

                        if status == "completed":
                            log.info(f"Sora job {video_id} completed")
                            print(f"    Segment {seg_id}: completed, downloading...")

                            # Download MP4 via raw HTTP (SDK has no .content() method)
                            api_key = os.getenv("OPENAI_API_KEY")
                            retry_with_backoff(
                                lambda _vid=video_id, _out=output_path, _key=api_key: _download_video(_vid, _out, _key),
                                max_retries=5,
                                base_delay=5.0,
                            )

                            all_video_paths.append(output_path)
                            log.info(f"Saved video to {output_path}")
                            print(f"    Segment {seg_id}: saved to {output_path}")
                            break

                        elif status == "failed":
                            err = getattr(status_resp, "error", None)
                            err_code = getattr(err, "code", "unknown") if err else "unknown"
                            err_msg = getattr(err, "message", "no details") if err else "no details"
                            log.error(f"Sora job {video_id} failed: [{err_code}] {err_msg}")
                            print(f"    Segment {seg_id}: FAILED — [{err_code}] {err_msg}")
                            raise RuntimeError(f"Sora failed for {video_id}: [{err_code}] {err_msg}")

                        else:
                            progress = getattr(status_resp, "progress", None)
                            if poll_num % 6 == 0:  # Print status every ~30s
                                pct = f" {progress}%" if progress else ""
                                print(f"    Segment {seg_id}: {status}{pct}... (polling)")
                            time.sleep(poll_interval)
                    else:
                        raise TimeoutError(
                            f"Sora job {video_id} timed out after {max_polls * poll_interval}s"
                        )

                    # If we get here, download succeeded — break retry loop
                    break

                except Exception as e:
                    log.error(f"Error generating segment {seg_id} (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        log.info("Retrying...")
                        print(f"    Segment {seg_id}: error, retrying...")
                    else:
                        log.error(f"Failed to generate segment {seg_id} after {max_retries} attempts")
                        print(f"    Segment {seg_id}: FAILED after {max_retries} attempts — {e}")

    new_state = {
        "video_paths": all_video_paths,
        "current_step": "generate_videos",
    }

    save_thoughts(sid, "13_generate_videos", {**state, **new_state})
    log.debug("Saved thoughts for generate_videos")

    # Final summary
    videos_dir = os.path.join("output", sid, "videos")
    abs_videos = os.path.abspath(videos_dir)

    print("\n" + "=" * 60)
    print("VIDEO GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nAll videos saved to: {abs_videos}")
    for p in all_video_paths:
        print(f"  {os.path.abspath(p)}")
    print(f"\n{len(all_video_paths)} video segments generated.")
    print("=" * 60 + "\n")

    return new_state
