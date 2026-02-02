"""Output file management utilities."""

import os
import json
import logging
import aiofiles
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages output directory and file operations."""

    def __init__(self, base_dir: str = "output"):
        """
        Initialize output manager.

        Args:
            base_dir: Base output directory
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def get_run_dir(self, run_id: str) -> Path:
        """
        Get directory path for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            Path to run directory
        """
        return self.base_dir / run_id

    def create_run_directory(self, run_id: str) -> Path:
        """
        Create directory structure for a run.

        Creates:
        - output/<run_id>/
        - output/<run_id>/images/
        - output/<run_id>/voice/

        Args:
            run_id: Run identifier

        Returns:
            Path to run directory
        """
        run_dir = self.get_run_dir(run_id)
        run_dir.mkdir(exist_ok=True)
        (run_dir / "images").mkdir(exist_ok=True)
        (run_dir / "voice").mkdir(exist_ok=True)

        logger.info("Output: created run directory %s", str(run_dir))
        return run_dir

    async def write_script(self, run_id: str, script: str):
        """
        Write script to script.md file.

        Args:
            run_id: Run identifier
            script: Script content
        """
        run_dir = self.get_run_dir(run_id)
        script_path = run_dir / "script.md"

        async with aiofiles.open(script_path, 'w', encoding='utf-8') as f:
            await f.write(f"# Educational Video Script\n\n")
            await f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            await f.write(f"---\n\n")
            await f.write(script)
        logger.debug("Output: wrote script %s", str(script_path))

    async def write_images_json(
        self,
        run_id: str,
        images_mapping: Dict[str, List[int]],
        images: List[Dict[str, Any]]
    ):
        """
        Write images.json mapping file.

        Args:
            run_id: Run identifier
            images_mapping: Mapping of script parts to image indices
            images: List of image metadata
        """
        run_dir = self.get_run_dir(run_id)
        images_json_path = run_dir / "images.json"

        data = {
            "mapping": images_mapping,
            "images": images,
            "total_images": len(images),
            "generated_at": datetime.now().isoformat()
        }

        async with aiofiles.open(images_json_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        logger.debug("Output: wrote images metadata %s", str(images_json_path))

    async def write_metadata(self, run_id: str, metadata: Dict[str, Any]):
        """
        Write meta.json metadata file.

        Args:
            run_id: Run identifier
            metadata: Metadata dictionary
        """
        run_dir = self.get_run_dir(run_id)
        meta_path = run_dir / "meta.json"

        # Add completion timestamp
        metadata['completed_at'] = datetime.now().isoformat()

        # Calculate duration if start time present
        if 'started_at' in metadata:
            start = datetime.fromisoformat(metadata['started_at'])
            end = datetime.fromisoformat(metadata['completed_at'])
            metadata['duration_seconds'] = (end - start).total_seconds()

        async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(metadata, indent=2, ensure_ascii=False))
        logger.debug("Output: wrote run metadata %s", str(meta_path))

    async def save_image(
        self,
        run_id: str,
        image_data: bytes,
        filename: str
    ) -> str:
        """
        Save image file to images directory.

        Args:
            run_id: Run identifier
            image_data: Image binary data
            filename: Filename (e.g., "001_description.jpg")

        Returns:
            Relative path to saved image
        """
        run_dir = self.get_run_dir(run_id)
        images_dir = run_dir / "images"
        image_path = images_dir / filename

        async with aiofiles.open(image_path, 'wb') as f:
            await f.write(image_data)

        logger.debug("Output: saved image %s", str(image_path))
        return str(image_path.relative_to(run_dir))

    async def save_audio(
        self,
        run_id: str,
        audio_data: bytes,
        filename: str = "narration.mp3"
    ) -> str:
        """
        Save audio file to voice directory.

        Args:
            run_id: Run identifier
            audio_data: Audio binary data
            filename: Filename (default: "narration.mp3")

        Returns:
            Full path to saved audio file
        """
        run_dir = self.get_run_dir(run_id)
        voice_dir = run_dir / "voice"
        audio_path = voice_dir / filename

        async with aiofiles.open(audio_path, 'wb') as f:
            await f.write(audio_data)

        logger.debug("Output: saved audio %s", str(audio_path))
        return str(audio_path)

    def cleanup_old_runs(self, max_age_days: int = 7):
        """
        Delete runs older than specified age.

        Args:
            max_age_days: Maximum age in days
        """
        import shutil
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(days=max_age_days)

        for run_dir in self.base_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check modification time
            mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
            if mtime < cutoff_time:
                logger.info("Deleting old run: %s", run_dir.name)
                shutil.rmtree(run_dir)
