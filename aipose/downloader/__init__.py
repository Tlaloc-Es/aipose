import hashlib
import os
from pathlib import Path

import requests
from tqdm import tqdm


class Downloader:
    """A class to download files from the internet."""

    def download(
        self,
        model_name: str,
        model_hash: str,
        url_download: str,
    ) -> str:
        """Download a file from the provided URL and store it locally.

        Args:
            model_name (str): Name of the model.
            model_hash (str): Hash of the expected model file.
            url_download (str): URL from where to download the model.

        Returns:
            str: Path to the downloaded model file.
        """
        aipose_path = Path.home() / ".aipose"
        os.makedirs(aipose_path, exist_ok=True)

        aipose_model_folder_path = aipose_path / model_name

        if not aipose_model_folder_path.is_file():
            self._download_file(url_download, aipose_model_folder_path)

        current_model_hash = self._calculate_hash(aipose_model_folder_path)

        if model_hash != current_model_hash:
            self._download_file(url_download, aipose_model_folder_path)

        return str(aipose_model_folder_path)

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate the MD5 hash of a file.

        Args:
            file_path (Path): Path to the file for which to calculate the hash.

        Returns:
            str: The MD5 hash of the file.
        """
        with file_path.open("rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _download_file(self, url: str, file_path: Path) -> None:
        """Download a file from the provided URL and store it locally.

        Args:
            url (str): URL from where to download the file.
            file_path (Path): Path to save the downloaded file.

        Returns:
            None
        """
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            total_size = int(r.headers.get("Content-Length", 0))

            with file_path.open("wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
