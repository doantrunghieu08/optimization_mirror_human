import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

# Compatibility patch for starlette TestClient with httpx >= 0.28
_orig_httpx_init = httpx.Client.__init__
def _compat_httpx_init(self, *args, **kwargs):
    kwargs.pop("app", None)
    _orig_httpx_init(self, *args, **kwargs)
httpx.Client.__init__ = _compat_httpx_init

from fastapi.testclient import TestClient
import torch

from api.main import app
from api.jobs import job_manager, JobStatus


class APIEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
        job_manager._base_dir = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _dummy_pt_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save({"smpl_params_incam": {"global_orient": torch.zeros(1, 3)}}, buf)
        buf.seek(0)
        return buf.read()

    def test_health_check(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "ok"})

    def test_create_job_and_upload_5_files(self):
        pt_bytes = self._dummy_pt_bytes()
        video_bytes = b"fake video content"

        files = [
            ("real_pt", ("real.pt", pt_bytes, "application/octet-stream")),
            ("mirror_pt", ("mirror.pt", pt_bytes, "application/octet-stream")),
            ("vitpose_real", ("vitpose_real.pt", pt_bytes, "application/octet-stream")),
            ("vitpose_mirror", ("vitpose_mirror.pt", pt_bytes, "application/octet-stream")),
            ("video", ("test_video.mp4", video_bytes, "video/mp4")),
        ]

        response = self.client.post("/jobs", files=files)
        self.assertEqual(response.status_code, 201)

        data = response.json()
        job_id = data["job_id"]
        self.assertEqual(data["status"], "created")
        self.assertIn("real_pt", data["uploaded_files"])
        self.assertIn("mirror_pt", data["uploaded_files"])
        self.assertIn("vitpose_real", data["uploaded_files"])
        self.assertIn("vitpose_mirror", data["uploaded_files"])
        self.assertIn("video", data["uploaded_files"])

        # Check files exist on disk in job directory
        job = job_manager.get_job(job_id)
        self.assertTrue((job.inputs_dir / "real" / "hmr4d_results.pt").exists())
        self.assertTrue((job.inputs_dir / "mirror" / "hmr4d_results.pt").exists())
        self.assertTrue((job.inputs_dir / "real" / "preprocess" / "vitpose_real.pt").exists())
        self.assertTrue((job.inputs_dir / "mirror" / "preprocess" / "vitpose_mirror.pt").exists())
        self.assertTrue(list(job.inputs_dir.glob("input_video.*"))[0].exists())

    def test_run_infer_by_job_id_and_download(self):
        pt_bytes = self._dummy_pt_bytes()
        files = [
            ("real_pt", ("real.pt", pt_bytes, "application/octet-stream")),
            ("mirror_pt", ("mirror.pt", pt_bytes, "application/octet-stream")),
            ("vitpose_real", ("vitpose_real.pt", pt_bytes, "application/octet-stream")),
            ("vitpose_mirror", ("vitpose_mirror.pt", pt_bytes, "application/octet-stream")),
            ("video", ("test_video.mp4", b"fake", "video/mp4")),
        ]
        create_res = self.client.post("/jobs", files=files)
        job_id = create_res.json()["job_id"]

        # Mock run_inference to create a fake inference_results.pkl
        def fake_run_inference(config, bypass_refit=False):
            output_dir = Path(config["visualization"]["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            pkl_path = output_dir / "inference_results.pkl"
            pkl_path.write_bytes(b"dummy pkl content")
            return str(pkl_path)

        with patch("inference.run_inference", side_effect=fake_run_inference):
            infer_res = self.client.post(f"/jobs/{job_id}/infer?wait=true", json={"bypass_refit": True})
            self.assertEqual(infer_res.status_code, 200)
            self.assertEqual(infer_res.json()["status"], "inference_done")

            download_res = self.client.get(f"/jobs/{job_id}/download/inference_pkl")
            self.assertEqual(download_res.status_code, 200)
            self.assertEqual(download_res.content, b"dummy pkl content")

    def test_pipeline_1_click_endpoint(self):
        pt_bytes = self._dummy_pt_bytes()
        files = [
            ("real_pt", ("real.pt", pt_bytes, "application/octet-stream")),
            ("mirror_pt", ("mirror.pt", pt_bytes, "application/octet-stream")),
            ("vitpose_real", ("vitpose_real.pt", pt_bytes, "application/octet-stream")),
            ("vitpose_mirror", ("vitpose_mirror.pt", pt_bytes, "application/octet-stream")),
            ("video", ("test_video.mp4", b"fake", "video/mp4")),
        ]

        def fake_run_inference(config, bypass_refit=False):
            output_dir = Path(config["visualization"]["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            pkl_path = output_dir / "inference_results.pkl"
            pkl_path.write_bytes(b"pipeline pkl content")
            return str(pkl_path)

        with patch("inference.run_inference", side_effect=fake_run_inference):
            response = self.client.post("/jobs/pipeline", files=files)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, b"pipeline pkl content")


if __name__ == "__main__":
    unittest.main()
