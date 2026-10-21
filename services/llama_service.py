import os
import json
import time
import uuid
import datetime
from pathlib import Path
from llama_cloud import LlamaCloud
import config

class LlamaService:
    def __init__(self, api_key=None):
        self.api_key = api_key or config.LLAMA_CLOUD_API_KEY
        self._init_client()

    def _init_client(self):
        if self.api_key:
            self.client = LlamaCloud(api_key=self.api_key)
        else:
            self.client = None

    def update_api_key(self, api_key):
        self.api_key = api_key
        self._init_client()

    def parse_document(self, file_path, tier="agentic_plus", version="latest", expand=None, custom_doc_id=None):
        """
        Parses a document using LlamaCloud parsing API.
        Returns document metadata and paths to output .md and .txt files.
        """
        if not self.client:
            raise ValueError("LlamaCloud API key is not configured. Please provide a valid API key in settings or .env.")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = custom_doc_id or f"doc_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        if expand is None:
            expand = ["markdown_full", "text_full"]

        # 1. Upload file
        with open(file_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="parse")

        # 2. Submit parsing job and get result
        result = self.client.parsing.parse(
            file_id=file_obj.id,
            tier=tier,
            version=version,
            expand=expand
        )

        markdown_content = result.markdown_full or ""
        text_content = result.text_full or ""

        # Persist results
        md_file = config.PARSED_DIR / f"{doc_id}.md"
        txt_file = config.PARSED_DIR / f"{doc_id}.txt"
        meta_file = config.PARSED_DIR / f"{doc_id}.meta.json"

        md_file.write_text(markdown_content, encoding="utf-8")
        txt_file.write_text(text_content, encoding="utf-8")

        metadata = {
            "doc_id": doc_id,
            "original_filename": file_path.name,
            "file_size": file_path.stat().st_size,
            "tier": tier,
            "version": version,
            "file_id": getattr(file_obj, "id", None),
            "parsed_at": datetime.datetime.now().isoformat(),
            "md_file": str(md_file.name),
            "txt_file": str(txt_file.name),
            "markdown_len": len(markdown_content),
            "text_len": len(text_content),
            "status": "COMPLETED"
        }

        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def extract_document(self, file_path, data_schema, tier="agentic", parse_tier="agentic", cite_sources=True, confidence_scores=True, custom_doc_id=None):
        """
        Extracts structured JSON data according to a schema using LlamaCloud extraction API.
        """
        if not self.client:
            raise ValueError("LlamaCloud API key is not configured. Please provide a valid API key in settings or .env.")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = custom_doc_id or f"ext_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        if isinstance(data_schema, str):
            data_schema = json.loads(data_schema)

        # 1. Upload file
        with open(file_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="extract")

        # 2. Submit extract job
        job = self.client.extract.create(
            file_input=file_obj.id,
            configuration={
                "data_schema": data_schema,
                "tier": tier,
                "extraction_target": "per_doc",
                "parse_tier": parse_tier,
                "cite_sources": cite_sources,
                "confidence_scores": confidence_scores
            }
        )

        # 3. Poll until terminal state
        start_time = time.time()
        while job.status not in ("COMPLETED", "FAILED", "CANCELLED"):
            if time.time() - start_time > 300: # 5 min timeout
                raise TimeoutError("Extraction job timed out after 5 minutes.")
            time.sleep(2)
            job = self.client.extract.get(job.id)

        if job.status != "COMPLETED":
            raise RuntimeError(f"Extract job {job.id} ended in {job.status}: {getattr(job, 'error_message', 'Unknown error')}")

        json_file = config.EXTRACTED_DIR / f"{doc_id}.json"
        meta_file = config.EXTRACTED_DIR / f"{doc_id}.meta.json"

        extracted_result = getattr(job, "extract_result", {})
        json_file.write_text(json.dumps(extracted_result, indent=2), encoding="utf-8")

        field_metadata = {}
        if getattr(job, "extract_metadata", None) and getattr(job.extract_metadata, "field_metadata", None):
            doc_meta = getattr(job.extract_metadata.field_metadata, "document_metadata", {}) or {}
            field_metadata = {k: str(v) for k, v in doc_meta.items()}

        metadata = {
            "doc_id": doc_id,
            "original_filename": file_path.name,
            "tier": tier,
            "parse_tier": parse_tier,
            "cite_sources": cite_sources,
            "confidence_scores": confidence_scores,
            "extracted_at": datetime.datetime.now().isoformat(),
            "json_file": str(json_file.name),
            "schema_used": data_schema,
            "field_metadata": field_metadata,
            "status": "COMPLETED"
        }

        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {
            "metadata": metadata,
            "result": extracted_result,
            "field_metadata": field_metadata
        }

    def list_parsed_documents(self):
        """Returns a list of all parsed documents with their metadata."""
        docs = []
        for meta_file in sorted(config.PARSED_DIR.glob("*.meta.json"), key=os.path.getmtime, reverse=True):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                docs.append(data)
            except Exception:
                continue
        return docs

    def get_parsed_document(self, doc_id):
        """Gets full parsed content for a document."""
        meta_file = config.PARSED_DIR / f"{doc_id}.meta.json"
        md_file = config.PARSED_DIR / f"{doc_id}.md"
        txt_file = config.PARSED_DIR / f"{doc_id}.txt"

        if not meta_file.exists():
            return None

        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        md_content = md_file.read_text(encoding="utf-8") if md_file.exists() else ""
        txt_content = txt_file.read_text(encoding="utf-8") if txt_file.exists() else ""

        return {
            "metadata": metadata,
            "markdown": md_content,
            "text": txt_content
        }

    def list_extracted_documents(self):
        """Returns a list of all structured extractions."""
        extractions = []
        for meta_file in sorted(config.EXTRACTED_DIR.glob("*.meta.json"), key=os.path.getmtime, reverse=True):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                json_file = config.EXTRACTED_DIR / f"{data['doc_id']}.json"
                if json_file.exists():
                    data["content"] = json.loads(json_file.read_text(encoding="utf-8"))
                extractions.append(data)
            except Exception:
                continue
        return extractions

    def delete_document(self, doc_id):
        """Deletes all files associated with a parsed or extracted document."""
        for folder in [config.PARSED_DIR, config.EXTRACTED_DIR]:
            for p in folder.glob(f"{doc_id}*"):
                try:
                    p.unlink()
                except Exception:
                    pass
        return True

llama_service = LlamaService()

