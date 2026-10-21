import os
import json
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename

import config
from services.llama_service import llama_service
from services.rag_service import rag_service

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max upload

ALLOWED_EXTENSIONS = {"pdf", "docx", "pptx", "txt", "md", "csv", "xlsx", "html", "png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    parsed_docs = llama_service.list_parsed_documents()
    extracted_docs = llama_service.list_extracted_documents()
    return render_template(
        "index.html",
        parsed_count=len(parsed_docs),
        extracted_count=len(extracted_docs),
        recent_parsed=parsed_docs[:5],
        recent_extracted=extracted_docs[:5],
        api_key_configured=bool(llama_service.api_key),
        tiers=config.PARSE_TIERS
    )

@app.route("/parse", methods=["GET", "POST"])
def parse_page():
    if request.method == "POST":
        # Handled via AJAX or regular form
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        tier = request.form.get("tier", config.DEFAULT_PARSE_TIER)
        version = request.form.get("version", config.DEFAULT_PARSE_VERSION)

        try:
            filename = secure_filename(file.filename) or "document.pdf"
            save_path = config.UPLOAD_DIR / filename
            file.save(save_path)

            metadata = llama_service.parse_document(
                file_path=save_path,
                tier=tier,
                version=version
            )
            # Rebuild RAG index
            rag_service.refresh_index()

            return jsonify({
                "success": True,
                "message": f"Successfully parsed {filename}",
                "metadata": metadata
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(e)}), 500

    parsed_docs = llama_service.list_parsed_documents()
    return render_template(
        "parse.html",
        tiers=config.PARSE_TIERS,
        default_tier=config.DEFAULT_PARSE_TIER,
        parsed_docs=parsed_docs,
        api_key_configured=bool(llama_service.api_key)
    )

@app.route("/extract", methods=["GET", "POST"])
def extract_page():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        schema_raw = request.form.get("schema", "{}")
        tier = request.form.get("tier", "agentic")
        parse_tier = request.form.get("parse_tier", "agentic")
        cite_sources = request.form.get("cite_sources", "true").lower() == "true"
        confidence_scores = request.form.get("confidence_scores", "true").lower() == "true"

        try:
            schema_data = json.loads(schema_raw) if schema_raw.strip() else {}
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid JSON Schema: {str(e)}"}), 400

        try:
            filename = secure_filename(file.filename) or "document.pdf"
            save_path = config.UPLOAD_DIR / filename
            file.save(save_path)

            result = llama_service.extract_document(
                file_path=save_path,
                data_schema=schema_data,
                tier=tier,
                parse_tier=parse_tier,
                cite_sources=cite_sources,
                confidence_scores=confidence_scores
            )
            return jsonify({
                "success": True,
                "message": f"Successfully extracted data from {filename}",
                "data": result
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(e)}), 500

    extracted_docs = llama_service.list_extracted_documents()
    return render_template(
        "extract.html",
        extracted_docs=extracted_docs,
        api_key_configured=bool(llama_service.api_key)
    )

@app.route("/rag", methods=["GET", "POST"])
def rag_page():
    parsed_docs = llama_service.list_parsed_documents()
    return render_template(
        "rag.html",
        parsed_docs=parsed_docs,
        has_openai=bool(rag_service.openai_key),
        has_gemini=bool(rag_service.gemini_key)
    )

@app.route("/documents")
def documents_page():
    parsed_docs = llama_service.list_parsed_documents()
    extracted_docs = llama_service.list_extracted_documents()
    return render_template(
        "documents.html",
        parsed_docs=parsed_docs,
        extracted_docs=extracted_docs
    )

@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    if request.method == "POST":
        llama_key = request.form.get("llama_cloud_key", "").strip()
        openai_key = request.form.get("openai_key", "").strip()
        gemini_key = request.form.get("gemini_key", "").strip()

        if llama_key:
            llama_service.update_api_key(llama_key)
            config.LLAMA_CLOUD_API_KEY = llama_key
        if openai_key is not None:
            rag_service.openai_key = openai_key
            config.OPENAI_API_KEY = openai_key
        if gemini_key is not None:
            rag_service.gemini_key = gemini_key
            config.GEMINI_API_KEY = gemini_key

        flash("Settings updated successfully!", "success")
        return redirect(url_for("settings_page"))

    return render_template(
        "settings.html",
        llama_key=llama_service.api_key or "",
        openai_key=rag_service.openai_key or "",
        gemini_key=rag_service.gemini_key or ""
    )

# --- API ENDPOINTS ---

@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"success": False, "error": "Prompt query cannot be empty"}), 400

    doc_ids = data.get("doc_ids", None)
    top_k = int(data.get("top_k", 5))
    provider = data.get("provider", "auto")

    try:
        res = rag_service.query(prompt=prompt, doc_ids=doc_ids, top_k=top_k, provider=provider)
        return jsonify({
            "success": True,
            "result": res
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/document/<doc_id>")
def api_get_document(doc_id):
    doc = llama_service.get_parsed_document(doc_id)
    if not doc:
        return jsonify({"success": False, "error": "Document not found"}), 404
    return jsonify({"success": True, "document": doc})

@app.route("/api/document/<doc_id>", methods=["DELETE"])
def api_delete_document(doc_id):
    success = llama_service.delete_document(doc_id)
    rag_service.refresh_index()
    return jsonify({"success": success})

@app.route("/download/<doc_id>/<file_type>")
def download_file(doc_id, file_type):
    if file_type == "md":
        target = config.PARSED_DIR / f"{doc_id}.md"
        mime = "text/markdown"
    elif file_type == "txt":
        target = config.PARSED_DIR / f"{doc_id}.txt"
        mime = "text/plain"
    elif file_type == "json":
        target = config.EXTRACTED_DIR / f"{doc_id}.json"
        if not target.exists():
            target = config.PARSED_DIR / f"{doc_id}.meta.json"
        mime = "application/json"
    else:
        return "Invalid file type", 400

    if not target.exists():
        return "File not found", 404

    return send_file(target, mimetype=mime, as_attachment=True, download_name=target.name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

