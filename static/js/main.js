// Global JS helpers

function copyToClipboard(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const text = el.innerText || el.textContent;
    navigator.clipboard.writeText(text).then(() => {
        showToast("Copied to clipboard!");
    }).catch(err => {
        alert("Failed to copy: " + err);
    });
}

function showToast(message) {
    const toast = document.createElement("div");
    toast.className = "position-fixed bottom-0 end-0 p-3";
    toast.style.zIndex = "9999";
    toast.innerHTML = `
        <div class="toast align-items-center text-white bg-primary border-0 show" role="alert">
            <div class="d-flex">
                <div class="toast-body"><i class="bi bi-check-circle me-2"></i>${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// Quick Parse Form Handler on Dashboard
document.addEventListener("DOMContentLoaded", function() {
    const quickForm = document.getElementById("quickParseForm");
    if (!quickForm) return;

    quickForm.addEventListener("submit", async function(e) {
        e.preventDefault();
        const fileInput = document.getElementById("quickFile");
        if (!fileInput.files.length) {
            alert("Please select a file to parse.");
            return;
        }

        const formData = new FormData(quickForm);
        const loading = document.getElementById("quickParseLoading");
        const resultDiv = document.getElementById("quickParseResult");
        const btn = document.getElementById("quickParseBtn");

        loading.classList.remove("d-none");
        resultDiv.classList.add("d-none");
        btn.disabled = true;

        try {
            const resp = await fetch("/parse", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();
            if (data.success) {
                resultDiv.innerHTML = `
                    <div class="alert alert-success d-flex justify-content-between align-items-center">
                        <div>
                            <i class="bi bi-check-circle-fill me-2"></i>
                            <strong>Successfully Parsed!</strong> (${data.metadata.markdown_len} chars extracted)
                        </div>
                        <div class="btn-group btn-group-sm">
                            <a href="/download/${data.metadata.doc_id}/md" class="btn btn-dark">Download .md</a>
                            <a href="/rag" class="btn btn-outline-dark">Query RAG</a>
                        </div>
                    </div>
                `;
                resultDiv.classList.remove("d-none");
                quickForm.reset();
            } else {
                resultDiv.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="bi bi-exclamation-triangle-fill me-2"></i>
                        Parsing error: ${data.error || "Unknown error"}
                    </div>
                `;
                resultDiv.classList.remove("d-none");
            }
        } catch (err) {
            resultDiv.innerHTML = `<div class="alert alert-danger">Network error: ${err.message}</div>`;
            resultDiv.classList.remove("d-none");
        } finally {
            loading.classList.add("d-none");
            btn.disabled = false;
        }
    });
});

