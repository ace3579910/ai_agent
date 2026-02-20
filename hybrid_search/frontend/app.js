// Use the current location's protocol and host for API calls
const API_URL = `${window.location.protocol}//${window.location.host}`;

async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const status = document.getElementById('uploadStatus');
    
    if (fileInput.files.length === 0) {
        status.textContent = "Please select a file first.";
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    status.textContent = "Uploading and indexing...";

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            status.textContent = `Success! ${result.filename} indexed.`;
            // refresh available sources
            loadSources();
        } else {
            // Try chunked upload fallback for large files
            console.warn('Initial upload failed, attempting chunked upload');
            const ok = await chunkedUpload(file, status);
            if (ok) {
                status.textContent = `Success! ${file.name} indexed (chunked).`;
                loadSources();
            } else {
                const text = await response.text().catch(() => '');
                status.textContent = `Upload failed. ${text}`;
            }
        }
    } catch (error) {
        console.error("Error:", error);
        status.textContent = "Error uploading file.";
    }
}

async function chunkedUpload(file, statusEl, chunkSize = 5 * 1024 * 1024) {
    // chunkSize default 5MB
    const total = Math.ceil(file.size / chunkSize);
    const filename = file.name;
    const tempStatus = statusEl;
    try {
        for (let i = 0; i < total; i++) {
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, file.size);
            const blob = file.slice(start, end);
            const fd = new FormData();
            fd.append('filename', filename);
            fd.append('index', String(i));
            fd.append('total', String(total));
            fd.append('file', blob, filename);

            tempStatus.textContent = `Uploading chunk ${i+1}/${total}...`;
            const resp = await fetch(`${API_URL}/upload_chunk`, { method: 'POST', body: fd });
            if (!resp.ok) {
                const txt = await resp.text().catch(() => '');
                console.error('Chunk upload failed', i, txt);
                return false;
            }
            const resJson = await resp.json().catch(() => null);
            // if assembled, break
            if (resJson && resJson.assembled) {
                tempStatus.textContent = 'All chunks uploaded and assembled.';
            }
        }
        return true;
    } catch (e) {
        console.error('Chunked upload error', e);
        return false;
    }
}

async function performSearch() {
    const query = document.getElementById('queryInput').value;
    const resultsSection = document.getElementById('resultsSection');
    const aiAnswer = document.getElementById('aiAnswer');
    const keyPointsList = document.getElementById('keyPointsList');
    const sourcesList = document.getElementById('sourcesList');
    const confidenceBadge = document.getElementById('confidenceBadge');
    
    if (!query) return;

    // Show loading state implies clearing previous results
    resultsSection.classList.remove('hidden');
    aiAnswer.innerHTML = "<p>Analyzing documents...</p>";
    keyPointsList.innerHTML = "";
    sourcesList.innerHTML = "";

    try {
        const source = document.getElementById('sourceSelect')?.value || "";
        const payload = { query: query };
        if (source) payload.source = source;

        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        // Render Answer
        aiAnswer.innerHTML = marked.parse(data.answer);

        // Render Key Points
        keyPointsList.innerHTML = data.key_points.map(point => `<li>${point}</li>`).join('');

        // Render Confidence
        confidenceBadge.textContent = `Confidence: ${data.confidence}`;
        
        // Render Sources
        sourcesList.innerHTML = data.sources.map(source => `
            <div class="source-item">
                <span class="source-meta">Source: ${source.source} | Score: ${source.score.toFixed(4)}</span>
                <p>"${source.content}..."</p>
            </div>
        `).join('');

        // Auto-save user query and assistant answer to persistent memory
        try {
            // save user query
            fetch(`${API_URL}/memory/add`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: query, role: 'user' })
            }).catch(e => console.warn('Failed to store user memory', e));

            // save assistant answer
            fetch(`${API_URL}/memory/add`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: data.answer || '', role: 'assistant' })
            }).catch(e => console.warn('Failed to store assistant memory', e));
        } catch (e) {
            console.warn('Memory save error', e);
        }

    } catch (error) {
        console.error("Error:", error);
        aiAnswer.textContent = "An error occurred during analysis.";
    }
}

async function loadSources() {
    try {
        const res = await fetch(`${API_URL}/documents`);
        if (!res.ok) return;
        const docs = await res.json();
        const sel = document.getElementById('sourceSelect');
        if (!sel) return;
        // keep an option for All
        sel.innerHTML = `<option value="">All sources</option>` + docs.map(d => `<option value="${d}">${d}</option>`).join('');
    } catch (e) {
        console.error('Failed to load sources', e);
    }
}

window.addEventListener('load', () => {
    loadSources();
});
