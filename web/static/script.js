const API_URL = "http://127.0.0.1:8000/api";

async function uploadFile() {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    if (!file) {
        alert("Please select a file first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            getDataAnalysis();
        } else {
            alert("Upload failed.");
        }
    } catch (error) {
        console.error("Error:", error);
    }
}

async function getDataAnalysis() {
    try {
        const response = await fetch(`${API_URL}/analyze`, { method: 'POST' });
        const data = await response.json();

        // Populate Analysis Section
        const analysisDiv = document.getElementById('analysisContent');
        let html = `<h3 class="font-bold">Rows: ${data.rows}, Columns: ${data.columns}</h3>`;

        html += `<h4 class="mt-2 font-semibold">Column Info:</h4><ul class="list-disc pl-5">`;
        for (const [col, dtype] of Object.entries(data.dtypes)) {
            html += `<li>${col} (${dtype}) - Missing: ${data.missing_values[col]}</li>`;
        }
        html += `</ul>`;

        // Fill Target Select
        const targetSelect = document.getElementById('targetSelect');
        targetSelect.innerHTML = "";
        data.column_names.forEach(col => {
            const option = document.createElement("option");
            option.value = col;
            option.text = col;
            targetSelect.appendChild(option);
        });

        analysisDiv.innerHTML = html;
        document.getElementById('analysisSection').classList.remove('hidden');

        getPlots();

    } catch (error) {
        console.error("Error analyzing:", error);
    }
}

async function getPlots() {
    try {
        const response = await fetch(`${API_URL}/plot`, { method: 'POST' });
        const plots = await response.json();

        const plotDiv = document.getElementById('plotContent');
        plotDiv.innerHTML = "";

        for (const [name, imgBase64] of Object.entries(plots)) {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${imgBase64}`;
            img.className = "w-full rounded shadow";
            plotDiv.appendChild(img);
        }
        document.getElementById('visualizationSection').classList.remove('hidden');
        document.getElementById('modelingSection').classList.remove('hidden');

    } catch (error) {
        console.error("Error getting plots:", error);
    }
}

async function generateCode() {
    const target = document.getElementById('targetSelect').value;
    const formData = new FormData();
    formData.append("target", target);

    try {
        const response = await fetch(`${API_URL}/modeling`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        document.getElementById('codeContent').textContent = data.code;

    } catch (error) {
        console.error("Error generating code:", error);
    }
}
