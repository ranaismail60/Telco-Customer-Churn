// Point this at your deployed backend URL after Phase 8.
// For local testing, this is correct as-is.
const API_URL = "http://localhost:5000/predict";

document.getElementById("churn-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = new FormData(e.target);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    // Cast numeric-looking fields to numbers, leave the rest as strings
    payload[key] = isNaN(value) || value === "" ? value : Number(value);
  }

  const resultBox = document.getElementById("result");
  resultBox.classList.remove("hidden");
  document.getElementById("risk-level").textContent = "Checking...";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    document.getElementById("risk-level").textContent = data.risk_level;
    document.getElementById("risk-prob").textContent =
      (data.churn_probability * 100).toFixed(1) + "%";

    const list = document.getElementById("reasons-list");
    list.innerHTML = "";
    (data.top_reasons || []).forEach((reason) => {
      const li = document.createElement("li");
      li.textContent = reason;
      list.appendChild(li);
    });
  } catch (err) {
    document.getElementById("risk-level").textContent = "Error — is the backend running?";
    console.error(err);
  }
});
