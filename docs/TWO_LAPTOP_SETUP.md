# PathSnap — Two-Laptop Network Setup & Deployment Guide

This guide explains how to deploy PathSnap across **two Windows laptops on the same Wi-Fi / LAN network**:

- **Laptop 1 (Inference & Evaluation Server):** Runs the FastAPI + Uvicorn backend (`api.main:app`), holds the trained ML model artifacts (`ml/mlp/artifacts/`, `artifacts/baselines/`), road-network trip folders, and server-side evaluation logic.
- **Laptop 2 (Evaluator Frontend Client):** Runs the Streamlit + Folium dashboard (`ui/evaluator_dashboard/app.py`) and communicates with Laptop 1 over HTTP using the `PATHSNAP_API_URL` environment variable.

> **Important:** Do not hardcode IP addresses into source code, and do not assume both laptops share the same Python installation or virtual environment. Each laptop sets up its own lightweight environment.

---

## 1. Connect Both Laptops to the Same Network

1. Connect **Laptop 1** and **Laptop 2** to the **same Wi-Fi SSID, mobile hotspot, or Ethernet LAN** (e.g., a dedicated mobile hotspot avoids university campus Wi-Fi client isolation).
2. On Windows, set the network profile to **Private** (Settings → Network & internet → Wi-Fi → network properties → **Private network**) so Windows Firewall allows local LAN traffic on port `8000`.

---

## 2. Find Laptop 1's Local IPv4 Address

Run the following in **Windows PowerShell on Laptop 1**:

```powershell
ipconfig
```

Look for **Wireless LAN adapter Wi-Fi** (or **Ethernet adapter**) and note the **IPv4 Address**, for example:

```text
IPv4 Address. . . . . . . . . . . : 192.168.1.100
```

Alternatively, print only non-loopback IPv4 addresses directly:

```powershell
Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
  Select-Object InterfaceAlias, IPAddress
```

Let `<LAPTOP_1_IP>` denote this address (e.g., `192.168.1.100`) and `<API_PORT>` denote the chosen port (default `8000`).

---

## 3. Configure Windows Firewall on Laptop 1

Open **PowerShell as Administrator on Laptop 1** and allow inbound TCP traffic on `<API_PORT>` (default `8000`) for Private networks:

```powershell
New-NetFirewallRule `
  -DisplayName "PathSnap FastAPI Backend (TCP 8000)" `
  -Direction Inbound `
  -Protocol TCP `
  -LocalPort 8000 `
  -Action Allow `
  -Profile Private,Domain
```

*(If your campus/hotspot network is marked Public and cannot be changed, temporarily add `Public` to `-Profile`, and remove the rule after evaluation with `Remove-NetFirewallRule -DisplayName "PathSnap FastAPI Backend (TCP 8000)"`.)*

---

## 4. Prepare Python & Start the FastAPI Server on Laptop 1

On **Laptop 1**, navigate to the repository root (`pathsnap-team13`), activate its Python environment, install backend dependencies, point `PATHSNAP_DATA_ROOT` to the dataset directory, and launch Uvicorn bound to **`0.0.0.0`** (all network interfaces):

```powershell
cd path\to\pathsnap-team13

# Create & activate Laptop 1 virtual environment (first time only)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Point to the folder containing train/ and student_test/ (and evaluator_test/ when provided)
$env:PATHSNAP_DATA_ROOT = "C:\path\to\Student folder\Student folder"

# Start FastAPI server listening on all LAN interfaces
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Verify locally on Laptop 1:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Keep the Laptop 1 PowerShell terminal visible during evaluation so evaluators can see live incoming HTTP request logs (`POST /predict -> HTTP 200 ... from <LAPTOP_2_IP>`).

---

## 5. Prepare Python & Connect Laptop 2 to Laptop 1

**Laptop 2 does NOT need PyTorch, scikit-learn model weights, or raw dataset folders.** It only needs Streamlit, Folium, and HTTP client packages:

```powershell
cd path\to\pathsnap-team13

# Create & activate Laptop 2 virtual environment (independent of Laptop 1)
python -m venv .venv_ui
.\.venv_ui\Scripts\Activate.ps1
pip install streamlit streamlit-folium folium requests pandas numpy matplotlib
```

Configure the configurable `PATHSNAP_API_URL` environment variable pointing to Laptop 1:

```powershell
$env:PATHSNAP_API_URL = "http://<LAPTOP_1_IP>:8000"
# Example:
# $env:PATHSNAP_API_URL = "http://192.168.1.100:8000"
```

---

## 6. Test the Network Connection from Laptop 2 Before Launching UI

Run these checks in **PowerShell on Laptop 2**:

```powershell
# 1. Test TCP port reachability
Test-NetConnection -ComputerName <LAPTOP_1_IP> -Port 8000

# 2. Query FastAPI /health endpoint
Invoke-RestMethod -Uri "$env:PATHSNAP_API_URL/health"

# 3. Query available models and trips
Invoke-RestMethod -Uri "$env:PATHSNAP_API_URL/models"
Invoke-RestMethod -Uri "$env:PATHSNAP_API_URL/trips?partition=student_test"
```

You should see `TcpTestSucceeded : True` and `"status": "ok"` with `"models_loaded": true`.

---

## 7. Start the Streamlit Frontend on Laptop 2

```powershell
streamlit run ui/evaluator_dashboard/app.py --server.port 8501
```

Open `http://localhost:8501` in the browser on Laptop 2:
- The sidebar automatically picks up `PATHSNAP_API_URL` (and also lets you edit/test a different `http://<IP>:<PORT>` at runtime without restarting).
- Navigate to **System Information** to verify `"Connected"`, loaded model names, and API version.
- Navigate to **GPS Map Matching**, select a trip and model, and click **Run Map Matching** — you will see live request logs appear immediately on Laptop 1's PowerShell console!

---

## 8. Troubleshooting Common Connection Failures

| Symptom | Likely Cause | Fix |
|---|---|---|
| `Cannot connect to Laptop 1 FastAPI` / `Connection refused` | Uvicorn bound to `127.0.0.1` instead of `0.0.0.0`, or server not running | Restart Laptop 1 with `--host 0.0.0.0 --port 8000` |
| `Test-NetConnection` times out (`TcpTestSucceeded: False`) | Windows Firewall blocking port `8000` OR Wi-Fi AP Client Isolation enabled on campus Wi-Fi | Add the Firewall rule from Section 3; if still blocked on campus Wi-Fi, connect both laptops to a **mobile hotspot** |
| `data_root_available: false` or `0 trips` in `GET /trips` | `PATHSNAP_DATA_ROOT` not set or points to wrong folder level | Set `$env:PATHSNAP_DATA_ROOT` on Laptop 1 to the exact folder directly containing `train/` and `student_test/` |
| `HTTP 404` on `/predict` | Selected `model_name` or `trip_id` does not exist on Laptop 1 | Check `GET /models` and `GET /trips`; click **Test / Refresh Connection** in the Streamlit sidebar |
| Slow first prediction on a trip, fast afterwards | First request extracts spatial KD-tree candidates (~0.5–1.5 s); subsequent requests hit Laptop 1's in-memory cache | Expected behavior — keep `timeout_s = 90s` in `api_client.py` |
| Different Python versions between Laptop 1 & Laptop 2 | Laptop 2 only speaks JSON over HTTP | Safe by design — Laptop 1 and Laptop 2 can run completely different Python versions and virtual environments |
