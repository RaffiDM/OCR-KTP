# 🪪 KTP OCR API

A Flask-based REST API for extracting text from Indonesian National ID Cards (KTP) using a fine-tuned [Donut](https://huggingface.co/docs/transformers/model_doc/donut) (Document Understanding Transformer) model.

---

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
  - [GET /health](#get-health)
  - [POST /ocr/ktp](#post-ocrktp)
  - [POST /ocr/ktp/batch](#post-ocrktpbatch)
- [Response Format](#response-format)
- [Error Handling](#error-handling)
- [Project Structure](#project-structure)

---

## ✨ Features

- 🔍 OCR extraction from a single KTP image
- 📦 Batch processing for multiple KTP images at once, with per-file success/failure tracking (`success_count`, `failed_count`)
- 🖼️ Supports JPG and PNG image formats
- ⚡ Automatic GPU acceleration (falls back to CPU if unavailable)
- 🩺 Health check endpoint to verify server and model status

---

## 📦 Requirements

- Python 3.8+
- A local Donut model saved in the model-ktp-lokal/ directory

Install Python dependencies:

```bash
pip install flask transformers==4.57.6 pillow==11.3.0 torch==2.3.1
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
flask
transformers==4.57.6
pillow==11.3.0
torch==2.3.1
```

---

## 🚀 Installation

1. **Clone or download** this repository.

2. **Install dependencies** as shown above.

---

## ▶️ Running the Server

```bash
python app.py
```

The server will start on `http://0.0.0.0:5000`. You should see:

```
⏳ Memuat model...
✅ Use device: cuda   # or cpu
✅ Model Ready!
```

---

## 📡 API Endpoints

### `GET /health`

Check if the server and model are ready.

**Request:**
```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "model": "model-ktp-lokal"
}
```

---

### `POST /ocr/ktp`

Extract KTP data from a **single** image.

**Request:**

| Type            | Key     | Value            |
|-----------------|---------|------------------|
| `multipart/form-data` | `image` | JPG or PNG file |

**Example (cURL):**
```bash
curl -X POST http://localhost:5000/ocr/ktp \
  -F "image=@/path/to/ktp.jpg"
```

**Example (Python):**
```python
import requests

with open("ktp.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:5000/ocr/ktp",
        files={"image": f}
    )

print(response.json())
```

**Success Response (`200 OK`):**
```json
{
  "success": true,
  "filename": "ktp.jpg",
  "data": {
    "nik": "1234567890123456",
    "nama": "JOHN DOE",
    "tempat_lahir": "JAKARTA",
    "tanggal_lahir": "01-01-1990",
    "jenis_kelamin": "LAKI-LAKI",
    "alamat": "JL. CONTOH NO. 1",
    "rt_rw": "001/002",
    "kel_desa": "KELURAHAN",
    "kecamatan": "KECAMATAN",
    "agama": "ISLAM",
    "status_perkawinan": "BELUM KAWIN",
    "pekerjaan": "KARYAWAN SWASTA",
    "kewarganegaraan": "WNI"
  }
}
```

> **Note:** The fields returned in `data` depend on what the model successfully reads from the image. Empty or undetected fields are automatically excluded from the response.

---

### `POST /ocr/ktp/batch`

Extract KTP data from **multiple** images in a single request.

**Request:**

| Type            | Key      | Value                         |
|-----------------|----------|-------------------------------|
| `multipart/form-data` | `images` | One or more JPG/PNG files |

**Example (cURL):**
```bash
curl -X POST http://localhost:5000/ocr/ktp/batch \
  -F "images=@ktp1.jpg" \
  -F "images=@ktp2.png"
```

**Example (Python):**
```python
import requests

files = [
    ("images", open("ktp1.jpg", "rb")),
    ("images", open("ktp2.png", "rb")),
]
response = requests.post("http://localhost:5000/ocr/ktp/batch", files=files)
print(response.json())
```

**Success Response (`200 OK`):**
```json
{
  "success": true,
  "total": 2,
  "success_count": 2,
  "failed_count": 0,
  "results": [
    {
      "filename": "ktp1.jpg",
      "success": true,
      "data": { "nik": "...", "nama": "..." }
    },
    {
      "filename": "ktp2.png",
      "success": true,
      "data": { "nik": "...", "nama": "..." }
    }
  ]
}
```

**Partial failure example** — if one file fails, the rest still process:
```json
{
  "success": true,
  "total": 2,
  "success_count": 1,
  "failed_count": 1,
  "results": [
    {
      "filename": "ktp1.jpg",
      "success": true,
      "data": { "nik": "...", "nama": "..." }
    },
    {
      "filename": "document.pdf",
      "success": false,
      "error": "Format 'pdf' is not supported. Use JPG or PNG."
    }
  ]
}
```

> **Note:** The top-level `success` is always `true` as long as the request itself was valid. Check `failed_count` and per-item `success` fields to detect individual file errors.

---

## ⚠️ Error Handling

The API returns structured JSON errors. Common error responses:

| HTTP Status | Cause |
|-------------|-------|
| `400` | Missing `image`/`images` field, no file selected, or all files have empty filenames |
| `415` | Unsupported file format (only JPG and PNG are accepted) |
| `500` | Internal server error during OCR processing |

**Error response format:**
```json
{
  "success": false,
  "error": "Description of the error."
}
```

For batch requests, per-file errors (unsupported format, empty file, processing failure) are reported individually inside the `results` array **without failing the entire request**. The overall request only returns `400` if the `images` field is missing entirely or all submitted files have no filename.

---

## 📁 Project Structure

```
.
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── model-ktp-lokal/        # Local Donut model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── ...
└── README.md
```

---

## 📝 Notes

- Images are automatically resized to **960×720 pixels** before processing.
- The model uses **greedy decoding** (`num_beams=1`) for fast inference.
- For production deployment, consider using a WSGI server such as **Gunicorn**:
  ```bash
  gunicorn -w 1 -b 0.0.0.0:5000 app:app
  ```
  > Use only 1 worker (`-w 1`) to avoid loading the model multiple times into memory.

---

## 📄 License

This project is intended for internal/private use. Please ensure compliance with applicable data privacy regulations (e.g., Indonesia's Personal Data Protection Law / UU PDP) when processing KTP data.