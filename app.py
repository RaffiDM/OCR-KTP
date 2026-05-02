from flask import Flask, request, jsonify
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json
import re
import os
import io
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

app = Flask(__name__)

# ─── Load Model (once at startup) ────────────────────────────────────────
print("⏳ Memuat model...")
processor = DonutProcessor.from_pretrained("model-ktp-lokal")
model = VisionEncoderDecoderModel.from_pretrained("model-ktp-lokal")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Use device: {device}")
model.to(device)
model.eval()
print("✅ Model Ready!\n")


# ─── Core OCR Function ────────────────────────────────────────────────────────
def ocr_ktp(image: Image.Image) -> dict:
    """Accepts a PIL Image object and returns a dict of OCR results."""

    image = image.convert("RGB")
    image = image.resize((960, 720), Image.LANCZOS)

    TASK_PROMPT = "<s_iitcdip>"

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    result = processor.token2json(sequence)
    result = {k: v for k, v in result.items() if v and str(v).strip() != ""}

    return result


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    """Check if the server and model are ready."""
    return jsonify({
        "status": "ok",
        "device": device,
        "model": "model-ktp-lokal"
    }), 200


@app.route("/ocr/ktp", methods=["POST"])
def ocr_ktp_endpoint():
    """
    OCR KTP from uploaded file.

    Request  : multipart/form-data  →  field "image" (jpg/jpeg/png)
    Response : JSON OCR results
    """
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "The field 'image' was not found. Submit an image with the key 'image'."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file was selected."
        }), 400

    allowed_extensions = {"jpg", "jpeg", "png"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed_extensions:
        return jsonify({
            "success": False,
            "error": f"The file format '{ext}' is not supported. Use JPG or PNG."
        }), 415

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        hasil = ocr_ktp(image)

        return jsonify({
            "success": True,
            "filename": file.filename,
            "data": hasil
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/ocr/ktp/batch", methods=["POST"])
def ocr_ktp_batch_endpoint():
    print("Files received:", request.files)
    print("Form data:", request.form)

    files = request.files.getlist("images")

    if not files:
        return jsonify({
            "success": False,
            "error": f"The field 'images' was not found or is empty. "
                     f"Files diterima: {list(request.files.keys())}"
        }), 400

    valid_files = [f for f in files if f.filename != ""]
    if not valid_files:
        return jsonify({
            "success": False,
            "error": "Semua file tidak memiliki nama."
        }), 400

    results = []
    allowed_extensions = {"jpg", "jpeg", "png"}

    for file in valid_files:
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

        if ext not in allowed_extensions:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": f"Format '{ext}' is not supported. Use JPG or PNG."
            })
            continue

        try:
            image_bytes = file.read()
            if not image_bytes:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File is empty or could not be read."
                })
                continue

            image = Image.open(io.BytesIO(image_bytes))
            hasil = ocr_ktp(image)

            results.append({
                "filename": file.filename,
                "success": True,
                "data": hasil
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    success_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - success_count

    return jsonify({
        "success": True,
        "total": len(results),
        "success_count": success_count,
        "failed_count": failed_count,
        "results": results
    }), 200


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)