import warnings
warnings.filterwarnings("ignore")   # suppresses ALL deprecation warnings

import os
os.environ["PYTHONWARNINGS"] = "ignore"
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights   # new-style weights — no warnings
from PIL import Image
import io
import base64
from pathlib import Path
import numpy as np
import uuid
from bird_info import bird_database, get_bird_by_name
from audio_model import predict_audio

# ── NEW: GPS + Database imports ───────────────────────────────────────────────
from gps_service import (
    apply_gps_boost,
    get_nearby_species_summary,
    get_hotspots_nearby,
)
from database import (
    init_db,
    save_sighting,
    get_sightings,
    get_lifelist,
    get_sighting_stats,
    delete_sighting,
)

app = Flask(__name__)

ALLOWED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_audio")
os.makedirs(TEMP_DIR, exist_ok=True)

model              = None
device             = None
class_names        = None
general_classifier = None


# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP BANNER
# ─────────────────────────────────────────────────────────────────────────────
def print_banner():
    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║                                                                  ║")
    print("  ║    ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ██████╗ ██████╗  ║")
    print("  ║    ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔══██╗██╔══██╗ ║")
    print("  ║    ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██████╔╝██║  ██║ ║")
    print("  ║    ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══██╗██║  ██║ ║")
    print("  ║    ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝██████╔╝██████╔╝ ║")
    print("  ║    ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═════╝  ║")
    print("  ║                                                                  ║")
    print("  ║          Deep Learning Bird Species Identification System        ║")
    print("  ║                      v2.0  —  GPS Edition                        ║")
    print("  ║                                                                  ║")
    print("  ╠══════════════════════════════════════════════════════════════════╣")
    print("  ║                                                                  ║")
    print("  ║   Developed by   :  Vignesh Yadala                              ║")
    print("  ║   Project        :  NeuroBird  v2.0  (GPS Edition)              ║")
    print("  ║   Model          :  ResNet-50  (Fine-tuned on CUB-200-2011)     ║")
    print("  ║   Dataset        :  CUB-200-2011  |  200 Species                ║")
    print("  ║   Framework      :  PyTorch + Flask                             ║")
    print("  ║   Audio Engine   :  librosa  (MFCC + Chroma + Spectral)         ║")
    print("  ║   GPS Engine     :  eBird API  (Cornell Lab)                    ║")
    print("  ║   Database       :  SQLite  (Personal Sighting Logbook)         ║")
    print("  ║   Academic Year  :  2022 - 2026                                 ║")
    print("  ║                                                                  ║")
    print("  ╠══════════════════════════════════════════════════════════════════╣")
    print("  ║                                                                  ║")
    print("  ║   Department  :  Computer Science and Engineering               ║")
    print("  ║   College     :  Bharath Institute of Higher Education          ║")
    print("  ║                  and Research                                   ║")
    print("  ║                                                                  ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    global model, device, class_names, general_classifier

    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = Path('bird-dataset/train')
    class_names  = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

    # Bird classifier (custom trained)
    model    = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 200)
    checkpoint = torch.load(
        'trained-models/best_bird_classifier.pth', map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # General ImageNet classifier (input guard)
    general_classifier = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    general_classifier = general_classifier.to(device)
    general_classifier.eval()

    print(f"  ✔  Model loaded!              Device  : {device}")
    print(f"  ✔  Species loaded             Count   : {len(class_names)}")
    print(f"  ✔  General classifier ready   (input guard active)")
    print(f"  ✔  Bird database loaded       Species : {len(bird_database)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def get_bird_info_from_prediction(species_name):
    species_clean = species_name.strip().title()
    bird_info     = get_bird_by_name(species_clean)
    if bird_info:
        return {
            'common_name'             : bird_info['common_name'],
            'scientific_name'         : bird_info['scientific_name'],
            'description'             : bird_info['description'],
            'physical_characteristics': bird_info['physical_characteristics'],
            'habitat'                 : bird_info['habitat'],
            'behavior'                : bird_info['behavior'],
            'diet'                    : bird_info['diet'],
            'breeding'                : bird_info['breeding'],
            'conservation_status'     : bird_info['conservation_status'],
            'interesting_facts'       : ' • '.join(bird_info['interesting_facts']),
            'distribution'            : bird_info['distribution'],
            'size'                    : f"{bird_info['size']['length']} length, "
                                         f"{bird_info['size']['wingspan']} wingspan",
            'lifespan'                : bird_info['lifespan'],
            'call'                    : bird_info['call'],
        }
    return {
        'common_name'             : species_clean,
        'scientific_name'         : 'Information not available',
        'description'             : f'The {species_clean} is a bird species. '
                                     'Detailed information is not yet in our database.',
        'physical_characteristics': 'Information not available in database.',
        'habitat'                 : 'Information not available in database.',
        'behavior'                : 'Information not available in database.',
        'diet'                    : 'Information not available in database.',
        'breeding'                : 'Information not available in database.',
        'conservation_status'     : 'Information not available in database.',
        'interesting_facts'       : 'Detailed information coming soon!',
        'distribution'            : 'Various regions',
        'size'                    : 'Information not available',
        'lifespan'                : 'Information not available',
        'call'                    : 'Information not available',
    }


def is_bird_or_person(image_tensor):
    with torch.no_grad():
        outputs       = general_classifier(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top10_prob, top10_indices = torch.topk(probabilities, 10)

    bird_classes      = list(range(7, 101))
    person_classes    = list(range(0, 3))
    non_living_things = list(range(400, 900))
    top_indices       = top10_indices[0].cpu().numpy()
    top_probs         = top10_prob[0].cpu().numpy()
    bird_prob         = 0.0
    person_prob       = 0.0
    max_any_prob      = top_probs[0]

    for idx, prob in zip(top_indices, top_probs):
        if idx in bird_classes:
            bird_prob   += prob
        if idx in person_classes:
            person_prob += prob

    if person_prob > 0.15:
        return False, "person", person_prob
    if top_indices[0] in non_living_things and max_any_prob > 0.3:
        return False, "object", max_any_prob
    return True, "potential_bird", bird_prob


def predict_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    image        = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    is_valid, object_type, obj_confidence = is_bird_or_person(image_tensor)
    if not is_valid:
        if object_type == "person":
            return {
                'error'      : True,
                'message'    : 'HUMAN DETECTED! Please upload a bird image.',
                'object_type': 'person',
                'confidence' : round(obj_confidence * 100, 2),
            }
        return {
            'error'      : True,
            'message'    : 'NO BIRD DETECTED! Please upload a clear bird photograph.',
            'object_type': 'object',
            'confidence' : 0,
        }

    with torch.no_grad():
        outputs       = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top5_prob, top5_indices = torch.topk(probabilities, 5)

    results = []
    for i in range(5):
        species       = class_names[top5_indices[0][i]]
        species_clean = species.split('.')[-1].replace('_', ' ')
        confidence    = top5_prob[0][i].item() * 100
        results.append({'species': species_clean, 'confidence': round(confidence, 2)})

    top_species    = results[0]['species']
    bird_info      = get_bird_info_from_prediction(top_species)
    top_confidence = results[0]['confidence']

    warning      = None
    warning_type = None
    if top_confidence < 20:
        warning      = "Very low confidence — species may not be in our database."
        warning_type = "warning"
    elif top_confidence < 40:
        warning      = "Low confidence — try a clearer image."
        warning_type = "warning"

    return {
        'error'       : False,
        'predictions' : results,
        'bird_info'   : bird_info,
        'warning'     : warning,
        'warning_type': warning_type,
    }


def get_bird_image_b64(species_name):
    import glob
    test_data_path = Path('test-data')
    if not test_data_path.exists():
        return None

    species_words = species_name.strip().lower().split()
    all_images    = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        all_images.extend(glob.glob(str(test_data_path / '**' / ext), recursive=True))
        all_images.extend(glob.glob(str(test_data_path / ext)))

    if not all_images:
        return None

    best_match = None
    for img_path in all_images:
        img_lower = img_path.lower().replace('\\', '/').replace('-', '_').replace(' ', '_')
        if all(word in img_lower for word in species_words):
            best_match = img_path
            break

    if not best_match:
        best_match = all_images[0]

    try:
        with open(best_match, 'rb') as f:
            img_bytes = f.read()
        ext  = os.path.splitext(best_match)[1].lower()
        mime = 'image/png' if ext == '.png' else 'image/jpeg'
        b64  = base64.b64encode(img_bytes).decode('utf-8')
        return f'data:{mime};base64,{b64}'
    except Exception as e:
        print(f"  [WARN] Could not load bird image: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  ORIGINAL ROUTES  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        image_bytes = file.read()
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            file.stream.seek(0)
            image_bytes = file.read()
        except Exception:
            return jsonify({'error': 'Invalid image file. Please upload JPG, PNG, etc.'}), 400

        result       = predict_image(image_bytes)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        if result.get('error'):
            return jsonify({
                'success'    : False,
                'error'      : result['message'],
                'object_type': result.get('object_type'),
                'image'      : f'data:image/jpeg;base64,{image_base64}',
            })
        return jsonify({
            'success'     : True,
            'predictions' : result['predictions'],
            'bird_info'   : result['bird_info'],
            'image'       : f'data:image/jpeg;base64,{image_base64}',
            'warning'     : result.get('warning'),
            'warning_type': result.get('warning_type'),
        })
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@app.route('/predict-audio', methods=['POST'])
def predict_audio_route():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400

    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in ALLOWED_AUDIO:
        return jsonify({'error': f'Unsupported format: {ext}. Use .wav .mp3 .flac .ogg .m4a'}), 400

    tmp_name = os.path.join(TEMP_DIR, f"tmp_audio_{uuid.uuid4().hex}{ext}")
    try:
        audio_file.save(tmp_name)
        if not os.path.exists(tmp_name) or os.path.getsize(tmp_name) == 0:
            return jsonify({'error': 'Audio file could not be saved.'}), 500

        try:
            predictions = predict_audio(tmp_name, top_k=5)
        except RuntimeError as e:
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            return jsonify({'error': f'Audio processing failed: {str(e)}'}), 500

        bird_info  = None
        bird_image = None
        if predictions:
            bird_info  = get_bird_info_from_prediction(predictions[0]['species'])
            bird_image = get_bird_image_b64(predictions[0]['species'])

        return jsonify({
            'success'    : True,
            'source'     : 'audio',
            'predictions': predictions,
            'bird_info'  : bird_info,
            'image'      : bird_image,
        })
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass


@app.route('/predict-combined', methods=['POST'])
def predict_combined():
    has_image = 'file'  in request.files
    has_audio = 'audio' in request.files
    if not has_image and not has_audio:
        return jsonify({'error': 'Provide image and/or audio file'}), 400

    img_scores       = {}
    audio_scores     = {}
    bird_info_result = None

    if has_image:
        img_bytes = request.files['file'].read()
        try:
            img_result = predict_image(img_bytes)
            if not img_result.get('error'):
                for p in img_result['predictions']:
                    img_scores[p['species']] = p['confidence']
                bird_info_result = img_result.get('bird_info')
        except Exception as e:
            print(f"  [WARN] Combined image prediction failed: {e}")

    if has_audio:
        aud_file = request.files['audio']
        ext      = os.path.splitext(aud_file.filename)[1].lower()
        aud_tmp  = os.path.join(TEMP_DIR, f"tmp_aud_{uuid.uuid4().hex}{ext}")
        aud_file.save(aud_tmp)
        try:
            aud_preds    = predict_audio(aud_tmp, top_k=5)
            audio_scores = {p['species']: p['confidence'] for p in aud_preds}
        except Exception as e:
            print(f"  [WARN] Combined audio prediction failed: {e}")
        finally:
            if os.path.exists(aud_tmp):
                try:
                    os.remove(aud_tmp)
                except Exception:
                    pass

    if not img_scores and not audio_scores:
        return jsonify({'error': 'Both image and audio analysis failed.'}), 500

    all_species = set(img_scores) | set(audio_scores)
    fused = {
        sp: round(0.60 * img_scores.get(sp, 0) + 0.40 * audio_scores.get(sp, 0), 2)
        for sp in all_species
    }
    top5 = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:5]

    if top5 and not bird_info_result:
        bird_info_result = get_bird_info_from_prediction(top5[0][0])

    return jsonify({
        'success'    : True,
        'source'     : 'combined',
        'predictions': [{'species': s, 'confidence': c} for s, c in top5],
        'bird_info'  : bird_info_result,
        'img_used'   : has_image,
        'audio_used' : has_audio,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  NEW GPS ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/predict-with-location', methods=['POST'])
def predict_with_location():
    """
    Image prediction + GPS confidence boost.
    Form fields: file (image), lat, lng, date (optional)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    lat  = request.form.get('lat',  type=float)
    lng  = request.form.get('lng',  type=float)
    date = request.form.get('date', None)

    if lat is None or lng is None:
        return jsonify({'error': 'lat and lng are required for GPS prediction'}), 400

    try:
        image_bytes = file.read()
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            file.stream.seek(0)
            image_bytes = file.read()
        except Exception:
            return jsonify({'error': 'Invalid image file.'}), 400

        result = predict_image(image_bytes)
        if result.get('error'):
            return jsonify({'success': False, 'error': result['message']})

        # Apply GPS boost
        boosted = apply_gps_boost(result['predictions'], lat, lng, date)

        top_species = boosted[0]['species']
        bird_info   = get_bird_info_from_prediction(top_species)
        image_b64   = base64.b64encode(image_bytes).decode('utf-8')

        # Auto-save to logbook
        try:
            sci_name = bird_info.get('scientific_name', '')
            save_sighting(
                species        = top_species,
                scientific_name= sci_name,
                confidence     = boosted[0]['confidence'],
                source         = 'image+gps',
                lat            = lat,
                lng            = lng,
                location_badge = boosted[0].get('location_badge', ''),
                date_identified= date,
            )
        except Exception as db_err:
            print(f"  [WARN] Could not save sighting: {db_err}")

        return jsonify({
            'success'    : True,
            'source'     : 'image+gps',
            'predictions': boosted,
            'bird_info'  : bird_info,
            'image'      : f'data:image/jpeg;base64,{image_b64}',
            'gps_used'   : True,
            'location'   : {'lat': lat, 'lng': lng},
        })

    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/nearby-species', methods=['GET'])
def nearby_species():
    """
    GET /nearby-species?lat=XX&lng=YY&radius=50
    Returns species recently observed near the user's location.
    """
    lat       = request.args.get('lat',    type=float)
    lng       = request.args.get('lng',    type=float)
    radius_km = request.args.get('radius', type=int, default=50)

    if lat is None or lng is None:
        return jsonify({'error': 'lat and lng query parameters required'}), 400

    try:
        summary = get_nearby_species_summary(lat, lng, radius_km)
        return jsonify({'success': True, **summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/hotspots', methods=['GET'])
def hotspots():
    """
    GET /hotspots?lat=XX&lng=YY
    Returns top birding hotspots near the user.
    """
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)

    if lat is None or lng is None:
        return jsonify({'error': 'lat and lng required'}), 400

    try:
        spots = get_hotspots_nearby(lat, lng)
        formatted = [{
            'name'        : h.get('locName', ''),
            'lat'         : h.get('lat', 0),
            'lng'         : h.get('lng', 0),
            'num_species' : h.get('numSpeciesAllTime', 0),
            'loc_id'      : h.get('locId', ''),
        } for h in spots[:10]]
        return jsonify({'success': True, 'hotspots': formatted, 'count': len(formatted)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/species-likely', methods=['GET'])
def species_likely():
    """
    GET /species-likely?lat=XX&lng=YY&month=4
    Returns species most likely in this area this time of year.
    """
    lat   = request.args.get('lat',   type=float)
    lng   = request.args.get('lng',   type=float)
    month = request.args.get('month', type=int,
                             default=__import__('datetime').date.today().month)

    if lat is None or lng is None:
        return jsonify({'error': 'lat and lng required'}), 400

    try:
        from gps_service import get_nearby_observations, _build_local_name_set
        obs   = get_nearby_observations(lat, lng)
        names = list({o.get('comName', '') for o in obs if o.get('comName')})
        return jsonify({
            'success'      : True,
            'species'      : names[:30],
            'month'        : month,
            'count'        : len(names),
            'api_available': bool(os.getenv('EBIRD_API_KEY', '')),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  SIGHTING LOGBOOK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/my-sightings', methods=['GET', 'POST'])
def my_sightings():
    """
    GET  /my-sightings          — retrieve sighting history
    POST /my-sightings          — save a new sighting manually
    """
    if request.method == 'POST':
        data = request.get_json() or {}
        try:
            sid = save_sighting(
                species         = data.get('species', 'Unknown'),
                scientific_name = data.get('scientific_name', ''),
                confidence      = float(data.get('confidence', 0)),
                source          = data.get('source', 'manual'),
                lat             = data.get('lat'),
                lng             = data.get('lng'),
                location_name   = data.get('location_name', ''),
                notes           = data.get('notes', ''),
                location_badge  = data.get('location_badge', ''),
                date_identified = data.get('date'),
            )
            return jsonify({'success': True, 'sighting_id': sid})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # GET — return paginated list
    limit     = request.args.get('limit',     type=int, default=50)
    species   = request.args.get('species',   default=None)
    date_from = request.args.get('date_from', default=None)
    date_to   = request.args.get('date_to',   default=None)

    rows = get_sightings(limit=limit, species=species,
                         date_from=date_from, date_to=date_to)
    return jsonify({'success': True, 'sightings': rows, 'count': len(rows)})


@app.route('/my-sightings/<int:sighting_id>', methods=['DELETE'])
def delete_sighting_route(sighting_id):
    """DELETE /my-sightings/<id> — remove a sighting."""
    ok = delete_sighting(sighting_id)
    return jsonify({'success': ok})


@app.route('/lifelist', methods=['GET'])
def lifelist():
    """GET /lifelist — return all unique species ever identified."""
    rows = get_lifelist()
    return jsonify({'success': True, 'lifelist': rows, 'count': len(rows)})


@app.route('/sighting-stats', methods=['GET'])
def sighting_stats():
    """GET /sighting-stats — dashboard statistics."""
    stats = get_sighting_stats()
    return jsonify({'success': True, **stats})


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print_banner()

    print("  " + "=" * 66)
    print("    Initialising Database...")
    print("  " + "=" * 66)
    init_db()

    print("  " + "=" * 66)
    print("    Loading NeuroBird Model...")
    print("  " + "=" * 66)
    load_model()

    print("  " + "=" * 66)
    print("    Starting Web Server...")
    print("  " + "=" * 66)
    print()
    print("    >>  Open browser  :  http://127.0.0.1:5000")
    print("    >>  Stop server   :  Press  Ctrl + C")
    print()
    print("    NEW GPS ROUTES:")
    print("    >>  /predict-with-location  POST  image + lat/lng → boosted predictions")
    print("    >>  /nearby-species         GET   lat/lng → local species list")
    print("    >>  /hotspots               GET   lat/lng → top birding spots nearby")
    print("    >>  /species-likely         GET   lat/lng/month → likely species")
    print("    >>  /my-sightings           GET/POST → personal sighting logbook")
    print("    >>  /lifelist               GET   → all unique species identified")
    print("    >>  /sighting-stats         GET   → dashboard statistics")
    print()

    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    app.run(debug=True, use_reloader=False)