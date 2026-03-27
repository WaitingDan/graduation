import os
import csv
import argparse
import sys

# wrapper to generate visuals from preds CSV
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.gradcam_cnn_models import generate_gradcam
from utils.vit_attention_rollout import generate_vit_rollout


def read_preds(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # ensure types
            r['true_idx'] = int(r['true_idx'])
            r['pred_idx'] = int(r['pred_idx'])
            r['prob'] = float(r['prob'])
            rows.append(r)
    return rows


def select_examples(rows, n=5):
    # high confidence correct
    correct = [r for r in rows if r['true_idx'] == r['pred_idx']]
    correct_sorted = sorted(correct, key=lambda x: -x['prob'])
    high_conf_correct = correct_sorted[:n]

    # low confidence incorrect
    incorrect = [r for r in rows if r['true_idx'] != r['pred_idx']]
    low_conf_incorrect = sorted(incorrect, key=lambda x: x['prob'])[:n]

    # high confidence incorrect (high prob but wrong)
    high_conf_incorrect = sorted(incorrect, key=lambda x: -x['prob'])[:n]

    return high_conf_correct, low_conf_incorrect, high_conf_incorrect


def call_gradcam(image_path, model_name, weights=None, out_dir=None):
    saved = generate_gradcam(
        image_path=image_path,
        model_name=model_name,
        weight_path=weights,
        out_dir=out_dir,
    )
    return saved


def call_vit_rollout(image_path, weights=None, out_dir=None):
    out_file = None
    if out_dir:
        out_file = os.path.join(out_dir, os.path.basename(image_path).replace('.', '_') + '_vit.png')
    saved = generate_vit_rollout(
        image_path=image_path,
        weight_path=weights,
        output_path=out_file,
    )
    return saved


def run_visuals(model, csv_path, n=3, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(ROOT_DIR, 'outputs', 'visualizations', 'attention', 'default_eval')

    os.makedirs(out_dir, exist_ok=True)

    # support csv paths given relative to project root
    if not os.path.exists(csv_path):
        alt = os.path.join(ROOT_DIR, csv_path)
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"Predictions CSV not found: '{csv_path}'. Try running evaluation first or provide the full path (e.g. outputs/evaluation/default_eval/preds_vit.csv relative to project root).")

    rows = read_preds(csv_path)
    high_conf_correct, low_conf_incorrect, high_conf_incorrect = select_examples(rows, n=n)

    print('Selected: high_conf_correct', len(high_conf_correct), 'low_conf_incorrect', len(low_conf_incorrect), 'high_conf_incorrect', len(high_conf_incorrect))

    # process each selected image and record manifest
    manifest = []
    groups = [
        ('high_conf_correct', high_conf_correct),
        ('low_conf_incorrect', low_conf_incorrect),
        ('high_conf_incorrect', high_conf_incorrect),
    ]

    for tag, group in groups:
        for r in group:
            img_path = r['filepath']
            prob = r['prob']
            true_idx = r['true_idx']
            pred_idx = r['pred_idx']
            base = os.path.splitext(os.path.basename(img_path))[0]
            if model in ('resnet', 'vgg'):
                saved = call_gradcam(img_path, model, out_dir=out_dir)
                if saved and os.path.exists(saved):
                    new_name = f"{tag}_{base}_t{true_idx}_p{pred_idx}_{prob:.3f}.png"
                    new_path = os.path.join(out_dir, new_name)
                    try:
                        os.replace(saved, new_path)
                        saved = new_path
                    except Exception:
                        pass
            else:
                saved = call_vit_rollout(img_path, out_dir=out_dir)
                if saved and os.path.exists(saved):
                    new_name = f"{tag}_{base}_t{true_idx}_p{pred_idx}_{prob:.3f}_vit.png"
                    new_path = os.path.join(out_dir, new_name)
                    try:
                        os.replace(saved, new_path)
                        saved = new_path
                    except Exception:
                        pass

            manifest.append({
                'tag': tag,
                'filepath': img_path,
                'saved_visual': saved,
                'true_idx': true_idx,
                'pred_idx': pred_idx,
                'prob': prob,
            })

    # write manifest
    manifest_path = os.path.join(out_dir, 'visuals_manifest.csv')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['tag','filepath','saved_visual','true_idx','pred_idx','prob'])
        writer.writeheader()
        for m in manifest:
            writer.writerow(m)

    print('Done. Visuals saved to', out_dir)
    print('Manifest:', manifest_path)
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['resnet', 'vgg', 'vit', 'vit_fusion'], required=True)
    parser.add_argument('--csv', default=os.path.join(ROOT_DIR, 'outputs', 'preds_resnet.csv'))
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--out_dir', default=os.path.join(ROOT_DIR, 'outputs', 'visualizations', 'attention', 'default_eval'))
    args = parser.parse_args()

    run_visuals(model=args.model, csv_path=args.csv, n=args.n, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
