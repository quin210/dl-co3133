"""Verify recorded M1 results and replay checkpoints using repository-local files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

import nbformat
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'notebooks/01_eda_fashion_mnist.ipynb'


class Classifier(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        layers, width = [nn.Flatten()], 784
        for next_width in hidden:
            layers.extend([nn.Linear(width, next_width), nn.ReLU()])
            width = next_width
        layers.append(nn.Linear(width, 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images)


def verify():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    assert len(code) == 44
    assert [c.execution_count for c in code] == list(range(1, 45))
    assert all(o.output_type != 'error' for c in code for o in c.outputs)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '<notebook>', 'exec')
        else:
            for target in re.findall(r'\]\(([^)]+)\)', cell.source):
                if not target.startswith(('https://', 'http://', '#')):
                    assert (NOTEBOOK.parent / target.split('#')[0]).exists(), target

    batch = ROOT / notebook.metadata.last_full_run.batch_directory
    assert batch.resolve().is_relative_to(ROOT)
    protocol = json.loads((batch / 'protocol.json').read_text())
    source_hash = hashlib.sha256(json.dumps(
        [(c.cell_type, c.source) for c in notebook.cells], ensure_ascii=False).encode()).hexdigest()
    assert source_hash == protocol['notebook_source_sha256']
    arrays = {}
    for name, record in protocol['data_files'].items():
        packed = (ROOT / 'data/FashionMNIST/raw' / (name + '.gz')).read_bytes()
        assert hashlib.md5(packed).hexdigest() == record['md5']
        data = gzip.decompress(packed)
        assert hashlib.sha256(data).hexdigest() == record['raw_sha256']
        dimensions = data[3]
        shape = tuple(np.frombuffer(data, dtype='>u4', count=dimensions, offset=4))
        arrays[name] = np.frombuffer(data, dtype=np.uint8, offset=4+4*dimensions).reshape(shape)
    images = arrays['train-images-idx3-ubyte']
    labels = arrays['train-labels-idx1-ubyte']
    with np.load(batch / 'split_indices.npz') as split:
        train, val = split['train'], split['validation']
    assert (len(train), len(val)) == (54000, 6000)
    assert np.intersect1d(train, val).size == 0
    np.testing.assert_array_equal(np.sort(np.r_[train, val]), np.arange(60000))
    np.testing.assert_array_equal(np.bincount(labels[train]), np.full(10, 5400))
    np.testing.assert_array_equal(np.bincount(labels[val]), np.full(10, 600))
    for key, indices in [('train', train), ('validation', val)]:
        assert hashlib.sha256(indices.tobytes()).hexdigest() == protocol['split_hashes'][key]
    # Independent chunked sums verify the histogram-fitted normalization.
    total = square_total = 0.0
    count = 0
    for chunk in np.array_split(train, 54):
        scaled = images[chunk].astype(np.float64) / 255
        total += scaled.sum()
        square_total += np.square(scaled).sum()
        count += scaled.size
    mean = total / count
    std = np.sqrt(square_total / count - mean**2)
    np.testing.assert_allclose([mean, std], list(protocol['normalization'].values()), rtol=0, atol=1e-12)
    mean, std = protocol['normalization']['mean'], protocol['normalization']['std']
    val_images = torch.from_numpy((images[val].astype(np.float32) / np.float32(255)
                                   - np.float32(mean)) / np.float32(std)).unsqueeze(1)
    val_labels = torch.from_numpy(labels[val].astype(np.int64))
    torch.set_num_threads(protocol['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    records = json.loads((batch / 'results.json').read_text())
    assert {(r['config']['name'], r['config']['seed']) for r in records} == {
        (name, seed) for name in ['Linear', 'MLP64', 'MLP256'] for seed in [42, 7, 123]}
    checks = []
    for result in records:
        folder = batch / result['artifact_folder']
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        assert checkpoint['protocol'] == protocol
        assert checkpoint['config'] == result['config']
        model = Classifier(result['config']['hidden'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        predictions, loss_sum = [], 0.0
        with torch.inference_mode():
            for start in range(0, len(val), protocol['batch_size']):
                xb = val_images[start:start+protocol['batch_size']]
                yb = val_labels[start:start+protocol['batch_size']]
                logits = model(xb)
                loss_sum += nn.functional.cross_entropy(logits, yb).item() * len(yb)
                predictions.append(logits.argmax(1).numpy())
        predictions = np.concatenate(predictions)
        with np.load(folder / 'validation_predictions.npz') as saved:
            np.testing.assert_array_equal(saved['indices'], val)
            np.testing.assert_array_equal(saved['labels'], labels[val])
            np.testing.assert_array_equal(saved['predictions'], predictions)
        measured = dict(loss=loss_sum/len(val), accuracy=accuracy_score(labels[val], predictions),
                        macro_f1=f1_score(labels[val], predictions, average='macro', labels=range(10)))
        for key, value in measured.items():
            np.testing.assert_allclose(value, result['validation'][key], rtol=0, atol=1e-12)
        matrix = confusion_matrix(labels[val], predictions, labels=range(10))
        np.testing.assert_array_equal(matrix, np.loadtxt(folder/'confusion_counts.csv', delimiter=',', dtype=int))
        class_report = classification_report(labels[val], predictions, labels=list(range(10)),
            target_names=protocol['class_names'], zero_division=0, output_dict=True)
        assert class_report == json.loads((folder/'class_metrics.json').read_text())
        history = result['history']
        assert len(history) == result['stopped_epoch']
        assert all(np.isfinite(v) for h in history for group in ['train', 'validation'] for v in h[group].values())
        best = 1 + int(np.argmax([h['validation']['macro_f1'] for h in history]))
        assert best == result['best_epoch'] == checkpoint['epoch']
        assert len(history) == protocol['max_epochs'] or len(history)-best == protocol['patience']
        assert sum(p.numel() for p in model.parameters()) == result['parameters']
        checks.append(dict(model=result['config']['name'], seed=result['config']['seed'],
                           predictions_exact=True, metrics_match=True, selected_epoch=best))
    replays = json.loads((batch/'replay.json').read_text())
    assert len(replays) == len(checks) == 9
    assert all(r['prediction_parity'] and r['metric_parity'] for r in replays)
    assert protocol['official_test_evaluated'] is False
    report = dict(passed=True, notebook='notebooks/01_eda_fashion_mnist.ipynb',
        notebook_sha256=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(), notebook_source_sha256=source_hash,
        batch_directory=batch.relative_to(ROOT).as_posix(), code_cells_executed=len(code),
        training_runs=len(records), plots=sum('image/png' in o.get('data', {}) for c in code for o in c.outputs),
        checks=['Source identity', 'Dataset checksums', 'Split coverage and disjointness',
                'Training-only normalization', 'Checkpoint predictions, loss, accuracy and macro-F1',
                'Confusion matrices and per-class reports', 'Checkpoint selection and parameter counts'],
        checkpoint_replays=checks, environment=protocol['environment'],
        execution='Fresh Python namespace; repository-local cached data; CPU with one PyTorch thread.',
        untested=['Live download', 'Clean package installation', 'Jupyter kernel and live widget callbacks'],
        official_test_evaluated=False)
    (ROOT/'reports/M1_VERIFICATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Verified: 44 code cells, nine fresh runs, nine independent checkpoint replays.')
    return report, records


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    verify()
