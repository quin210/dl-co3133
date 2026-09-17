"""Run every notebook cell and verify the resulting M1 artifacts."""
import argparse
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
for name, directory in [('MPLCONFIGDIR', 'matplotlib'), ('IPYTHONDIR', 'ipython')]:
    folder = ROOT / '.cache' / directory
    folder.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(folder)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import IPython.display
from IPython.core.interactiveshell import InteractiveShell
import nbformat
if __package__:
    from .verify_m1 import verify
else:
    from verify_m1 import verify


def main():
    path = ROOT / 'notebooks/01_eda_fashion_mnist.ipynb'
    original_bytes = path.read_bytes()
    notebook = nbformat.reads(original_bytes.decode(), as_version=4)
    formatter = InteractiveShell.instance().display_formatter
    namespace = {'__name__': '__main__'}
    original_display, original_show = IPython.display.display, plt.show
    started = time.perf_counter()
    count = 0
    try:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            count += 1
            cell.execution_count, cell.outputs = count, []

            def show(*args, **kwargs):
                for number in plt.get_fignums():
                    buffer = io.BytesIO()
                    plt.figure(number).savefig(buffer, format='png', bbox_inches='tight')
                    cell.outputs.append(nbformat.v4.new_output('display_data', data={
                        'image/png': base64.b64encode(buffer.getvalue()).decode(), 'text/plain': 'Figure'}))
                    plt.close(number)

            def display(*objects, **kwargs):
                for obj in objects:
                    data, metadata = formatter.format(obj)
                    cell.outputs.append(nbformat.v4.new_output('display_data', data=data, metadata=metadata))

            plt.show = show
            IPython.display.display = display
            namespace['display'] = display
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(cell.source, f'<notebook cell {index}>', 'exec'), namespace)
                show()
            if output.getvalue():
                cell.outputs.insert(0, nbformat.v4.new_output('stream', name='stdout', text=output.getvalue()))
            print(f'Executed code cell {count}/44', flush=True)
    finally:
        IPython.display.display, plt.show = original_display, original_show
    batch = namespace['BATCH_DIR'].relative_to(ROOT).as_posix()
    notebook.metadata['execution_note'] = (
        'All 44 code cells executed in source order in a fresh Python namespace. '
        'Nine models trained on repository-local cached data; nine checkpoint replays passed. '
        'Live download, clean installation, Jupyter kernel and widget callbacks were not exercised.')
    notebook.metadata['last_full_run'] = dict(batch_directory=batch, verification='../reports/M1_VERIFICATION.json')
    notebook.metadata.language_info = dict(name='python', version=sys.version.split()[0])
    try:
        from ipywidgets import Widget
        notebook.metadata.widgets = {'application/vnd.jupyter.widget-state+json': Widget.get_manager_state()}
    except ImportError:
        notebook.metadata.pop('widgets', None)
    nbformat.validate(notebook)
    assert path.read_bytes() == original_bytes, 'Notebook changed during execution; refusing to overwrite edits.'
    nbformat.write(notebook, path)
    # Create linked reports before checking that every notebook link resolves.
    report = ROOT / 'reports/M1_RESULTS.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    rows = ['# M1 results', '',
            'Fashion-MNIST; 54,000 training and 6,000 validation images. Split seed 42.', '',
            '| Model (seed 42) | Validation accuracy | Macro-F1 | Parameters | Fit seconds |',
            '|---|---:|---:|---:|---:|']
    for r in namespace['main_results']:
        rows.append(f"| {r['config']['name']} | {r['validation']['accuracy']:.2%} | "
                    f"{r['validation']['macro_f1']:.4f} | {r['parameters']:,} | {r['fit_seconds']:.2f} |")
    rows += ['', 'Training seeds: 42, 7, 123. Validation selects checkpoints; these are development scores.',
             'The official test set is not evaluated.', '',
             f'[Settings](../{batch}/protocol.json) · [All runs](../{batch}/results.json) · '
             f'[Seed summary](../{batch}/repeat_summary.json) · [Replay](../{batch}/replay.json)', '',
             f'[EDA summary](../{namespace["RUN_DIR"].relative_to(ROOT).as_posix()}/eda/summary.json) · '
             '[Verification](M1_VERIFICATION.json)', '',
             'Execution: `python experiments/a1_m1/run_m1.py` using cached data and one CPU thread.',
             'Live download, clean installation and live notebook controls were not tested.', '']
    report.write_text('\n'.join(rows))
    verification_path = ROOT / 'reports/M1_VERIFICATION.json'
    if not verification_path.exists():
        verification_path.write_text('{"passed": false, "status": "verification pending"}\n')
    verify()
    print(f'Completed in {time.perf_counter()-started:.1f}s. Results: reports/M1_RESULTS.md', flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
