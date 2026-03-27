import os


DEFAULT_ROBUSTNESS_ROOT = os.path.join('outputs', 'robustness')
DEFAULT_EXPERIMENT_NAME = 'keypart_experiments'


def resolve_experiment_root(output_subdir=None, experiment_name=DEFAULT_EXPERIMENT_NAME):
    """Resolve experiment root path relative to repository root.

    If output_subdir is provided, it is used directly for backward compatibility.
    Otherwise, the standardized root is outputs/robustness/<experiment_name>.
    """
    if output_subdir:
        return output_subdir
    return os.path.join(DEFAULT_ROBUSTNESS_ROOT, experiment_name)


def get_robustness_layout(output_subdir=None, experiment_name=DEFAULT_EXPERIMENT_NAME):
    root = resolve_experiment_root(output_subdir=output_subdir, experiment_name=experiment_name)
    return {
        'root': root,
        'runs': os.path.join(root, 'runs'),
        'metrics': os.path.join(root, 'metrics'),
        'ranking': os.path.join(root, 'ranking'),
        'plots': os.path.join(root, 'plots'),
        'reports': os.path.join(root, 'reports'),
        'visuals': os.path.join(root, 'visuals'),
    }


def ensure_layout_dirs(root_dir, layout):
    for rel in layout.values():
        os.makedirs(os.path.join(root_dir, rel), exist_ok=True)
