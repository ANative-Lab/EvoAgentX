"""Load exact staged trusted AFlow code without filesystem/import-name coupling."""

import importlib.util
import re
import sys
import types
from pathlib import Path

from ...prompts.optimizers.aflow_optimizer import WORKFLOW_TEMPLATE
from .types import digest


def prepare_graph(graph, prompt, generated=False):
    """Normalize the known prompt import before hashing and staging the source."""
    if not isinstance(graph, str) or not isinstance(prompt, str) or not graph.strip():
        raise ValueError('proposal must contain graph and prompt source strings')
    source = WORKFLOW_TEMPLATE.format(graph=graph) if generated else graph
    source = re.sub(r'^\s*import\s+[\w.]+\.prompt\s+as\s+prompt_custom\s*$',
                    'from . import prompt as prompt_custom', source, flags=re.MULTILINE)
    prompt = prompt.replace('prompt_custom.', '') if generated else prompt
    compile(source, '<aflow staged graph>', 'exec')
    compile(prompt, '<aflow staged prompt>', 'exec')
    return source, prompt


def load_staged_graph(store, trial_id):
    """Import a verified immutable generation using a unique private package.

    Workflows remain trusted user code. This loader supplies the same public
    Workflow constructor and does not claim execution isolation or sandboxing.
    """
    _, _, manifest = store.read_artifacts(trial_id)
    directory = store.path / manifest['directory']
    namespace = '_aflow_trial_' + digest([str(store.path.resolve()), trial_id, manifest['digest']])
    graph_name = namespace + '.graph'
    if graph_name in sys.modules:
        return sys.modules[graph_name].Workflow
    package = types.ModuleType(namespace)
    package.__path__ = [str(directory)]
    package.__package__ = namespace
    sys.modules[namespace] = package
    loaded = [namespace]
    try:
        for name in ('prompt', 'graph'):
            module_name = namespace + '.' + name
            spec = importlib.util.spec_from_file_location(module_name, directory / f'{name}.py')
            if spec is None or spec.loader is None:
                raise ImportError('unable to load staged AFlow artifacts')
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            loaded.append(module_name)
            spec.loader.exec_module(module)
            setattr(package, name, module)
        workflow = sys.modules[graph_name].Workflow
        if not callable(workflow):
            raise TypeError('staged graph has no callable Workflow')
        return workflow
    except BaseException:
        for module_name in loaded:
            sys.modules.pop(module_name, None)
        raise


def publish_round(store, trial_id, root):
    """Export one verified generation for existing round readers and test callers."""
    import os
    import tempfile

    from .store import sync_directory
    graph, prompt, _ = store.read_artifacts(trial_id)
    trial = store.trial(trial_id)
    directory = Path(root) / f"round_{trial['round']}"
    directory.mkdir(parents=True, exist_ok=True)
    # These files are compatibility exports. The immutable generation in the
    # reliable store remains authoritative if publication is interrupted.
    for name, content in (('graph.py', graph), ('prompt.py', prompt), ('__init__.py', '')):
        fd, temporary = tempfile.mkstemp(prefix='.' + name, dir=directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    sync_directory(directory)
