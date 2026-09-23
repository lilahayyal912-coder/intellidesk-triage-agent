"""
Verifies app/main.py startup error handling without a live Streamlit server.
Tests two scenarios in isolated subprocesses to avoid module-cache interference.
"""
import subprocess, sys, os, textwrap

python = sys.executable

# ── Scenario 1: GROQ_API_KEY present — _CONFIG_ERROR must be None ────────────
code_valid = textwrap.dedent("""
import sys, os, unittest.mock as mock
sys.path.insert(0, 'src')
os.environ['GROQ_API_KEY'] = 'test-valid-key'
import importlib.util
spec = importlib.util.spec_from_file_location('main', 'app/main.py')
mod = importlib.util.module_from_spec(spec)
with mock.patch('streamlit.set_page_config'), \\
     mock.patch('streamlit.cache_resource', lambda **kw: (lambda f: f)):
    spec.loader.exec_module(mod)
assert mod._CONFIG_ERROR is None, f'Got: {mod._CONFIG_ERROR}'
assert mod.GROQ_API_KEY == 'test-valid-key', 'Key not loaded'
print('PASS')
""")

r1 = subprocess.run([python, "-c", code_valid], capture_output=True, text=True,
                    cwd=os.path.abspath("."))
print("Scenario 1 (valid key):", r1.stdout.strip() or f"FAIL – {r1.stderr.strip()[:200]}")

# ── Scenario 2: GROQ_API_KEY absent — _CONFIG_ERROR must be set ──────────────
code_missing = textwrap.dedent("""
import sys, os, unittest.mock as mock
sys.path.insert(0, 'src')
os.environ.pop('GROQ_API_KEY', None)
# Prevent dotenv from loading the real .env so the key stays absent
mock.patch('dotenv.load_dotenv', return_value=False).start()
import importlib.util
spec = importlib.util.spec_from_file_location('main', 'app/main.py')
mod = importlib.util.module_from_spec(spec)
with mock.patch('streamlit.set_page_config'), \\
     mock.patch('streamlit.cache_resource', lambda **kw: (lambda f: f)):
    spec.loader.exec_module(mod)
assert mod._CONFIG_ERROR is not None, 'Expected error string, got None'
assert 'GROQ_API_KEY' in mod._CONFIG_ERROR, f'Missing key name in: {mod._CONFIG_ERROR}'
assert mod.GROQ_API_KEY == '', f'Placeholder not empty: {mod.GROQ_API_KEY!r}'
assert mod.GROQ_BASE_URL == '', 'BASE_URL placeholder not empty'
assert mod.GROQ_MODEL == '', 'MODEL placeholder not empty'
print('PASS – error:', mod._CONFIG_ERROR)
""")

r2 = subprocess.run([python, "-c", code_missing], capture_output=True, text=True,
                    cwd=os.path.abspath("."))
print("Scenario 2 (missing key):", r2.stdout.strip() or f"FAIL – {r2.stderr.strip()[:200]}")
