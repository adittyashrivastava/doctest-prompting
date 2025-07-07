#!/usr/bin/env python3

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading NLTK punkt_tab tokenizer...")
nltk.download('punkt_tab')
print("✅ NLTK punkt_tab downloaded successfully!")