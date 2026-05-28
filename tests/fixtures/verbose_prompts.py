VERBOSE_SYSTEM_PROMPT = """\
You are a helpful assistant.    You are a helpful assistant.

You should always be polite and professional in your responses.
You should always be polite and professional in your responses.

Please make sure to:
  - Answer questions clearly
  - Answer questions clearly
  - Be concise
  - Be concise

When the user asks a question, think carefully before responding.
When the user asks a question, think carefully before responding.
When the user asks a question, think carefully before responding.

Always respond in the language of the user.
Always respond in the language of the user.
"""

LANGCHAIN_RAG_OUTPUT = """\
<document index="1">
<source>docs/overview.md</source>
<content>
TokenShrink is a library for reducing LLM token usage. It provides lossless compression.
TokenShrink is a library for reducing LLM token usage. It provides lossless compression.
</content>
</document>

<document index="2">
<source>docs/overview.md</source>
<content>
TokenShrink is a library for reducing LLM token usage. It provides lossless compression.
This is the second document with the exact same content repeated verbatim here.
TokenShrink is a library for reducing LLM token usage. It provides lossless compression.
</content>
</document>

<document index="3">
<source>docs/api.md</source>
<content>
The compress_prompt method takes a string and returns a CompressionResult with compressed_text.
Use the select_context method to select the most relevant RAG chunks within a token budget.
</content>
</document>
"""

RAG_CHUNKS = [
    "TokenShrink is a Python library for reducing LLM token usage by 70% using lossless compression techniques.",
    "The library supports Anthropic, OpenAI, and any other LLM SDK through a transparent wrapper pattern.",
    "BM25 scoring is used for relevance-based context selection in RAG pipelines.",
    "The weather in Paris today is partly cloudy with temperatures around 18 degrees Celsius.",
    "Whitespace normalization removes redundant spaces, newlines, and indentation to reduce token count.",
    "The French Revolution began in 1789 and lasted until 1799, fundamentally changing French society.",
    "Semantic deduplication uses MinHash and Jaccard similarity to detect near-duplicate content.",
]

RAG_QUERY = "How does TokenShrink compress prompts and handle RAG context?"
