export SSL_CERT_FILE=$(python -m certifi)  
LD_PRELOAD=/usr/lib/libstdc++.so.6 uv run python src/agent.py console