# Virtual Agent

A voice AI agent application built with [LiveKit Agents](https://docs.livekit.io/agents) featuring a Python backend and React/Next.js frontend.

## Project Structure

```
virtualagent/
├── backend/          # Python-based LiveKit Agent
├── frontend/         # React/Next.js web application
└── commands.md       # Backend command reference
```

---

## Backend

The backend is a Python-based voice AI agent built with LiveKit Agents framework.

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- LiveKit Cloud account or self-hosted LiveKit server

### Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Configure environment variables by copying `.env.example` to `.env.local`:
   ```bash
   cp .env.example .env.local
   ```

4. Fill in the required keys in `.env.local`:
   - `LIVEKIT_URL`
   - `LIVEKIT_API_KEY`
   - `LIVEKIT_API_SECRET`

   Or use the [LiveKit CLI](https://docs.livekit.io/home/cli/cli-setup) for automatic setup:
   ```bash
   lk cloud auth
   lk app env -w -d .env.local
   ```

5. Download required models (first run only):
   ```bash
   uv run python src/agent.py download-files
   ```

### Running the Backend

Run the agent in development mode:

```bash
export SSL_CERT_FILE=$(python -m certifi)
LD_PRELOAD=/usr/lib/libstdc++.so.6 uv run python src/agent.py dev
```

**Alternative modes:**

- **Console mode** (speak directly in terminal):
  ```bash
  uv run python src/agent.py console
  ```

- **Production mode**:
  ```bash
  uv run python src/agent.py start
  ```

---

## Frontend

The frontend is a React/Next.js application providing a web interface for the voice AI agent.

### Prerequisites

- Node.js 18+
- [pnpm](https://pnpm.io/) (Package manager)

### Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   pnpm install
   ```

3. Configure environment variables by copying `.env.example` to `.env.local`:
   ```bash
   cp .env.example .env.local
   ```

4. Fill in the required keys in `.env.local`:
   ```env
   LIVEKIT_API_KEY=your_livekit_api_key
   LIVEKIT_API_SECRET=your_livekit_api_secret
   LIVEKIT_URL=https://your-livekit-server-url
   ```

### Running the Frontend

Start the development server:

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Running the Full Application

To run the complete application, you need to start both the backend and frontend:

1. **Terminal 1 - Start the Backend:**
   ```bash
   cd backend
   export SSL_CERT_FILE=$(python -m certifi)
   LD_PRELOAD=/usr/lib/libstdc++.so.6 uv run python src/agent.py dev
   ```

2. **Terminal 2 - Start the Frontend:**
   ```bash
   cd frontend
   pnpm dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) and start talking to your agent!

---

## Features

- **Voice Interaction**: Real-time voice communication with the AI agent
- **Transcriptions**: Live transcription of conversations
- **Camera/Screen Share**: Video streaming and screen sharing support
- **Virtual Avatars**: Support for virtual avatar integration
- **Theme Support**: Light/dark mode with system preference detection

---
