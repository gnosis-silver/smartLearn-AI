# Day 2 Deployment

## URLs
- Frontend: https://smart-learn-me3lr6961-gnosis4.vercel.app
- Backend health: https://smartlearn-ai-production-acf8.up.railway.app/health
- Backend docs: https://smartlearn-ai-production-acf8.up.railway.app/docs

## Source
- Repository: gnosis-silver/smartLearn-AI (fork of yangzhr1/smartLearn-AI)
- Deployed branch / merge target: main
- Merged commit: 817f77a
- Latest deployment commit: d98ce27
- Pull Request: #1 (feature/day2-lite → main)

## Root Directories
- Railway: repository root (Dockerfile with COPY smartlearn-backend/ paths)
- Vercel: smartlearn-frontend

## Environment variable names
- Railway: OPENROUTER_API_KEY, ALLOWED_ORIGINS
- Vercel: VITE_API_URL

## Acceptance results
- /health: pass
- Upload: pass
- Known /chat + citations: pass (Transformer document, Page 1 citation)
- Unknown question: pass (honest refusal, empty citations)
- CORS restart + re-upload recovery: pass

## Known limitations
- Railway restart clears in-memory uploaded/chat state; re-upload is expected.
- No database, object storage, or authentication (Day 2 scope).
- Repository-root Dockerfile used as a workaround for Railway UI not exposing Root Directory setting.
