from __future__ import annotations

import os

import uvicorn

from server.app import app


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
