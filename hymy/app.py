from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hymy.storage import load_config, save_config
from hymy.zsxq_service import CrawlManager


BASE_DIR = Path(__file__).resolve().parent
manager = CrawlManager()
app = FastAPI(title="HYMY ZSXQ Console")
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")


class ConfigPayload(BaseModel):
    authorization: str = Field(default="")
    user_agent: str = Field(default="")
    group_id: str = Field(default="")
    topics_url: str = Field(default="")
    scope: str = Field(default="")
    crawl_mode: str = Field(default="after_baseline")
    window_start_time: str = Field(default="")
    window_end_time: str = Field(default="")
    max_new_topics_per_run: int = Field(default=50)
    auto_export: bool = Field(default=True)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "app" / "templates" / "index.html")


@app.get("/api/config")
def get_config():
    return load_config()


@app.post("/api/config")
def update_config(payload: ConfigPayload):
    saved = save_config(payload.model_dump())
    return {"ok": True, "config": saved}


@app.get("/api/status")
def get_status():
    return manager.get_runtime_status()


@app.post("/api/test-connection")
def test_connection():
    try:
        return manager.test_connection()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/crawl")
def start_crawl():
    result = manager.start_crawl()
    if not result["ok"]:
        raise HTTPException(status_code=409, detail=result["message"])
    return result
