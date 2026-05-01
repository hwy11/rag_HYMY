from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests


DEFAULT_TIMEOUT = 20
DEFAULT_COUNT = 30
DEFAULT_SCOPE = "all"


@dataclass
class ZsxqConfig:
    authorization: str
    user_agent: str
    group_id: str
    topics_url: str = ""
    scope: str = DEFAULT_SCOPE


class ZsxqClient:
    def __init__(self, config: ZsxqConfig):
        self.config = config

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": self.config.authorization.strip(),
            "User-Agent": self.config.user_agent.strip(),
        }

    def _base_topics_url(self) -> str:
        if self.config.topics_url.strip():
            parsed = urlparse(self.config.topics_url.strip())
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return f"https://api.zsxq.com/v2/groups/{self.config.group_id}/topics"

    def _default_params(self) -> dict[str, str]:
        params = {"scope": DEFAULT_SCOPE, "count": str(DEFAULT_COUNT)}
        if self.config.topics_url.strip():
            parsed = urlparse(self.config.topics_url.strip())
            for key, values in parse_qs(parsed.query).items():
                if values:
                    params[key] = values[-1]
        if self.config.scope.strip():
            params["scope"] = self.config.scope.strip()
        return params

    def fetch_topics(
        self,
        count: int = DEFAULT_COUNT,
        begin_time: str | None = None,
        end_time: str | None = None,
    ) -> list[dict[str, Any]]:
        params = self._default_params()
        params["count"] = str(count)
        if begin_time:
            params["begin_time"] = begin_time
        else:
            params.pop("begin_time", None)
        if end_time:
            params["end_time"] = end_time
        else:
            params.pop("end_time", None)

        response = requests.get(
            self._base_topics_url(),
            headers=self.headers,
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("succeeded", True):
            code = payload.get("code")
            error = payload.get("error") or payload.get("info") or "知识星球接口返回失败"
            raise RuntimeError(f"ZSXQ API error {code}: {error}")
        return payload.get("resp_data", {}).get("topics", []) or []

    def test_connection(self) -> dict[str, Any]:
        topics = self.fetch_topics(count=1)
        return {
            "ok": True,
            "sample_count": len(topics),
            "latest_topic_id": (topics[0].get("topic_id") if topics else ""),
            "latest_topic_time": (topics[0].get("create_time") if topics else ""),
        }
