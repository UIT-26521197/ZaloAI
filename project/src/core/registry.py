import json
from pathlib import Path


class ToolRegistry:
    def __init__(self, catalog: dict):
        self.apps = {}
        for app_name, app in catalog["apps"].items():
            tools = []
            for item in app["tools"]:
                tool = item.get("function", item)
                name = tool["name"]
                if "." not in name:
                    name = f"{app_name}.{name}"
                tools.append(dict(tool, name=name))
            self.apps[app_name] = tools

    @classmethod
    def load(cls, path: Path):
        return cls(json.loads(path.read_text(encoding="utf-8-sig")))

    def for_context(self, context: dict) -> dict[str, dict]:
        installed = context.get("installed_apps")
        if isinstance(installed, list) and all(isinstance(app, str) for app in installed):
            installed = dict.fromkeys(installed)
        if not isinstance(installed, dict):
            raise ValueError("Expected context.installed_apps to contain the available apps")
        allowed = {}
        for name, app in installed.items():
            declared = app.get("tools") if isinstance(app, dict) else None
            if declared is not None:
                declared = {
                    item if "." in item else f"{name}.{item}"
                    for item in (entry if isinstance(entry, str) else entry["name"] for entry in declared)
                }
            for tool in self.apps.get(name, []):
                if declared is None or tool["name"] in declared:
                    allowed[tool["name"]] = tool
        # System tools are independent of the user's installed apps.
        for tool in self.apps.get("sys", []):
            allowed[tool["name"]] = tool
        return allowed
