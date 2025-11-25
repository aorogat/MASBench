import pkgutil
import inspect
import importlib

MODULES = [
    "langchain.tools",
    "langchain_community.tools",
    "langchain_experimental.tools",
]

def list_tools():
    found_tools = []

    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        print(f"\n=== Scanning {module_name} ===")

        # walk all submodules
        for loader, name, ispkg in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
            try:
                submod = importlib.import_module(name)
            except Exception:
                continue

            # inspect attributes for tool-like objects
            for obj_name, obj in inspect.getmembers(submod):
                if inspect.isclass(obj) or inspect.isfunction(obj):
                    # Check if it looks like a tool
                    if "tool" in obj_name.lower() or "search" in obj_name.lower() or "loader" in obj_name.lower():
                        found_tools.append((obj_name, name))

        for t, m in found_tools:
            print(f"{t:40s}  <--  {m}")

    return found_tools


if __name__ == "__main__":
    tools = list_tools()
    print("\nTotal tools found:", len(tools))
