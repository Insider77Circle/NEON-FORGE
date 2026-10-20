# main.py
from manager.orchestrator import MetaManager

if __name__ == "__main__":
    forge = MetaManager()
    forge.initialize()
    forge.run_pipeline()
    forge.report()
