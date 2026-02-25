import datasets

def doc_to_text(doc: dict) -> str:
    return doc["prompt"]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        return {
            "prompt": doc["prompt"],
            "output": doc["output"],
        }
    return dataset.map(_process_doc)