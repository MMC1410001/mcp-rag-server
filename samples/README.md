# samples/

Public-domain content used by `demo.py` so the server can be exercised without any private data.

| File | Purpose |
|---|---|
| `orbital-mechanics-primer.md` | A short technical document. Chunked, embedded and indexed by the demo. |

## Bringing your own documents

`ingestion.load_document()` accepts either a local path or a publicly shared Google Drive URL:

```
https://drive.google.com/file/d/<FILE_ID>/view                 # single file
https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing # whole folder
```

Supported types: `.pdf`, `.docx`, `.txt`, `.md`, and images (`.png`, `.jpg`) which are read
through Claude's vision API. Anything else in a folder is skipped and reported in the summary.
