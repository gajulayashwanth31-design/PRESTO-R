# PRESTO-R Evaluation Dashboard

This dashboard reads the existing project outputs in `../results` and the architecture graphic in `../output/images`. It does not modify the report, graphs, or evaluation data.

## Run

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\streamlit_app.py
```

The app has an overview, a gallery of saved PNG graphs, selectable CSV evaluation outputs, and filtered episode-level data with CSV downloads. The sidebar filters topology, scenario, and method.

If setting up a separate environment, install the packages listed in `dashboard/requirements.txt` first.
