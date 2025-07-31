import papermill as pm

# Liste deiner 6 Gebäudenamen – exakt so, wie sie im Schema stehen
building_names = [
    "Building_1",
    "Building_2",
    "Building_3",
    "Building_4",
    "Building_5",
    "Building_6"
]

# Für jedes Gebäude ein neues, ausgeführtes Notebook erzeugen
for b in building_names:
    pm.execute_notebook(
        'dqn_agent.ipynb',              # Quelldatei
        f'dqn_agent_executed_{b}.ipynb',# Zieldatei für diese Ausführung
        parameters = dict(building_name=b)
    )
    print(f"✓ Notebook für {b} fertig")
    