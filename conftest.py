"""
conftest.py (root)
==================
La sola presenza di questo file nella radice del progetto fa sì che pytest
aggiunga la cartella del repository a `sys.path`, così `import app...` funziona
sia in locale sia su CI senza dover installare il pacchetto.
"""
