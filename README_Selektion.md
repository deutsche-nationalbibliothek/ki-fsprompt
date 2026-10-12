# Selektionsbasierte Vorverarbeitung (TOC)

Arbeitsbranch `Preprocessing_selections` · Cezar Harb · Masterarbeit an der DNB

Dieser Branch ergänzt KI-FSPrompt um einen zweiten Weg, wie der Text entsteht, der
ins Few-Shot-Prompting geht. Alles Nachgelagerte (retrieve → complete → map → rank)
bleibt unverändert.

## Fragestellung

Die Arbeit untersucht, **wie die Auswahl des Textes die Qualität der
GND-Sacherschließung beeinflusst**. Bisher bekommt die Pipeline Titeldaten. Die
Idee dahinter kommt aus der intellektuellen Erschließungspraxis: Menschen lesen
nicht das ganze Buch, sondern gezielt Inhaltsverzeichnis, Abstract, Einleitung
und Fazit. Genau diese Teile werden hier maschinell (Docling) extrahiert und
einzeln als Eingabe getestet.

Der erste umgesetzte Fall ist das **Inhaltsverzeichnis (TOC)**. Weitere
Selektionen (Abstract, Einleitung, Fazit, Volltext) folgen nach demselben Muster.

## Grundprinzip

Alle Selektionen nutzen **dieselben Dokumente, dieselben IDs, denselben Split und
denselben Gold-Standard**. Es ändert sich ausschließlich die Spalte `text` — nur so
sind die Varianten überhaupt vergleichbar.

```
corpora/title/{split}.arrow     welche idn zu welchem Split gehört
corpora/ground-truth.arrow      Gold-Standard-Labels pro idn (dieselbe Quelle wie casimir)
corpora/Sections/TOC_Extraction/  ein Docling-TOC pro idn
          |
          v   src/preprocess_selektion.py  (DoclingPartAdapter)
pipelines/preprocess/toc/{train,validate,test}.csv
          |
          v   unverändert
train-Pipeline  ->  predict-Pipeline  ->  casimir-Metriken
```

| Split | Rolle in der Arbeit | Index-Datei (bleibt immer dieselbe) |
|---|---|---|
| train | Retrieval-Pool | `corpora/title/train.arrow` |
| validate | Kalibrierung (XGBoost) | `corpora/title/validate.arrow` |
| test | Evaluation | `corpora/title/test.arrow` |

Das funktioniert, weil alle nachgelagerten Stufen nur vier Spalten lesen:
`text, doc_id, label_ids, label_texts` (siehe `src/data_adapters/constants.py`).
Solange eine neue Selektion dieses Schema schreibt, muss am Rest nichts angefasst
werden.

## Was in diesem Branch neu ist

| Datei | Zweck |
|---|---|
| `src/preprocess_selektion.py` | CLI: eine Docling-Selektion → normalisiertes CSV |
| `src/data_adapters/docling_part_adapter.py` | Adapter: liest ein Dokumentteil-Verzeichnis, hängt Labels an |
| `src/data_adapters/cleaners.py` | Textreinigung, aktuell `none` und `toc` |
| `pipelines/preprocess/dvc.yaml` | neue Stage `preprocess_toc` (foreach über die drei Splits) |
| `pipelines/preprocess/params.yaml` | Parameter der TOC-Selektion |
| `corpora/Sections.dvc` | `dvc import` von `datasets/Sections` aus `aen/txt-praktikum` (TOC, Abstract, Intro) |
| `instructions/instruction_toc.txt` | Prompt-Instruktion für Inhaltsverzeichnisse |
| `src/create_train_collection.py` | Schutz gegen versehentlich überschriebene Weaviate-Collections |

## Getroffene Entscheidungen — und warum

**Labels direkt aus `ground-truth.arrow`, nicht über den prefLabel-Umweg.**
Die Datei enthält GND-URI und Labeltext bereits zusammen und ist dieselbe Quelle,
die auch casimir zur Bewertung heranzieht. Damit kann die Selektion nicht
versehentlich gegen einen anderen Gold-Standard laufen als die Evaluation.

**Kein Header-Entfernen mehr.** Die frühere Version der TOC-Dateien begann mit einem
Docling-Metadatenkopf (`# Inhaltsverzeichnis — …`, `> Quelle: … · Score: …`), der
vor dem Cleaning entfernt wurde. Die Dateien aus `datasets/Sections` enthalten diesen
Kopf nicht mehr, der Inhalt beginnt direkt in Zeile 1. Deshalb wird keine Zeile mehr
vorab gelöscht.

**`cleaning: toc` räumt die Struktur auf.** Aufzählungszeichen, Kapitelnummern und
Seitenangaben `(S. 5)` tragen nichts zur inhaltlichen Erschließung bei, kosten aber
Prompt-Token. Zwei Details sind bewusst eng gefasst: Es werden nur Seitenzahlen in
Ziffern oder römischen Zahlen entfernt, damit `(S. Freud)` stehen bleibt, und
Kapitelnummern nur bis drei Stellen je Ebene, damit eine Überschrift wie
`1848 Revolution` nicht ihren Jahresanfang verliert.

**`empty_policy: drop` mit mitgeschriebenem Index.** Für 13 Dokumente gibt es keine
TOC-Datei — Docling hat dort kein Inhaltsverzeichnis gefunden. Ein leerer Text ist kein Messwert, sondern eine fehlende Beobachtung;
bliebe er drin, würde die Bedingung für ein Dokument bewertet, zu dem sie gar keine
Eingabe hatte. Da die Evaluation Index und CSV zeilenweise gegeneinander hält,
schreibt die Stage über `--index_output` einen passend gefilterten Index gleich mit.

**Schutz für die Weaviate-Collection.** Jede Bedingung baut einen eigenen
Retrieval-Pool. Läuft eine zweite Bedingung versehentlich unter demselben
Collection-Namen, wäre der Pool der ersten still überschrieben und das Ergebnis
wertlos. `create_train_collection.py` schreibt deshalb die Quell-CSV in die
Collection-Beschreibung und bricht ab, wenn sie nicht zum aktuellen Lauf passt.

## Ausführen

```bash
cd pipelines/preprocess
dvc repro preprocess_toc@toc_train preprocess_toc@toc_validate preprocess_toc@toc_test
```

Die drei Stages entstehen durch `foreach`, deshalb der Zusatz `@toc_*` im Namen;
`dvc stage list` zeigt alle vorhandenen Namen.

Oder direkt, aus dem Repo-Root und ohne DVC:

```bash
python src/preprocess_selektion.py \
  --part_dir corpora/Sections/TOC_Extraction \
  --part_suffix _TOC.md \
  --index_file corpora/title/test.arrow \
  --ground_truth corpora/ground-truth.arrow \
  --cleaning toc \
  --empty_policy drop \
  --csv_output pipelines/preprocess/toc/test.csv \
  --index_output pipelines/preprocess/toc/test.arrow
```

## Arbeitsschritte: einen Fall durch die Pipeline schicken

Jeder Fall (TOC, TOC strict, Titel + Abstract, …) läuft nach demselben Ablauf.
Die beiden Parameterdateien in Schritt 1 und 2 können in beliebiger Reihenfolge
bearbeitet werden, müssen aber **beide** fertig sein, bevor `dvc repro` startet.

**0. Vorverarbeitung** — nur wenn der Fall neu ist oder sich Cleaner/Adapter geändert haben:

```bash
cd pipelines/preprocess
dvc repro -s <stage>@<fall>_train <stage>@<fall>_validate <stage>@<fall>_test
```

**1. `pipelines/train/params.yaml` — welche Daten:**

```yaml
train:
  text_data: ../preprocess/<fall>/train.csv
predict:
  text_data: ../preprocess/<fall>/validate.csv
  index: ../preprocess/<fall>/validate.arrow
```

**2. `pipelines/shared_params.yaml` — wie gerechnet wird:**

```yaml
retrieve:
  weaviate_collection_name: <fall>_train
completion:
  custom_instruction: instructions/instruction_<fall>.txt
```

- Jeder Fall bekommt einen **eigenen Collection-Namen**. Wird er vergessen, bricht
  `create_train_collection.py` ab, weil die Collection aus einer anderen CSV stammt.
- `max_model_len` und `max_total_tokens` bleiben über alle Fälle gleich, sonst sind
  die Ergebnisse nicht vergleichbar.

**3. Train-Pipeline laufen lassen:**

```bash
cd pipelines/train
dvc repro -s create_train_collection retrieve complete@mistral-7B-0p3 map@mistral-7B-0p3 compute_metrics_modelwise@mistral-7B-0p3
```

`-s` führt nur die genannten Stages aus, DVC ordnet sie selbst in die richtige
Reihenfolge und überspringt unveränderte. `create_train_collection` ist nicht
eingefroren und hat `--overwrite` im Befehl: Die Collection wird genau dann neu
gebaut, wenn sich die Train-CSV (oder ein Parameter der Stage) geändert hat. Zu
beachten: `dvc.lock` kennt nur den letzten Stand. Beim Zurückwechseln auf einen
früheren Fall wird dessen Collection deshalb neu eingebettet (~12 min); wer das
vermeiden will, lässt `create_train_collection` im Befehl weg.

**4. Ergebnisse sichern:**

```bash
R=results/<fall>_validate
mkdir -p $R/mistral-7B-0p3
cp predictions_model_wise/mistral-7B-0p3/{mapped_predictions.csv,mapping_stats.json,scores_at_100_R.yaml} $R/mistral-7B-0p3/
cp params.yaml ../shared_params.yaml $R/
```

`predictions_model_wise/` wird beim nächsten Fall überschrieben, deshalb landet jeder
Lauf samt Parameter-Snapshot unter `results/`.

**5. Committen:** beide Parameterdateien, beide `dvc.lock` und `results/<fall>_validate`.

## Stand der Daten (geprüft am 2026-10-11)

- Quelle: `aen/txt-praktikum`, Ordner `datasets/Sections`, gepinnt auf
  `rev_lock: 778785bb…` — die Extraktion ist damit eine feste Konstante
- TOC: 10.814 Dateien, Pfadschema `Sections/TOC_Extraction/{letztes Zeichen der idn}/{idn}_TOC.md`
- Ohne Metadaten-Header, der Inhalt beginnt in Zeile 1
- Abdeckung: train 9033/9043 · validate 837/837 · test 940/943 — die 13 fehlenden
  Dokumente haben kein TOC und werden über `empty_policy: drop` verworfen
- Länge in Zeichen: Median ~3.900 · p90 ~7.800 · max ~62.000

## Nächste Schritte

1. TOC-Durchlauf durch train- und predict-Pipeline, Bewertung mit casimir
   (R und AUCpr als Hauptmetriken), zusätzlich Token-Zahl und Laufzeit protokollieren
2. Vergleich gegen die Titel-Basis
3. Weitere Selektionen nach demselben Muster: Abstract, Abstract + Inhaltsverzeichnis,
   zusätzlich Einleitung und Fazit, Volltext als Referenz
4. Die Bedingungen automatisiert und reproduzierbar gegeneinander laufen lassen

Der Mechanismus, der die Bedingungen als DVC-Experimente startet, ist noch nicht in
diesem Branch — er wird separat nachgereicht, sobald er stabil ist.

## Fragen

Fragen oder Anmerkungen gerne direkt an mich (Cezar) oder als Kommentar im Branch.
