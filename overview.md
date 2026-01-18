# Gén–betegség priorizálás – összefoglaló

## Feladat és cél
- Gén–betegség kapcsolatbecslés/link-predikció: meglévő ismert élek alapján új kapcsolatok rangsorolása.
- Kimenet: valószínűség/pontszám minden lehetséges betegség–fehérje párra, majd top-K jelöltek listája.

## Adatforrás
- **OGB ogbl-biokg** heterogén tudásgráf.
- Csomópontok: betegség, fehérje (és egyéb biológiai entitások a teljes gráfban).
- Kapcsolatok: betegség–fehérje (cél), fehérje–fehérje több relációtípussal (binding, activation, inhibition, stb.).
- Javaslat: reverz élek felvétele (pl. fehérje→betegség), hogy mindkét irányban áramolhasson az információ.

## Gráfreprezentáció
- **Heterogén gráf** (PyG `HeteroData`): minden reláció külön élindex, relációnévben csak betű/szám/underscore (pl. `disease_protein`).
- **Bipartit lapítás** (opcionális): betegség–fehérje egy homogén gráfban; gyorsabb, de kevesebb információ (nem használja a PPI relációkat külön-külön).

## Modellek
- Heterogén GNN encoderek: GraphSAGE/GraphConv (GCN stílus), GAT, HGT. (RGCN csak akkor, ha integer edge_type áll rendelkezésre.)
- Dekóder: pontszorzat a két csomópont beágyazása között a kapcsolat pontszámára.

## Negatív mintavétel
- Pozitív élek: ismert betegség–fehérje párok.
- Negatív élek: külön mintavett betegség- és fehérje-indexek, hogy tartományon belül maradjanak.

## Tanítási beállítások
- Veszteség: bináris keresztentrópia logitokra (BCEWithLogitsLoss).
- Early stopping: F1-re optimalizálva, türelem (patience) alapján.
- Checkpoint: legjobb modell automatikus mentése (pl. `hetero_<modell>_best_model.pth`).
- Eszköz: CUDA, ha elérhető (`device=auto/cuda/cpu`).

## Metrikák
- **Elsődleges**: F1.
- **Továbbiak**: precision, recall, accuracy, AUC. (Kiegyensúlyozatlan esetben AUPR is hasznos; OGB leaderboardon MRR/Hits@K jellemző.)

## Hyperparaméterek
- `model_kind`: sage | gcn (GraphConv) | gat | hgt.
- `hidden`, `out`: rejtett és kimeneti dimenziók.
- `lr`, `weight_decay`, `epochs`, `eval_every`, `train_ratio`, `val_ratio`.
- `heads`: GAT/HGT fejek száma.
- `num_bases`: RGCN esetén bázisok száma (ha edge_type elérhető).
- `add_ppi`: fehérje–fehérje élek bevonása.

## Környezeti kompatibilitás (PyTorch/PyG/OGB)
- PyTorch és PyG binárisoknak illeszkedniük kell verzióban és CUDA buildben (pl. torch 2.3.0 + cu118 ↔ PyG torch-2.3.0+cu118 kerekek).
- OGB verzió: a legfrissebb `ogb` pip csomag hozza a `PygLinkPropPredDataset`-et.
- Teszt: `torch.cuda.is_available()` igaz-e; `from ogb.linkproppred import PygLinkPropPredDataset` import sikeres-e.

## Gyors futtatási folyamat
1) Környezet: CUDA-s PyTorch + illeszkedő PyG + OGB.
2) Adat betöltése: `load_biokg_hetero(include_ppi=True)`, relációnév-szanitizálás (underscore).
3) Él-split: `random_split_edges` train/val/test arányokkal.
4) Modellválasztás: `model_kind` (sage/gcn/gat/hgt).
5) Tanítás: early stopping F1-re; checkpoint mentés.
6) Értékelés: F1, precision, recall, accuracy, AUC; opcionálisan AUPR, MRR/Hits@K, ha kell a leaderboard-kompatibilitás.

## Hasznos megjegyzések
- Ha csak CPU-s PyTorch van, a kód fut, de lassú; válts CUDA-s buildre, ha a GPU támogatott.
- A relációnevekben kerüld a kötőjelet (PyG `to_hetero` figyelmeztet rá); használj `_` karaktert.
- A negatív mintavétel node-típusonkénti tartományra figyel, elkerülve indexelési hibákat.
