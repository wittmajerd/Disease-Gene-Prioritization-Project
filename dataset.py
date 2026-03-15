from __future__ import annotations

from pathlib import Path
import pandas as pd


# Ez sokkal bonyolultabb lesz mert szerintem paraméterek alapján futás közben kéne létrehozni a hármasokat
# ez is konfigban állítható hogy a primekg-ből melyik linkeket szeretnénk használni 
# illetve a spliteket is úgy kialakítani hogy ne legyen túl sok test mert soha nem fut le :D

# elsősorban a helyi teljes primekg-t használnám tehát azt kéne beolvasni és feldolgozni
# első körben akár a letöltött pykeen csv is ok
# a lényeg hogy állítható legyen a kapcsolatok típusa és a train val test számok
# de közben nem akarom túlbonyolítani 

path = Path("primekg/kg.csv")

primekg_df = pd.read_csv(path)

# milyen pykeen datasetek vannak és itt mi lenne a legjobb? hol legyen a logika az osztályban vagy külön?