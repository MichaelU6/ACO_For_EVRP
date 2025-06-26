# Prehľad

Electric Vehicle Routing Problem (EVRP) je rozšírením klasického Vehicle Routing Problem (VRP) so zohľadnením obmedzení batérie elektrických vozidiel. Cieľom je minimalizovať celkové náklady (vzdialenosť alebo čas) pri zabezpečení dobitia batérie v nabíjacích staniciach.

Ant Colony Optimization (ACO) je metaheuristický algoritmus inšpirovaný správaním mravcov pri hľadaní najkratších ciest. Mravce ukladajú feromón na navštívených cestách a tým ovplyvňujú pravdepodobnosť výberu ciest budúcimi mravcami.

# Algoritmus ACO

Inicializácia: Nastav feromónové hodnoty na všetkých hranách grafu na rovnocenné malé hodnoty.

Konštrukcia riešení: Každý mravec postupne vyberá nasledujúci uzol (klient alebo nabíjaciu stanicu) na základe kombinácie feromónu a heuristickej informácie (napr. inverznej vzdialenosti).

Aktualizácia feromónu:

Odparovanie: Zníženie feromónu na každej hrane o určitý koeficient.

Nanáška: Každý mravec prispieva feromónom na hranách svojej cesty úmerne kvalite riešenia (napr. inverznej dĺžke trasy).

Dobíjanie: Pri konštrukcii cesty sa berie do úvahy stav batérie. Ak mravec nemôže dosiahnuť ďalšieho klienta, navštívi najbližšiu nabíjaciu stanicu.

Iterácia: Kroky 2–4 sa opakujú, až kým sa nesplní kritérium ukončenia (počet iterácií alebo stabilita riešenia).
