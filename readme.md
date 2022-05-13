# GENERATOR KRAJOBRAZU

## Parametry
* `N (-N) - parametr rozmiaru macierzy`
* `sigma (-s) - stopień górzystości`
* `map save file (-mpsf) - nazwa pliku do zapisu mapy`
* `surface save file (-ssf) - nazwa pliku do zapisu powierzchni`
* `matrix save file (-mxsf) - nazwa pliku do zapisu macierzy`
* `map color (-c) - kolor wykresów`
* `matrix load file (-mxlf) - nazwa pliku do wyczytania macierzy`
* `start step (-bgs) - krok początkowy`
* `end step (-ends) - krok końcowy`

## Funkcjnalności

1. generowanie mapy i powierzchni krajobrazu i otwarcie ich w okienkach lub zapisanie do pliku wybranego wykresu.
 Możliwosć zapisu macierzy do pliku .txt
2. generowanie częściowo uzupełnionej macierzy i zapisanie jej do pliku
3. generowanie mapy i powierzchni dla częściowo uzupełnionej macierzy (otwarcie okien/zapis do pliku + zapis macierzy)

### Uwagi

* ze względu na wielkość macierzy przy N=14 i parametry komputera nie byłam w stanie sprawdzić czy program działa,
ale teorytycznie powinien
* dla N > 9 program działa, tylko trzeba poczekać chwilę aby usupełnił macierz