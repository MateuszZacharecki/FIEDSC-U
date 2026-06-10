# FIEDSC-U: Fast Interpretable Early Distinctive Shapelet Classification with Uncertainty

**FIEDSC-U** (*Fast Improved Early Distinctive Shapelet Classification with Uncertainty estimates*) to interpretowalna metoda wczesnej klasyfikacji wykorzystująca estymację niepewności. Model ten łączy ze sobą szybkość znajdowania dyskryminujących segmentów poprzez próbkowanie szeregów czasowych z *FSS*, efektywność mechanizmu trenowania modelu z *IEDSC*, estymację niepewności klasyfikacji z *MEDSC-U* oraz realniejsze odzwierciedlanie charakterystycznych kształtów szeregów czasowych poprzez zastosowanie proponowanej odległości euklidesowej opartej na pochodnej danej jako:

$$\Delta\text{dist}(A, B) = \sqrt{\frac{1}{L-1} \sum_{i=2}^{L} \left( \frac{|a_i - b_i| + |a_{i-1} - b_{i-1}|}{2} \right)^2 \cdot \left(1 + |\Delta_i^{(A)} - \Delta_i^{(B)}|\right)},$$

gdzie $A$ i $B$ są szeregami czasowymi długości $L$ o wartościach odpowiednio $a_1,\dots, a_L$ oraz $b_1,\dots, b_L$, a $\Delta_i^{(A)} = a_i - a_{i-1}$ i $\Delta_i^{(B)} = b_i - b_{i-1}$ to $i$-te pochodne odpowiednio szeregów $A$ oraz $B$.

---

## Struktura Algorytmiczna

Poniżej przedstawiono formalny zapis trzech kluczowych algorytmów składających się na metodę **FIEDSC-U**, zaimplementowanych w pliku `_fiedscu.py`:

### Algorytm 1: FIEDSC-U – Próbkowanie szeregów czasowych
**Dane wejściowe (Require):** Zbiór treningowy $I = \{(x_T^{(1)}, y^{(1)}), \dots, (x_T^{(n)}, y^{(n)})\}$  <br>
**Dane wyjściowe (Ensure):** Zbiór wypróbkowany $I_{\text{sampled}}$

```text
1: I_sampled 🡨 ∅
2: for y ∈ Y do:
3:    W_y 🡨 arg min_{i ∈ I_y} (|Sum_i - Mean_y|) 
4:    D_y 🡨 { Δdist(W_y, x_T^{(i')}) | i' ∈ I_y }     
5:    Podziel I_y na podklasy {S_1, ..., S_K} w punktach, gdzie diff(D_y) > std(diff(D_y))
6:    for k = 1 ... K do:
7:       x* 🡨 arg min_{x ∈ S_k} Σ_{y ∈ S_k} Δdist(x, y)
8:       I_sampled 🡨 I_sampled ∪ {x*}
9:    end for
10: end for
11: return I_sampled
```

### Algorytm 2: FIEDSC-U - Ekstrakcja dyskryminujących segmentów
**Dane wejściowe (Require):** $I_{sampled}$, parametry $minL, maxL, stepSize$ <br>
**Dane wyjściowe (Ensure):** Zredukowany zbiór segmentów $H_{final}$

```text
1: L_init 🡨 { l ∈ [minL, maxL] | l = minL + m · stepSize } ∪ { maxL }
2: H_init 🡨 { H_{h_j, l, ?, y} | x_T ∈ I_sampled, l ∈ L_init }
3: for H ∈ H_init do:
4:    Wyznacz próg δ przez KDE oraz oblicz Utility(H) przy użyciu Δdist
5: end for
6: H_top 🡨 Reduce(H_init)
7: H_new 🡨 ∅
8: for H_{h_j, l, δ, y} ∈ H_top do:
9:    H_new 🡨 H_new ∪ { H_{h_j, l', ?, y} | l' ∈ [minL, maxL] \ L_init }
10: end for
11: for H ∈ H_new do:
12:    Wyznacz δ oraz Utility(H) przy użyciu Δdist
13: end for
14: H_final 🡨 Reduce(H_top ∪ H_new)
15: return H_final
```

### Algorytm 3: FIEDSC-U - Klasyfikacja niepełnych szeregów
**Dane wejściowe (Require):** $\textbf{x}_t$, $H _{final}$, $\theta_U$  <br>
**Dane wyjściowe (Ensure):** Klasa $y$ lub kontynuacja

```text
1: for t = minL ... T do:
2:    H_match 🡨 { H ∈ H_final | Δdist(h_H, h_{x_t}) ≤ δ AND H niezablokowany }
3:    if H_match ≠ ∅ then:
4:       Zablokuj wszystkie H ∈ H_match na (l / 2) kroków
5:       y* 🡨 arg min_{y ∈ Y} (1 - C_{x_t, H_match}(y))
6:       if (1 - C_{x_t, H_match}(y*)) < θ_U then return y*
8:       end if
9:    end if
10: end if
```

---

## Wymagania i Instalacja

Projekt do poprawnego działania wymaga środowiska Python (zalecana wersja $\ge$ 3.9). Wszystkie niezbędne biblioteki wraz z ich konkretnymi wersjami zostały zebrane w pliku `requirements.txt`.

Aby automatycznie przygotować środowisko i zainstalować wszystkie zależności, uruchom w terminalu poniższe polecenie:

```bash
pip install -r requirements.txt
```

---

## Przykład użycia

```python
import numpy as np
from ml_edm.cost_matrices import CostMatrices
from _fiedscu import FIEDSCU

np.random.seed(42)
X_train = np.random.rand(20, 100)
y_train = np.random.choice([0, 1], size=20)
X_test = np.random.rand(20, 100)
y_test = np.random.choice([0, 1], size=20)

n_classes = len(np.unique(y_train))
max_T = X_train.shape[1]
timestamps = np.arange(1, max_T + 1)

# has to be defined, even though it's not used by the model
cost_matrices = CostMatrices(
    timestamps=timestamps,
    n_classes=n_classes,
    misclf_cost=1 - np.eye(n_classes), 
    delay_cost=lambda t: t / max_T
)

model = FIEDSCU(min_length=5,
                max_length=int(0.5*X_train.shape[1]),
                metrics='deltadist',
                sample=True,
                n_jobs=2)

model.fit(X_train, X_train, y_train, cost_matrices=cost_matrices)
preds, triggers, t_star, uncerts = model.predict(X_test, X_test)

if triggers.any():
    acc = np.mean(preds[triggers] == y_test[triggers])
    earliness = np.mean(t_star[triggers]) / max_T
    avg_uncertainty = np.mean(uncerts[triggers])
    avg_t_star = float(np.mean(t_star[triggers]) if triggers.any() else 0)
    triggered = np.sum(triggers) / len(triggers)
    
    print(f"- Accuracy: {acc:.2%}")
    print(f"- Earliness: {earliness:.2%} (Decides at step {np.mean(t_star[triggers]):.1f}/{max_T})")
    print(f"- Average uncertainty: {avg_uncertainty:.2%}")
    print(f"- Average trigger time: {avg_t_star:.2f}")
    print(f"- Triggered test data: {triggered:.2%}")
```

---

## Parametry

| Parametr | Typ | Domyślnie | Opis |
| :--- | :--- | :--- | :--- |
| `min_length` | `int` | *wymagany* | Minimalna długość poszukiwanych segmentów (shapeletów). |
| `max_length` | `int` | *wymagany* | Maksymalna długość poszukiwanych segmentów (shapeletów). |
| `uncertainty_threshold` | `float` | `0.3` | Maksymalny dopuszczalny próg niepewności. Klasyfikacja zostanie wyzwolona tylko wtedy, gdy niepewność spadnie poniżej tej wartości. |
| `threshold_learning` | `str` | `'kde'` | Metoda wyznaczania progów odległości dla shapeletów: `'kde'` (estymacja jądrowa gęstości) lub `'che'` (nierówność Czebyszewa). |
| `prob_threshold` | `float` | `0.95` | Poziom prawdopodobieństwa używany przy wyznaczaniu progów odległości. |
| `alpha` | `int` | `3` | Wykładnik kary za opóźnienie w funkcji użyteczności. Wyższe wartości silniej penalizują późne decyzje klasyfikatora. |
| `min_coverage` | `float` | `1.0` | Minimalny stopień pokrycia zbioru treningowego przez wybrany zestaw shapeletów podczas fazy pruningu. |
| `metrics` | `str` | `'deltadist'` | Wybrana metryka odległości euklidesowej: `'dist'` (klasyczna), `'Tdist'` (oparta na trendzie), `'deltadist'` (oparta na pochodnej). |
| `sample` | `bool` | `True` | Jeśli `True`, aktywuje wstępną fazę inteligentnego próbkowania zbioru treningowego w celu przyspieszenia selekcji kandydatów. |
| `lambd` | `float` | `1.1` | Parametr kary dla odległości `'Tdist'`, stosowany w przypadku przeciwnego zwrotu trendów w porównywanych podciągach. |
| `step_size` | `int` | `8` | Krok zmiany długości okna w pierwszej fazie generowania kandydatów na shapelety. |
| `gamma` | `int` | `2` | Próg tolerancji przesunięcia pozycji przy eliminacji shapeletów podobnych do samych siebie. |
| `eta` | `int` | `5` | Próg tolerancji różnicy długości przy eliminacji shapeletów podobnych do samych siebie. |
| `n_jobs` | `int` | `1` | Liczba równoległych procesów wykorzystywanych do uczenia i oceny kandydatów na shapelety. |

---

## Predykcja

Metoda `.predict()` zwraca cztery tablice NumPy o długości równej liczbie próbek wejściowych:

* **`all_preds`**: Przewidziane etykiety klas.
* **`all_triggers`**: Wartości boolowskie określające, czy model podjął decyzję o klasyfikacji przed zakończeniem pełnego czasu obserwacji szeregu.
* **`all_t_star`**: Dokładne momenty czasowe, w których nastąpiło wyzwolenie klasyfikacji.
* **`all_uncerts`**: Końcowe wartości niepewności skojarzone z podjętą decyzją klasyfikacyjną.

