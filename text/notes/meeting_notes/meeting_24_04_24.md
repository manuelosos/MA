# Notes on Meeting

Stoch. Euler t mit wurzel skalieren 
Test um SDE bib einfach eine Brownian motion lösen

Fürs erste nur Numerische Simulationen machen 
einzelne Trajektorien generieren Mittelwert,.. etc
Vergleichen mit echten Simulationen
Löser  Euler und etc.. von grund auf selber Programmieren
Peter Kloeden Numerical Solutions of SDEs

Euler Maruyama Verfahren in der Arbeit Solver Beschreiben 
Allgemein aufschreiben wie in Literatur beschreiben und dann in Pseudocode übersetzen
Konkret mit Euler Maruyama Verfahren beschäftigen

Negative Werte sind standardproblem
Macht nur Sinn wenn Population groß ist und negative Werte unwahrscheinlich sind
Man macht dabei statistischen Fehler  Aber bei großen Modellen ist dieser zu vernachlässigen.
Parameterstudies machen ob man mehr Fehler macht wenn man in die extremen geht.

Gillespie Algorithm behandeln in Arbeit

Warum macht man statistischen Fehler in extremem?

Vollständiges Netzwerk anschauen
SDE Sim mit Agentenbasierter Sim vergleichen
Zählprozes

# Agentenbasierte Sim
In einem Zustand für jed Agent eine Wechselrate abhängig von x Zustand und Agenten
Alle Agenten ziehen eine Zahl und es wird gechect welcher Agent die kleinste hat dieser wird geupdatet

Gillespie ist für Zählprozess geeignet Well mixed proces 


# Fragen:

Definitionen der Raten
Warum sind die beiden Arten von Raten äquivalent?