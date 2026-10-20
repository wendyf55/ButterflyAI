# Old plant models on real photos

Sanity (composites, probably seen in training): OVR 0.840, multiclass 0.965

## Gold (2162 real photos)

- top-1 accuracy: OVR 0.758, multiclass 0.849 (always-most-common-species baseline 0.267)
- OVR: no model says yes on 13.9% of photos; 2 or more say yes on 19.3%

Per species. 'top-1 recall' = share of that species' photos given the right species.
'binary' columns = that one OVR model on its own, yes/no at the threshold.

| species | n | OVR top-1 recall | MC top-1 recall | OVR binary recall | OVR binary precision | OVR AUC |
|---|---|---|---|---|---|---|
| Achillea_millefolium | 167 | 0.725 | 0.826 | 0.790 | 0.534 | 0.947 |
| Anaphalis_margaritacea | 310 | 0.832 | 0.874 | 0.806 | 0.694 | 0.953 |
| Apocynum_androsaemifolium | 122 | 0.934 | 0.893 | 0.918 | 0.636 | 0.989 |
| Asclepias_speciosa | 84 | 0.821 | 0.881 | 0.810 | 0.618 | 0.971 |
| Cirsium_arvense | 578 | 0.804 | 0.917 | 0.754 | 0.952 | 0.976 |
| Ericameria_nauseosa | 65 | 0.985 | 0.938 | 0.985 | 0.448 | 0.996 |
| Leucanthemum_vulgare | 240 | 0.600 | 0.692 | 0.675 | 0.717 | 0.929 |
| Medicago_sativa | 214 | 0.514 | 0.790 | 0.444 | 0.772 | 0.935 |
| Sisymbrium_loeselii | 115 | 0.730 | 0.930 | 0.687 | 0.908 | 0.984 |
| Tripleurospermum_inodorum | 267 | 0.783 | 0.790 | 0.745 | 0.545 | 0.924 |

Macro recall (average of the per-species recalls; chance = 0.100): OVR 0.773, multiclass 0.853

## Mixed plants (24 target species, 114 other plants)

- target photos given the right species: OVR 0.583, multiclass 0.750
- other-plant photos where no OVR model says yes (correct): 0.246
- average number of OVR models saying yes on an other-plant photo: 0.95
- multiclass top score, median: target 0.980, other 0.681
  (if 'other' is much lower than 'target', a cut-off on this score could reject other plants)
- which species the OVR models pick for other-plant photos:
  - Apocynum_androsaemifolium: 38
  - Tripleurospermum_inodorum: 15
  - Leucanthemum_vulgare: 14
  - Sisymbrium_loeselii: 12
  - Achillea_millefolium: 10
  - Medicago_sativa: 8
  - Asclepias_speciosa: 7
  - Cirsium_arvense: 5
  - Ericameria_nauseosa: 3
  - Anaphalis_margaritacea: 2
