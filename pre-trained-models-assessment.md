# Pre-trained plant ID models found online

## List

### Pl@ntNet API

- returns likely plant species with confidence scores from 0 to 1
- accepts flower/leaf/fruit/bark organ hints
- can use geographic/flora projects instead of a totally open-world guess.
- needs an api key - costs money?

### iNaturalist pretrained model

- full species models not public, only some species models

### BioCLIP-style models

- can give it candidate species names and ask “which of these plants does this image look most like?”?

### Generic flower classifiers on Hugging Face

- useful for proof-of-concept, but often trained on narrow datasets like Oxford-102.

This github has a nice worked example of flower classification: <https://github.com/Navneet2409/Flower-Classification>
