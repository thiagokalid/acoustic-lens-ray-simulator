# Machine learning–driven flaw detection for ultrasonic pipe inspections with acoustic lens (ENCDT 2026)

by 
[Thiago E. Kalid](https://orcid.org/0000-0002-2035-5349),
[André E. Lazzaretti](https://orcid.org/0000-0003-1861-3369),
[Tatiana de A. Prado](https://orcid.org/0000-0002-4876-2974),
[Gustavo P. Pires](https://orcid.org/0009-0008-3474-6077),
[Daniel R. Pipa](https://orcid.org/0000-0002-9398-332X),
[Thiago A. R. Passarin](https://orcid.org/0000-0003-1001-5911),.

 <br>
This repository contains the data and source code used to produce the results presented in:

> PLACEHOLDER
 
<br>
 
|                              | Info |
|------------------------------|------|
| Version of record            |   [``]()   |
| Open-access preprint |   [``]()   | 
| Archive of this repository   |   [`https:/doi.org/10.5281/zenodo.19410832`](https:/doi.org/10.5281/zenodo.19410832)   | 
| Reproducing our results | [`REPRODUCING.md`](REPRODUCING.md) |

## Abstract

Acoustic lenses have been recently employed during ultrasonic pipe inspection to increase the phased-array transducer coverage area. However, the lens also introduces geometrical artefacts, which affect the flaw detection process. Classic flaw detection methodologies commonly rely on thresholding, which becomes problematic when geometrical artifacts closely match the desired signal amplitude. We propose an adaptive method for flaw detection from ultrasonic inspections of tubes with acoustic lenses based on anomaly detection techniques. A  machine-learning model was trained on data from flawless pipe specimens using an acoustic lens. During training, the model captures patterns associated with the flawless pipe and lens. Then, during testing, it analyzes newly observed data and classifies it either as a flaw (anomaly) or a normal response (as in the training set). Our model, based on the Local Outlier Factor, achieved an area under the receiver operating characteristic curve of 0.96, a global accuracy of 0.95, a recall of 0.91, and an F2-score of 0.84. These results, achieved across a wide range of flaw geometries, suggest that our model could be part of an automated flaw detection system for pipeline inspections.

## License
All Python source code (`.py`) is made available
under the MIT license. You can freely use and modify the code, without
warranty, so long as you provide attribution to the authors. See
`LICENSE-MIT.txt` for the full license text.

Figures and data
produced as part of this research are available under the [Creative Commons
Attribution 4.0 License (CC-BY)][cc-by]. See `LICENSE-CC-BY.txt` for the full
license text.

[cc-by]: https://creativecommons.org/licenses/by/4.0/


