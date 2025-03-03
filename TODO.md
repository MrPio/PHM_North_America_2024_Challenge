# NOTES:

- Output Layer: Instead of one neuron, use two neurons (or more if using a more complex distribution) to output μ and
  log(σ) (*using the log variance or log standard deviation helps with numerical stability and ensures σ remains
  positive*)
- Spiega come la log likelihood 0.5*log(2πσ^2) + (y - μ)^2 / (2σ^2) deriva da -ln(f(ground_truth)) !!!!
- `Can I assume the PDF to be Gaussian? Why?`
  Unimodality and Symmetry (fatto a lezione nei test di HP. Il Tm può variare sia in negativo che in positivo)
  The torque margin itself might be a complex function of many variables. The CLT isn’t saying that the torque margin (
  as a raw value) must be normal. Rather, if you assume the true underlying process has many small, independent
  disturbances, then the deviation (error) between your model’s prediction and the actual observed value is likely to be
  normal
- `Why should I train using NLL instead of RMS?`
  NLL evaluates how probable the observed torque value is under the predicted distribution. Also trains variance.

# TODO:

- reverse della normalizzazione
- test split in holdout

# DONE:

- prova moltiplicazioni 