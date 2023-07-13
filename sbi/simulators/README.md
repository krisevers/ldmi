# Simulators

## Neural Models:
* DCM | Dynamic Causal Model from Havlicek et al. (2020)
* DMF | Dynamic Mean Field
## Observations Models:
* NVC | Neuro Vascular Coupling model
* LBR | Laminar BOLD Response model

# TODO: 
* perform SBI on models separate and together.
* link models: DCM > NVC > LBR.
    * Is it possible to keep part of the model fixed and only infer other parts of the parameters?
    * This should be possible because the models only have sequential input-output relations.
* Which parameters are dependent on each other?
* Investigate effects of interlaminar connectivity and external input LBR response.