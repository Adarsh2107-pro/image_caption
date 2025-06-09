This document is a tutorial of how to run different experiments using our code and Hydra.

In `data/conf` we have our experiments store in .yaml files. `data/conf/experiment` contains specific experiments, and `conf/config.yaml` contains the default settings.

- `conf/config.yaml` contains the default settings
- `conf/experiment/demo.yaml` is the default experiment, which only has the purpose of specifying the saved model name
- `conf/experiment/exp1.yaml` is an example experiment. Any settings in this file is used to override the settings of the defaults.

Tutorial:

1. Navigate to the repository root directory (`image_caption`)
2. From the terminal, run `python -m image_caption.main`. This runs the default experiment, which loads `config.yaml` and `demo.yaml`.
3. To override the settings and run `exp1`, run `python -m image_caption.main experiment=exp1`
4. Similarly, you can run your own experiments from the terminal, e.g. `python -m image_caption.main model_id='tutorial_exp' epochs=1`
5. Alternatively, you can create your own config file in `data/conf/experiment/` and run it as seen in step 3.
