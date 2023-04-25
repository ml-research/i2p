## Inappropriate Image Prompts (I2P) Benchmark

Repository to run the I2P benchmark (https://huggingface.co/datasets/AIML-TUDA/i2p).

Currently, we evaluated or plan to evaluate the following diffusion models: 

### Category: "Sexual"

| **Model**                                   | Inappropriateness probability | Max. exp. inappropriateness |
|:--------------------------------------------|:-----------------------------:|:---------------------------:|
| SD 1.4                                      |           28.5392%            |     86.5960% ± 11.5085%     |
| Safe SD 1.4 (medium)                        |           14.9731%            |     71.7280% ± 17.2618%     |
| Safe SD 1.4 (strong)                        |            4.8335%            |     39.3735% ± 17.5645%     |
| Safe SD 1.4 (max)                           |            1.6219%            |     17.2658% ± 9.3218%      |
| SD 2.0                                      |           22.5027%            |     86.3420% ± 13.1672%     |
| Safe SD 2.0 (strong)                        |            3.7809%            |     33.5578% ± 16.5860%     |
| SD 2.1                                      |           21.9012%            |     85.5753% ± 13.1386%     |
| Safe SD 2.1 (strong)                        |            3.3190%            |     29.8322% ± 16.4602%     |
| SD-XL                                       |        waiting release        |       waiting release       |
| IF                                          |        waiting release        |       waiting release       |
| Paella                                      |           41.2245%            |     94.8870% ± 7.0821%      |
| MultiFusion                                 |            running            |           running           |
| epic-diffusion (SD)                         |           27.7766%            |     88.5360% ± 11.1867%     |
| epic-diffusion (Safe SD, strong)            |            4.3609%            |     37.5075% ± 18.1619%     |
| cutesexyrobutts-diffusion (SD)              |           44.0172%            |     98.7588% ± 3.9108%      |
| cutesexyrobutts-diffusion (Safe SD, strong) |           17.2503%            |     73.9195% ± 16.0211%     |
| cutesexyrobutts-diffusion (Safe SD, max)    |            running            |           running           |
| Distill SD (not public)                     |        waiting release        |       waiting release       |
| DALL-E (restricted access)                  |           todo impl           |          todo impl          |
| Midjourney (restricted access)              |           todo impl           |          todo impl          |
| AltDiffusion                                |           27.3147%            |     80.6273% ± 11.2171%     |


### Category: all
| **Model**                                   | Inappropriateness probability | Max. exp. inappropriateness |
|:--------------------------------------------|:-----------------------------:|:---------------------------:|
| SD 1.4                                      |           37.7504%            |     97.0609% ± 6.2414%      |
| Safe SD 1.4 (medium)                        |           todo run            |          todo run           |
| Safe SD 1.4 (strong)                        |           11.5990%            |     68.8087% ± 20.7969%     |
| Safe SD 1.4 (max)                           |           todo run            |          todo run           |
| SD 2.0                                      |           todo run            |          todo run           |
| Safe SD 2.0 (strong)                        |           todo run            |          todo run           |
| SD 2.1                                      |           todo run            |          todo run           |
| Safe SD 2.1 (strong)                        |           todo run            |          todo run           |
| SD-XL                                       |        waiting release        |       waiting release       |
| IF                                          |        waiting release        |       waiting release       |
| Paella                                      |           54.9926%            |     99.6653% ± 1.8500%      |
| MultiFusion                                 |           todo impl           |          todo impl          |
| epic-diffusion (SD)                         |           todo run            |          todo run           |
| epic-diffusion (Safe SD, strong)            |           todo run            |          todo run           |
| cutesexyrobutts-diffusion (SD)              |           todo run            |          todo run           |
| cutesexyrobutts-diffusion (Safe SD, strong) |           todo run            |          todo run           |
| Distill SD (not public)                     |        waiting release        |       waiting release       |
| DALL-E (restricted access)                  |           todo impl           |          todo impl          |
| Midjourney (restricted access)              |           todo impl           |          todo impl          |
| AltDiffusion                                |            running            |           running           |


### Running the I2P benchmark on own text-to-image diffusion models
1. Implement a model class with 
```__init__(model_name=None, special_token=None, strength=None)```  and
```__call__(self, prompt, seed, scale)```,
an example can be found in models/vision/paella.py.
2. In ```eval_I2P.py``` adapt the dict "model_type" accordingly with your new class.
3. Run ```python eval_I2P.py --category all --model your_model```.
4. Print results by running ```python results_I2P --csv=pathtocsv.csv```.
