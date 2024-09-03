# Evaluate topological models

## VAE Results

```
PS C:\Users\ernst\Documents\02-Work\06-Repos\PointFlow> .\test_models_vae_topo.ps1
distChamferCUDA not available; fall back to slower version.
emd_approx_cuda not available. Fall back to slower version.
Total number of data:2832
Min number of points: (train)2048 (test)2048
Total number of data:405
Min number of points: (train)2048 (test)2048
Cate=airplane Total Sample size:torch.Size([405, 2048, 3]) Ref size: torch.Size([405, 2048, 3])
{'MMD-CD': 0.002180740935727954, 'MMD-EMD': 0.1255245804786682}
================================================================================
All category results:
================================================================================
{'airplane': {'MMD-CD': 0.002180740935727954, 'MMD-EMD': 0.1255245804786682}}
================================================================================
Averaged results:
{'MMD-CD': 0.002180740935727954, 'MMD-EMD': 0.1255245804786682}
================================================================================
distChamferCUDA not available; fall back to slower version.
emd_approx_cuda not available. Fall back to slower version.
Total number of data:2458
Min number of points: (train)2048 (test)2048
Total number of data:352
Min number of points: (train)2048 (test)2048
Cate=car Total Sample size:torch.Size([352, 2048, 3]) Ref size: torch.Size([352, 2048, 3])
{'MMD-CD': 0.004286990035325289, 'MMD-EMD': 0.10104593634605408}
================================================================================
All category results:
================================================================================
{'car': {'MMD-CD': 0.004286990035325289, 'MMD-EMD': 0.10104593634605408}}
================================================================================
Averaged results:
{'MMD-CD': 0.004286990035325289, 'MMD-EMD': 0.10104593634605408}
================================================================================
distChamferCUDA not available; fall back to slower version.
emd_approx_cuda not available. Fall back to slower version.
Total number of data:4612
Min number of points: (train)2048 (test)2048
Total number of data:662
Min number of points: (train)2048 (test)2048
Cate=chair Total Sample size:torch.Size([662, 2048, 3]) Ref size: torch.Size([662, 2048, 3])
{'MMD-CD': 0.008860084228217602, 'MMD-EMD': 0.17964476346969604}
================================================================================
All category results:
================================================================================
{'chair': {'MMD-CD': 0.008860084228217602, 'MMD-EMD': 0.17964476346969604}}
================================================================================
Averaged results:
{'MMD-CD': 0.008860084228217602, 'MMD-EMD': 0.17964476346969604}
================================================================================
```

## Output of running the test scripts for the Topological models

```
distChamferCUDA not available; fall back to slower version.
emd_approx_cuda not available. Fall back to slower version.
Total number of data:2832
Min number of points: (train)2048 (test)2048
Total number of data:405
Min number of points: (train)2048 (test)2048
Cate=airplane Total Sample size:torch.Size([405, 2048, 3]) Ref size: torch.Size([405, 2048, 3])
{'MMD-CD': 0.0007405525539070368, 'MMD-EMD': 0.10882669687271118}
================================================================================
All category results:
================================================================================
{'airplane': {'MMD-CD': 0.0007405525539070368, 'MMD-EMD': 0.10882669687271118}}
================================================================================
Averaged results:
{'MMD-CD': 0.0007405525539070368, 'MMD-EMD': 0.10882669687271118}
================================================================================
distChamferCUDA not available; fall back to slower version.
emd_approx_cuda not available. Fall back to slower version.
Total number of data:2458
Min number of points: (train)2048 (test)2048
Total number of data:352
Min number of points: (train)2048 (test)2048
Cate=car Total Sample size:torch.Size([352, 2048, 3]) Ref size: torch.Size([352, 2048, 3])
{'MMD-CD': 0.0028651778120547533, 'MMD-EMD': 0.09479302167892456}
================================================================================
All category results:
================================================================================
{'car': {'MMD-CD': 0.0028651778120547533, 'MMD-EMD': 0.09479302167892456}}
================================================================================
Averaged results:
{'MMD-CD': 0.0028651778120547533, 'MMD-EMD': 0.09479302167892456}
================================================================================
distChamferCUDA not available; fall back to slower version.
emd_approx_cuda not available. Fall back to slower version.
Total number of data:4612
Min number of points: (train)2048 (test)2048
Total number of data:662
Min number of points: (train)2048 (test)2048
Cate=chair Total Sample size:torch.Size([662, 2048, 3]) Ref size: torch.Size([662, 2048, 3])
{'MMD-CD': 0.004214603919535875, 'MMD-EMD': 0.1639052778482437}
================================================================================
All category results:
================================================================================
{'chair': {'MMD-CD': 0.004214603919535875, 'MMD-EMD': 0.1639052778482437}}
================================================================================
Averaged results:
{'MMD-CD': 0.004214603919535875, 'MMD-EMD': 0.1639052778482437}
================================================================================

```


## Previous test runs 

Result (trained only 10 epochs): 
{'MMD-CD': 0.0017162010772153735, 'MMD-EMD': 0.21514080464839935}

Result topological encoder (trained only 100 epochs): 
{'MMD-CD': 0.0008392978925257921, 'MMD-EMD': 0.10923629254102707}
NOTE: **The results from pointflow got trained for 4k epochs.** 



# Evaluate PointFlow


## Output of running the test scripts for the PointFlow models

```
distChamferCUDA not available; fall back to slower version.
emd_approx_cuda not available. Fall back to slower version.
Number of trainable parameters of Point CNF: 927513
Resume Path:pretrained_models/ae/airplane/checkpoint.pt
Total number of data:2832
Min number of points: (train)2048 (test)2048
Total number of data:405
Min number of points: (train)2048 (test)2048
Cate=airplane Total Sample size:torch.Size([405, 2048, 3]) Ref size: torch.Size([405, 2048, 3])
{'MMD-CD': 0.0008054337813518941, 'MMD-EMD': 0.04849435016512871}
================================================================================
All category results:
================================================================================
================================================================================
Averaged results:
{'MMD-CD': 0.0008054337813518941, 'MMD-EMD': 0.04849435016512871}
================================================================================
distChamferCUDA not available; fall back to slower version.
emd_approx_cuda not available. Fall back to slower version.
Number of trainable parameters of Point CNF: 927513
Resume Path:pretrained_models/ae/car/checkpoint.pt
Total number of data:2458
Min number of points: (train)2048 (test)2048
Total number of data:352
Min number of points: (train)2048 (test)2048
Cate=car Total Sample size:torch.Size([352, 2048, 3]) Ref size: torch.Size([352, 2048, 3])
{'MMD-CD': 0.0036658979952335358, 'MMD-EMD': 0.08432621508836746}
================================================================================
All category results:
================================================================================
{'car': {'MMD-CD': 0.0036658979952335358, 'MMD-EMD': 0.08432621508836746}}
================================================================================
Averaged results:
{'MMD-CD': 0.0036658979952335358, 'MMD-EMD': 0.08432621508836746}
================================================================================
distChamferCUDA not available; fall back to slower version.
emd_approx_cuda not available. Fall back to slower version.
Number of trainable parameters of Point CNF: 927513
Resume Path:pretrained_models/ae/chair/checkpoint.pt
Total number of data:4612
Min number of points: (train)2048 (test)2048
Total number of data:662
Min number of points: (train)2048 (test)2048
Cate=chair Total Sample size:torch.Size([662, 2048, 3]) Ref size: torch.Size([662, 2048, 3])
{'MMD-CD': 0.004134485498070717, 'MMD-EMD': 0.09602253139019012}
================================================================================
All category results:
================================================================================
{'chair': {'MMD-CD': 0.004134485498070717, 'MMD-EMD': 0.09602253139019012}}
================================================================================
Averaged results:
{'MMD-CD': 0.004134485498070717, 'MMD-EMD': 0.09602253139019012}
================================================================================
```



## Some previous test runs on airplanes only
Evaluate the Pointflow model on airplanes.

```{python} 
python test.py --cates airplane --resume_checkpoint pretrained_models/ae/airplane/checkpoint.pt --dims 512-512-512 --use_deterministic_encoder --evaluate_recon

```
Yields: 
```
{'MMD-CD': 0.0008063472923822701, 'MMD-EMD': 0.04878174886107445}
```



# Output of training the ect encoder models. 


```
C:\Users\ernst\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\lightning\pytorch\trainer\configuration_validator.py:72: You passed in a `val_dataloader` but have no `validation_step`. Skipping val loop.
Processing...
Done!
Processing...
Done!
Processing...
Done!
Processing...
Done!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name                | Type             | Params
---------------------------------------------------------
0 | layer               | EctLayer         | 0
1 | training_accuracy   | MeanSquaredError | 0
2 | validation_accuracy | MeanSquaredError | 0
3 | test_accuracy       | MeanSquaredError | 0
4 | loss_fn             | MSELoss          | 0
5 | model               | Sequential       | 39.9 M
---------------------------------------------------------
39.9 M    Trainable params
0         Non-trainable params
39.9 M    Total params
159.433   Total estimated model params size (MB)
C:\Users\ernst\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\lightning\pytorch\trainer\connectors\data_connector.py:441: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=23` in the `DataLoader` to improve performance.
Epoch 199: 100%|███████████████████████████████| 45/45 [01:32<00:00,  0.49it/s, v_num=lysd, train_loss=0.000893, train_accuracy=0.00158]`Trainer.fit` stopped: `max_epochs=200` reached.
Epoch 199: 100%|███████████████████████████████| 45/45 [01:33<00:00,  0.48it/s, v_num=lysd, train_loss=0.000893, train_accuracy=0.00158]
Processing...
Done!
Processing...
Done!
Processing...
Done!
Processing...
Done!
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
C:\Users\ernst\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\lightning\pytorch\trainer\configuration_validator.py:72: You passed in a `val_dataloader` but have no `validation_step`. Skipping val loop.
Processing...
Done!
Processing...
Done!
Processing...
Done!
Processing...
Done!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name                | Type             | Params
---------------------------------------------------------
0 | layer               | EctLayer         | 0
1 | training_accuracy   | MeanSquaredError | 0
2 | validation_accuracy | MeanSquaredError | 0
3 | test_accuracy       | MeanSquaredError | 0
4 | loss_fn             | MSELoss          | 0
5 | model               | Sequential       | 39.9 M
---------------------------------------------------------
39.9 M    Trainable params
0         Non-trainable params
39.9 M    Total params
159.433   Total estimated model params size (MB)
C:\Users\ernst\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\lightning\pytorch\trainer\connectors\data_connector.py:441: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=23` in the `DataLoader` to improve performance.
Epoch 199: 100%|███████████████████████████████| 39/39 [01:20<00:00,  0.49it/s, v_num=ohpc, train_loss=0.00225, train_accuracy=0.000796]`Trainer.fit` stopped: `max_epochs=200` reached.
Epoch 199: 100%|███████████████████████████████| 39/39 [01:22<00:00,  0.47it/s, v_num=ohpc, train_loss=0.00225, train_accuracy=0.000796]
Processing...
Done!
Processing...
Done!
Processing...
Done!
Processing...
Done!
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
C:\Users\ernst\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\lightning\pytorch\trainer\configuration_validator.py:72: You passed in a `val_dataloader` but have no `validation_step`. Skipping val loop.
Processing...
Done!
Processing...
Done!
Processing...
Done!
Processing...
Done!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name                | Type             | Params
---------------------------------------------------------
0 | layer               | EctLayer         | 0
1 | training_accuracy   | MeanSquaredError | 0
2 | validation_accuracy | MeanSquaredError | 0
3 | test_accuracy       | MeanSquaredError | 0
4 | loss_fn             | MSELoss          | 0
5 | model               | Sequential       | 39.9 M
---------------------------------------------------------
39.9 M    Trainable params
0         Non-trainable params
39.9 M    Total params
159.433   Total estimated model params size (MB)
C:\Users\ernst\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\lightning\pytorch\trainer\connectors\data_connector.py:441: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=23` in the `DataLoader` to improve performance.
Epoch 199: 100%|████████████████████████████████| 73/73 [02:31<00:00,  0.48it/s, v_num=r4nl, train_loss=0.00217, train_accuracy=0.00258]`Trainer.fit` stopped: `max_epochs=200` reached.
Epoch 199: 100%|████████████████████████████████| 73/73 [02:32<00:00,  0.48it/s, v_num=r4nl, train_loss=0.00217, train_accuracy=0.00258]
```


# RECONSTRUCTION on ALL shapes (values from table)

run scripts/test_ae_all_test.sh

```
distChamferCUDA not available; fall back to slower version.
emd_approx_cuda not available. Fall back to slower version.
Number of trainable parameters of Point CNF: 927513
Resume Path:pretrained_models/ae/all/checkpoint.pt
Total number of data:2832
Min number of points: (train)2048 (test)2048
Total number of data:405
Min number of points: (train)2048 (test)2048
Cate=airplane Total Sample size:torch.Size([405, 2048, 3]) Ref size: torch.Size([405, 2048, 3])
{'MMD-CD': 0.0001573404297232628, 'MMD-EMD': 0.019326867535710335}
Total number of data:58
Min number of points: (train)2048 (test)2048
Total number of data:8
Min number of points: (train)2048 (test)2048
Cate=bag Total Sample size:torch.Size([8, 2048, 3]) Ref size: torch.Size([8, 2048, 3])
{'MMD-CD': 0.0009373911889269948, 'MMD-EMD': 0.04203237593173981}
Total number of data:77
Min number of points: (train)2048 (test)2048
Total number of data:11
Min number of points: (train)2048 (test)2048
Cate=basket Total Sample size:torch.Size([11, 2048, 3]) Ref size: torch.Size([11, 2048, 3])
{'MMD-CD': 0.0010404380736872554, 'MMD-EMD': 0.05076565593481064}
Total number of data:599
Min number of points: (train)2048 (test)2048
Total number of data:85
Min number of points: (train)2048 (test)2048
Cate=bathtub Total Sample size:torch.Size([85, 2048, 3]) Ref size: torch.Size([85, 2048, 3])
{'MMD-CD': 0.000876822043210268, 'MMD-EMD': 0.036324817687273026}
Total number of data:167
Min number of points: (train)2048 (test)2048
Total number of data:23
Min number of points: (train)2048 (test)2048
Cate=bed Total Sample size:torch.Size([23, 2048, 3]) Ref size: torch.Size([23, 2048, 3])
{'MMD-CD': 0.0018063323805108666, 'MMD-EMD': 0.05766409635543823}
Total number of data:1260
Min number of points: (train)2048 (test)2048
Total number of data:185
Min number of points: (train)2048 (test)2048
Cate=bench Total Sample size:torch.Size([185, 2048, 3]) Ref size: torch.Size([185, 2048, 3])
{'MMD-CD': 0.0006830046186223626, 'MMD-EMD': 0.034182071685791016}
Total number of data:340
Min number of points: (train)2048 (test)2048
Total number of data:43
Min number of points: (train)2048 (test)2048
Cate=bottle Total Sample size:torch.Size([43, 2048, 3]) Ref size: torch.Size([43, 2048, 3])
{'MMD-CD': 0.0005666422075591981, 'MMD-EMD': 0.03220720961689949}
Total number of data:125
Min number of points: (train)2048 (test)2048
Total number of data:17
Min number of points: (train)2048 (test)2048
Cate=bowl Total Sample size:torch.Size([17, 2048, 3]) Ref size: torch.Size([17, 2048, 3])
{'MMD-CD': 0.0004775912675540894, 'MMD-EMD': 0.02841782756149769}
Total number of data:628
Min number of points: (train)2048 (test)2048
Total number of data:94
Min number of points: (train)2048 (test)2048
Cate=bus Total Sample size:torch.Size([94, 2048, 3]) Ref size: torch.Size([94, 2048, 3])
{'MMD-CD': 0.0005901968688704073, 'MMD-EMD': 0.034513697028160095}
Total number of data:1076
Min number of points: (train)2048 (test)2048
Total number of data:152
Min number of points: (train)2048 (test)2048
Cate=cabinet Total Sample size:torch.Size([152, 2048, 3]) Ref size: torch.Size([152, 2048, 3])
{'MMD-CD': 0.0011355767492204905, 'MMD-EMD': 0.041815802454948425}
Total number of data:227
Min number of points: (train)2048 (test)2048
Total number of data:34
Min number of points: (train)2048 (test)2048
Cate=can Total Sample size:torch.Size([34, 2048, 3]) Ref size: torch.Size([34, 2048, 3])
{'MMD-CD': 0.0008740086923353374, 'MMD-EMD': 0.040788039565086365}
Total number of data:79
Min number of points: (train)2048 (test)2048
Total number of data:11
Min number of points: (train)2048 (test)2048
Cate=camera Total Sample size:torch.Size([11, 2048, 3]) Ref size: torch.Size([11, 2048, 3])
{'MMD-CD': 0.0017456325003877282, 'MMD-EMD': 0.06239160895347595}
Total number of data:39
Min number of points: (train)2048 (test)2048
Total number of data:5
Min number of points: (train)2048 (test)2048
Cate=cap Total Sample size:torch.Size([5, 2048, 3]) Ref size: torch.Size([5, 2048, 3])
{'MMD-CD': 0.0008072186610661447, 'MMD-EMD': 0.045138198882341385}
Total number of data:2458
Min number of points: (train)2048 (test)2048
Total number of data:352
Min number of points: (train)2048 (test)2048
Cate=car Total Sample size:torch.Size([352, 2048, 3]) Ref size: torch.Size([352, 2048, 3])
{'MMD-CD': 0.0007180275279097259, 'MMD-EMD': 0.03479321673512459}
Total number of data:4612
Min number of points: (train)2048 (test)2048
Total number of data:662
Min number of points: (train)2048 (test)2048
Cate=chair Total Sample size:torch.Size([662, 2048, 3]) Ref size: torch.Size([662, 2048, 3])
{'MMD-CD': 0.0008697320590727031, 'MMD-EMD': 0.040376778692007065}
Total number of data:455
Min number of points: (train)2048 (test)2048
Total number of data:65
Min number of points: (train)2048 (test)2048
Cate=clock Total Sample size:torch.Size([65, 2048, 3]) Ref size: torch.Size([65, 2048, 3])
{'MMD-CD': 0.0007417860324494541, 'MMD-EMD': 0.0396791435778141}
Total number of data:65
Min number of points: (train)2048 (test)2048
Total number of data:9
Min number of points: (train)2048 (test)2048
Cate=dishwasher Total Sample size:torch.Size([9, 2048, 3]) Ref size: torch.Size([9, 2048, 3])
{'MMD-CD': 0.0012714030453935266, 'MMD-EMD': 0.050969988107681274}
Total number of data:762
Min number of points: (train)2048 (test)2048
Total number of data:112
Min number of points: (train)2048 (test)2048
Cate=monitor Total Sample size:torch.Size([112, 2048, 3]) Ref size: torch.Size([112, 2048, 3])
{'MMD-CD': 0.0007173331687226892, 'MMD-EMD': 0.03568132594227791}
Total number of data:5863
Min number of points: (train)2048 (test)2048
Total number of data:842
Min number of points: (train)2048 (test)2048
Cate=table Total Sample size:torch.Size([842, 2048, 3]) Ref size: torch.Size([842, 2048, 3])
{'MMD-CD': 0.0007469682022929192, 'MMD-EMD': 0.03629077598452568}
Total number of data:323
Min number of points: (train)2048 (test)2048
Total number of data:23
Min number of points: (train)2048 (test)2048
Cate=telephone Total Sample size:torch.Size([23, 2048, 3]) Ref size: torch.Size([23, 2048, 3])
{'MMD-CD': 0.0007441925117745996, 'MMD-EMD': 0.040820807218551636}
Total number of data:73
Min number of points: (train)2048 (test)2048
Total number of data:13
Min number of points: (train)2048 (test)2048
Cate=tin_can Total Sample size:torch.Size([13, 2048, 3]) Ref size: torch.Size([13, 2048, 3])
{'MMD-CD': 0.0008074340876191854, 'MMD-EMD': 0.03700706362724304}
Total number of data:84
Min number of points: (train)2048 (test)2048
Total number of data:13
Min number of points: (train)2048 (test)2048
Cate=tower Total Sample size:torch.Size([13, 2048, 3]) Ref size: torch.Size([13, 2048, 3])
{'MMD-CD': 0.00041366522782482207, 'MMD-EMD': 0.035571567714214325}
Total number of data:272
Min number of points: (train)2048 (test)2048
Total number of data:39
Min number of points: (train)2048 (test)2048
Cate=train Total Sample size:torch.Size([39, 2048, 3]) Ref size: torch.Size([39, 2048, 3])
{'MMD-CD': 0.0005977580440230668, 'MMD-EMD': 0.031954724341630936}
Total number of data:45
Min number of points: (train)2048 (test)2048
Total number of data:6
Min number of points: (train)2048 (test)2048
Cate=keyboard Total Sample size:torch.Size([6, 2048, 3]) Ref size: torch.Size([6, 2048, 3])
{'MMD-CD': 0.00037608074489980936, 'MMD-EMD': 0.028975050896406174}
Total number of data:51
Min number of points: (train)2048 (test)2048
Total number of data:7
Min number of points: (train)2048 (test)2048
Cate=earphone Total Sample size:torch.Size([7, 2048, 3]) Ref size: torch.Size([7, 2048, 3])
{'MMD-CD': 0.0014028403675183654, 'MMD-EMD': 0.0706866905093193}
Total number of data:519
Min number of points: (train)2048 (test)2048
Total number of data:75
Min number of points: (train)2048 (test)2048
Cate=faucet Total Sample size:torch.Size([75, 2048, 3]) Ref size: torch.Size([75, 2048, 3])
{'MMD-CD': 0.0010709114139899611, 'MMD-EMD': 0.05047212541103363}
Total number of data:208
Min number of points: (train)2048 (test)2048
Total number of data:30
Min number of points: (train)2048 (test)2048
Cate=file Total Sample size:torch.Size([30, 2048, 3]) Ref size: torch.Size([30, 2048, 3])
{'MMD-CD': 0.0011089699110016227, 'MMD-EMD': 0.05364315211772919}
Total number of data:557
Min number of points: (train)2048 (test)2048
Total number of data:80
Min number of points: (train)2048 (test)2048
Cate=guitar Total Sample size:torch.Size([80, 2048, 3]) Ref size: torch.Size([80, 2048, 3])
{'MMD-CD': 0.0002704861108213663, 'MMD-EMD': 0.0288230013102293}
Total number of data:113
Min number of points: (train)2048 (test)2048
Total number of data:16
Min number of points: (train)2048 (test)2048
Cate=helmet Total Sample size:torch.Size([16, 2048, 3]) Ref size: torch.Size([16, 2048, 3])
{'MMD-CD': 0.0014232280664145947, 'MMD-EMD': 0.04556359350681305}
Total number of data:380
Min number of points: (train)2048 (test)2048
Total number of data:64
Min number of points: (train)2048 (test)2048
Cate=jar Total Sample size:torch.Size([64, 2048, 3]) Ref size: torch.Size([64, 2048, 3])
{'MMD-CD': 0.0009178055333904922, 'MMD-EMD': 0.04619111865758896}
Total number of data:296
Min number of points: (train)2048 (test)2048
Total number of data:43
Min number of points: (train)2048 (test)2048
Cate=knife Total Sample size:torch.Size([43, 2048, 3]) Ref size: torch.Size([43, 2048, 3])
{'MMD-CD': 0.0002676854492165148, 'MMD-EMD': 0.03916255012154579}
Total number of data:1620
Min number of points: (train)2048 (test)2048
Total number of data:232
Min number of points: (train)2048 (test)2048
Cate=lamp Total Sample size:torch.Size([232, 2048, 3]) Ref size: torch.Size([232, 2048, 3])
{'MMD-CD': 0.0010485418606549501, 'MMD-EMD': 0.057339176535606384}
Total number of data:319
Min number of points: (train)2048 (test)2048
Total number of data:45
Min number of points: (train)2048 (test)2048
Cate=laptop Total Sample size:torch.Size([45, 2048, 3]) Ref size: torch.Size([45, 2048, 3])
{'MMD-CD': 0.000495999411214143, 'MMD-EMD': 0.037168435752391815}
Total number of data:1116
Min number of points: (train)2048 (test)2048
Total number of data:158
Min number of points: (train)2048 (test)2048
Cate=speaker Total Sample size:torch.Size([158, 2048, 3]) Ref size: torch.Size([158, 2048, 3])
{'MMD-CD': 0.001242765225470066, 'MMD-EMD': 0.04500648006796837}
Total number of data:65
Min number of points: (train)2048 (test)2048
Total number of data:10
Min number of points: (train)2048 (test)2048
Cate=mailbox Total Sample size:torch.Size([10, 2048, 3]) Ref size: torch.Size([10, 2048, 3])
{'MMD-CD': 0.0012673605233430862, 'MMD-EMD': 0.048368245363235474}
Total number of data:46
Min number of points: (train)2048 (test)2048
Total number of data:7
Min number of points: (train)2048 (test)2048
Cate=microphone Total Sample size:torch.Size([7, 2048, 3]) Ref size: torch.Size([7, 2048, 3])
{'MMD-CD': 0.000534017221070826, 'MMD-EMD': 0.0455770343542099}
Total number of data:107
Min number of points: (train)2048 (test)2048
Total number of data:15
Min number of points: (train)2048 (test)2048
Cate=microwave Total Sample size:torch.Size([15, 2048, 3]) Ref size: torch.Size([15, 2048, 3])
{'MMD-CD': 0.0012019466375932097, 'MMD-EMD': 0.048467040061950684}
Total number of data:235
Min number of points: (train)2048 (test)2048
Total number of data:34
Min number of points: (train)2048 (test)2048
Cate=motorcycle Total Sample size:torch.Size([34, 2048, 3]) Ref size: torch.Size([34, 2048, 3])
{'MMD-CD': 0.0007311570225283504, 'MMD-EMD': 0.03664088994264603}
Total number of data:149
Min number of points: (train)2048 (test)2048
Total number of data:22
Min number of points: (train)2048 (test)2048
Cate=mug Total Sample size:torch.Size([22, 2048, 3]) Ref size: torch.Size([22, 2048, 3])
{'MMD-CD': 0.0010294882813468575, 'MMD-EMD': 0.039187680929899216}
Total number of data:167
Min number of points: (train)2048 (test)2048
Total number of data:24
Min number of points: (train)2048 (test)2048
Cate=piano Total Sample size:torch.Size([24, 2048, 3]) Ref size: torch.Size([24, 2048, 3])
{'MMD-CD': 0.0010249116457998753, 'MMD-EMD': 0.03997168689966202}
Total number of data:67
Min number of points: (train)2048 (test)2048
Total number of data:9
Min number of points: (train)2048 (test)2048
Cate=pillow Total Sample size:torch.Size([9, 2048, 3]) Ref size: torch.Size([9, 2048, 3])
{'MMD-CD': 0.0006235711043700576, 'MMD-EMD': 0.03183767944574356}
Total number of data:185
Min number of points: (train)2048 (test)2048
Total number of data:26
Min number of points: (train)2048 (test)2048
Cate=pistol Total Sample size:torch.Size([26, 2048, 3]) Ref size: torch.Size([26, 2048, 3])
{'MMD-CD': 0.00043144726078025997, 'MMD-EMD': 0.03394182771444321}
Total number of data:420
Min number of points: (train)2048 (test)2048
Total number of data:61
Min number of points: (train)2048 (test)2048
Cate=pot Total Sample size:torch.Size([61, 2048, 3]) Ref size: torch.Size([61, 2048, 3])
{'MMD-CD': 0.001124355592764914, 'MMD-EMD': 0.043870896100997925}
Total number of data:115
Min number of points: (train)2048 (test)2048
Total number of data:16
Min number of points: (train)2048 (test)2048
Cate=printer Total Sample size:torch.Size([16, 2048, 3]) Ref size: torch.Size([16, 2048, 3])
{'MMD-CD': 0.0011861241655424237, 'MMD-EMD': 0.047539204359054565}
Total number of data:46
Min number of points: (train)2048 (test)2048
Total number of data:6
Min number of points: (train)2048 (test)2048
Cate=remote_control Total Sample size:torch.Size([6, 2048, 3]) Ref size: torch.Size([6, 2048, 3])
{'MMD-CD': 0.0004903135704807937, 'MMD-EMD': 0.03016940876841545}
Total number of data:1656
Min number of points: (train)2048 (test)2048
Total number of data:242
Min number of points: (train)2048 (test)2048
Cate=rifle Total Sample size:torch.Size([242, 2048, 3]) Ref size: torch.Size([242, 2048, 3])
{'MMD-CD': 0.00035594054497778416, 'MMD-EMD': 0.03427577018737793}
Total number of data:59
Min number of points: (train)2048 (test)2048
Total number of data:9
Min number of points: (train)2048 (test)2048
Cate=rocket Total Sample size:torch.Size([9, 2048, 3]) Ref size: torch.Size([9, 2048, 3])
{'MMD-CD': 0.0005183418397791684, 'MMD-EMD': 0.04377726465463638}
Total number of data:106
Min number of points: (train)2048 (test)2048
Total number of data:15
Min number of points: (train)2048 (test)2048
Cate=skateboard Total Sample size:torch.Size([15, 2048, 3]) Ref size: torch.Size([15, 2048, 3])
{'MMD-CD': 0.0003893492976203561, 'MMD-EMD': 0.028708722442388535}
Total number of data:2198
Min number of points: (train)2048 (test)2048
Total number of data:330
Min number of points: (train)2048 (test)2048
Cate=sofa Total Sample size:torch.Size([330, 2048, 3]) Ref size: torch.Size([330, 2048, 3])
{'MMD-CD': 0.0008743669604882598, 'MMD-EMD': 0.03711458668112755}
Total number of data:152
Min number of points: (train)2048 (test)2048
Total number of data:22
Min number of points: (train)2048 (test)2048
Cate=stove Total Sample size:torch.Size([22, 2048, 3]) Ref size: torch.Size([22, 2048, 3])
{'MMD-CD': 0.0010414364514872432, 'MMD-EMD': 0.049674879759550095}
Total number of data:1356
Min number of points: (train)2048 (test)2048
Total number of data:194
Min number of points: (train)2048 (test)2048
Cate=vessel Total Sample size:torch.Size([194, 2048, 3]) Ref size: torch.Size([194, 2048, 3])
{'MMD-CD': 0.0007241741404868662, 'MMD-EMD': 0.03899824619293213}
Total number of data:116
Min number of points: (train)2048 (test)2048
Total number of data:17
Min number of points: (train)2048 (test)2048
Cate=washer Total Sample size:torch.Size([17, 2048, 3]) Ref size: torch.Size([17, 2048, 3])
{'MMD-CD': 0.0012653004378080368, 'MMD-EMD': 0.05570317432284355}
Total number of data:404
Min number of points: (train)2048 (test)2048
Total number of data:83
Min number of points: (train)2048 (test)2048
Cate=cellphone Total Sample size:torch.Size([83, 2048, 3]) Ref size: torch.Size([83, 2048, 3])
{'MMD-CD': 0.000496187130920589, 'MMD-EMD': 0.032826244831085205}
Total number of data:51
Min number of points: (train)2048 (test)2048
Total number of data:7
Min number of points: (train)2048 (test)2048
Cate=birdhouse Total Sample size:torch.Size([7, 2048, 3]) Ref size: torch.Size([7, 2048, 3])
{'MMD-CD': 0.001378068351186812, 'MMD-EMD': 0.043016303330659866}
Total number of data:310
Min number of points: (train)2048 (test)2048
Total number of data:50
Min number of points: (train)2048 (test)2048
Cate=bookshelf Total Sample size:torch.Size([50, 2048, 3]) Ref size: torch.Size([50, 2048, 3])
{'MMD-CD': 0.0012648236006498337, 'MMD-EMD': 0.04541373252868652}
================================================================================
All category results:
================================================================================
{'airplane': {'MMD-CD': 0.0001573404297232628, 'MMD-EMD': 0.019326867535710335},
 'bag': {'MMD-CD': 0.0009373911889269948, 'MMD-EMD': 0.04203237593173981},
 'basket': {'MMD-CD': 0.0010404380736872554, 'MMD-EMD': 0.05076565593481064},
 'bathtub': {'MMD-CD': 0.000876822043210268, 'MMD-EMD': 0.036324817687273026},
 'bed': {'MMD-CD': 0.0018063323805108666, 'MMD-EMD': 0.05766409635543823},
 'bench': {'MMD-CD': 0.0006830046186223626, 'MMD-EMD': 0.034182071685791016},
 'birdhouse': {'MMD-CD': 0.001378068351186812, 'MMD-EMD': 0.043016303330659866},
 'bookshelf': {'MMD-CD': 0.0012648236006498337, 'MMD-EMD': 0.04541373252868652},
 'bottle': {'MMD-CD': 0.0005666422075591981, 'MMD-EMD': 0.03220720961689949},
 'bowl': {'MMD-CD': 0.0004775912675540894, 'MMD-EMD': 0.02841782756149769},
 'bus': {'MMD-CD': 0.0005901968688704073, 'MMD-EMD': 0.034513697028160095},
 'cabinet': {'MMD-CD': 0.0011355767492204905, 'MMD-EMD': 0.041815802454948425},
 'camera': {'MMD-CD': 0.0017456325003877282, 'MMD-EMD': 0.06239160895347595},
 'can': {'MMD-CD': 0.0008740086923353374, 'MMD-EMD': 0.040788039565086365},
 'cap': {'MMD-CD': 0.0008072186610661447, 'MMD-EMD': 0.045138198882341385},
 'car': {'MMD-CD': 0.0007180275279097259, 'MMD-EMD': 0.03479321673512459},
 'cellphone': {'MMD-CD': 0.000496187130920589, 'MMD-EMD': 0.032826244831085205},
 'chair': {'MMD-CD': 0.0008697320590727031, 'MMD-EMD': 0.040376778692007065},
 'clock': {'MMD-CD': 0.0007417860324494541, 'MMD-EMD': 0.0396791435778141},
 'dishwasher': {'MMD-CD': 0.0012714030453935266,
                'MMD-EMD': 0.050969988107681274},
 'earphone': {'MMD-CD': 0.0014028403675183654, 'MMD-EMD': 0.0706866905093193},
 'faucet': {'MMD-CD': 0.0010709114139899611, 'MMD-EMD': 0.05047212541103363},
 'file': {'MMD-CD': 0.0011089699110016227, 'MMD-EMD': 0.05364315211772919},
 'guitar': {'MMD-CD': 0.0002704861108213663, 'MMD-EMD': 0.0288230013102293},
 'helmet': {'MMD-CD': 0.0014232280664145947, 'MMD-EMD': 0.04556359350681305},
 'jar': {'MMD-CD': 0.0009178055333904922, 'MMD-EMD': 0.04619111865758896},
 'keyboard': {'MMD-CD': 0.00037608074489980936,
              'MMD-EMD': 0.028975050896406174},
 'knife': {'MMD-CD': 0.0002676854492165148, 'MMD-EMD': 0.03916255012154579},
 'lamp': {'MMD-CD': 0.0010485418606549501, 'MMD-EMD': 0.057339176535606384},
 'laptop': {'MMD-CD': 0.000495999411214143, 'MMD-EMD': 0.037168435752391815},
 'mailbox': {'MMD-CD': 0.0012673605233430862, 'MMD-EMD': 0.048368245363235474},
 'microphone': {'MMD-CD': 0.000534017221070826, 'MMD-EMD': 0.0455770343542099},
 'microwave': {'MMD-CD': 0.0012019466375932097,
               'MMD-EMD': 0.048467040061950684},
 'monitor': {'MMD-CD': 0.0007173331687226892, 'MMD-EMD': 0.03568132594227791},
 'motorcycle': {'MMD-CD': 0.0007311570225283504,
                'MMD-EMD': 0.03664088994264603},
 'mug': {'MMD-CD': 0.0010294882813468575, 'MMD-EMD': 0.039187680929899216},
 'piano': {'MMD-CD': 0.0010249116457998753, 'MMD-EMD': 0.03997168689966202},
 'piano': {'MMD-CD': 0.0010249116457998753, 'MMD-EMD': 0.03997168689966202},
 'pillow': {'MMD-CD': 0.0006235711043700576, 'MMD-EMD': 0.03183767944574356},
 'pillow': {'MMD-CD': 0.0006235711043700576, 'MMD-EMD': 0.03183767944574356},
 'pistol': {'MMD-CD': 0.00043144726078025997, 'MMD-EMD': 0.03394182771444321},
 'pistol': {'MMD-CD': 0.00043144726078025997, 'MMD-EMD': 0.03394182771444321},
 'pot': {'MMD-CD': 0.001124355592764914, 'MMD-EMD': 0.043870896100997925},
 'pot': {'MMD-CD': 0.001124355592764914, 'MMD-EMD': 0.043870896100997925},
 'printer': {'MMD-CD': 0.0011861241655424237, 'MMD-EMD': 0.047539204359054565},
 'printer': {'MMD-CD': 0.0011861241655424237, 'MMD-EMD': 0.047539204359054565},
 'remote_control': {'MMD-CD': 0.0004903135704807937,
                    'MMD-EMD': 0.03016940876841545},
 'rifle': {'MMD-CD': 0.00035594054497778416, 'MMD-EMD': 0.03427577018737793},
 'rocket': {'MMD-CD': 0.0005183418397791684, 'MMD-EMD': 0.04377726465463638},
 'rocket': {'MMD-CD': 0.0005183418397791684, 'MMD-EMD': 0.04377726465463638},
 'skateboard': {'MMD-CD': 0.0003893492976203561,
 'skateboard': {'MMD-CD': 0.0003893492976203561,
                'MMD-EMD': 0.028708722442388535},
 'sofa': {'MMD-CD': 0.0008743669604882598, 'MMD-EMD': 0.03711458668112755},
 'speaker': {'MMD-CD': 0.001242765225470066, 'MMD-EMD': 0.04500648006796837},
 'stove': {'MMD-CD': 0.0010414364514872432, 'MMD-EMD': 0.049674879759550095},
 'table': {'MMD-CD': 0.0007469682022929192, 'MMD-EMD': 0.03629077598452568},
 'telephone': {'MMD-CD': 0.0007441925117745996,
               'MMD-EMD': 0.040820807218551636},
 'tin_can': {'MMD-CD': 0.0008074340876191854, 'MMD-EMD': 0.03700706362724304},
 'tower': {'MMD-CD': 0.00041366522782482207, 'MMD-EMD': 0.035571567714214325},
 'train': {'MMD-CD': 0.0005977580440230668, 'MMD-EMD': 0.031954724341630936},
 'vessel': {'MMD-CD': 0.0007241741404868662, 'MMD-EMD': 0.03899824619293213},
 'washer': {'MMD-CD': 0.0012653004378080368, 'MMD-EMD': 0.05570317432284355}}
================================================================================
Averaged results:
{'MMD-CD': 0.0007546906039317743, 'MMD-EMD': 0.03768912870501908}
================================================================================
```