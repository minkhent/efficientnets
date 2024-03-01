# efficientCV

Implementations of computational efficient architectures ( CV )

### Paper list

* MobileNet V1  [https://arxiv.org/abs/1704.04861]
* MobileNet V2  [https://arxiv.org/abs/1801.04381]
* EfficientNet V1  [https://arxiv.org/abs/1905.11946]
* EfficientNet V2  [https://arxiv.org/abs/2104.00298]

# Install package

```
pip install efficientCV
```
 
# Check model's MACs 

```
pip install ptflops
```
```
from ptflops import get_model_complexity_info

def calculate_MACs(model):
    """MACs stand for Multiply–accumulate operation introduced on MobileNetV
    to compute model complexity."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    macs, params = get_model_complexity_info(
        model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))

```

```
from efficientCV.models import EfficientNetV2

efficient_v2s_config = {
    # expansion_factor, kernel_size, stride, ,channels, no_of layers
    "FusedMvConv_1": (1, 3, 1, 24, 2),
    "FusedMvConv_2": (4, 3, 2, 48, 4),
    "FusedMvConv_3": (4, 3, 2, 64, 4),
    "MvConv_1": (4, 3, 2, 128, 6),
    "MvConv_2": (6, 3, 1, 160, 9),
    "MvConv_3": (6, 3, 2, 256, 15),
}
efficient_v2_s =  EfficientNetV2(efficient_v2s_config)
calculate_MACs(efficient_v2_s)
```
