# adversarial-domain-adaptation

Contains the code for performing domain adaptation between different datasets using adversarial neural networks.

## Runs

| params                                                                                                                                                                                 | src    | tgt     | acc_src | acc_tgt |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|---------|---------|---------|
|--epochs 2000 --source_dom svhn.pkl --tgt_dom mnist.pkl --lr 0.001 --batch_size 128 --cuda True --dropout 0.5 --momentum 0.9 --weight_decay 0.0 --normalize zeroone    | SVHN   | MNIST   | 88.82%  | 71.84%  |
|--epochs 400 --source_dom synnum.pkl --tgt_dom svhn.pkl --lr 0.001 --batch_size 128 --cuda True --dropout 0.5 --momentum 0.9 --weight_decay 0.0 --normalize zeroone   | SYNNUM |    SVHN |  99.10% |  90.05% |
|--epochs 300 --source_dom mnist.pkl --tgt_dom mnist_m.pkl --lr 0.001 --batch_size 128 --cuda True --dropout 0.5 --momentum 0.9 --weight_decay 0.0 --normalize zeroone |  MNIST | MNIST_M |  99.11% |  82.71% |
|--epochs 200 --source_dom mnist.pkl --tgt_dom usps.pkl --lr 0.001 --batch_size 128 --cuda True --dropout 0.5 --momentum 0.9 --weight_decay 0.0 --normalize zeroone    |  MNIST |    USPS |  99.41% |  96.27% |

Note: The network should start to adapt after a few epochs of training, in particular for similar-looking datasets. While training for too long can cause performance degradation (due to overfitting) on the source domain, it does not seem to reduce performance on the target domain. In fact, for some settings (e.g., SVHN->MNIST and SYNNUM->SVHN) adaptation still takes place in the late epochs).

## Links

SVHN: <http://ufldl.stanford.edu/housenumbers/>

MNIST: <http://yann.lecun.com/exdb/mnist/>

USPS: <https://git-disl.github.io/GTDLBench/datasets/usps_dataset/>

EMNIST: <https://www.nist.gov/itl/products-and-services/emnist-dataset>

SYNNUM, MNIST_M: <http://yaroslav.ganin.net/> (download links seem to have been removed, consider contacting the author)



