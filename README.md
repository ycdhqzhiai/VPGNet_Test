# VPGNet_Test
Forked from [VPGNet](https://github.com/SeokjuLee/VPGNet)

## 1.Install
由于CUDNN版本问题(7.5)，USE_CUDNN=0
```shell
make -j40
make pycaffe
```
## Train
1.生成数据集</br>
目前使用caltech-lanes-dataset数据集训练，为了方便没有matlab的机器，生成lmdb数据txt已上传
```
cd models/vpgnet-novp/
make_lmdb.sh
````
2.训练
```shell
nohup ../../build/tools/caffe train --solver=./solver.prototxt &
tail -f nohup.out
```
## Test
在test文件夹中提供了py和cpp实现</br>
1.python脚本修改test.py对应文件路径</br>
`python test.py`</br>
2.修改run.sh对应路径编译运行
```
./run.sh
./test <net.prototxt> <net.caffemodel> <inputFile_txt>
```
## Reference resources
1.https://github.com/ArayCHN/VPGNet_for_lane
2.https://www.zybuluo.com/vivounicorn/note/872699
