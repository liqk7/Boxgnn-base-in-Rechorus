# Boxgnn-base-in-Rechorus
使用`Rechorus`框架中实现`BoxGNN`推荐算法

本仓库为中山大学-机器学习期末大作业

任务：使用`Rechorus`框架实现一篇论文中的推荐算法

本仓库贡献：成功在`Rechorus`框架中实现了`BoxGNN`算法

主要贡献体现在`src\models\BoxGNN.py`中

```

```bash
python main.py --model_name BoxGNN --epoch 100 --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food --test_all 1

```
由于实验采用的数据Grocery_and_Gourmet_Food在BoxGNN算法并不适合，所以在预处理等方面做的不是很好以及增加了目前看来不必要的操作，需要进行修改和优化，目前实验结果表现一般。另一方面可能是由于学习率等各种参数没有设置好，损失值的变化一直不是很大，后续仍然需要尝试和修改。
#运行结果：

## Top-k Recommendation on Grocery_and_Gourmet_Food

| Model                                                                                             | HR@5   | NDCG@5 | 
|:------------------------------------------------------------------------------------------------- |:------:|:------:|
| BoxGNN            | 1.000 | 0.6309 |

