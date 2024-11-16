This is code for the paper "On the Convergence of Adam-Type Algorithms for Bilevel Optimization under Unbounded Smoothness"

#### Requirements
python 3.9, numpy, sklearn, Pytorch>=2.0

Here we provide an examples for running the baselines and our algorithms on Hyper-representatioin with a BERT model.

#### Run bilevel algorithms for Hyper-representatioin:
`    python main.py --methods [algorithm] `

where the argument 'algorithm' can be chosen from [adambo, bo_rep, slip, saba, ma_soba, stocbio, ttsa, maml, anil].