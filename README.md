特点：基于 Vision Transformer 的生成模型改造，用于流场预测任务。结合了 vision transformer 、 swim transformer 等相关工作。

详细信息可参阅文章

Jundou Jiang, Guanxiong Li, Yi Jiang, Laiping Zhang, Xiaogang Deng,
TransCFD: A transformer-based decoder for flow field prediction,
Engineering Applications of Artificial Intelligence,
Volume 123, Part B,
2023,
106340,
ISSN 0952-1976,
https://doi.org/10.1016/j.engappai.2023.106340.
(https://www.sciencedirect.com/science/article/pii/S0952197623005249)
Abstract: The computational fluid dynamics (CFD) method is computationally intensive and costly, and evaluating aerodynamic performance through CFD is time-consuming and labor-intensive. For the design and optimization of aerodynamic shapes, it is essential to obtain aerodynamic performance efficiently and accurately. This paper proposed TransCFD, a Transformer-based decoding architecture for flow field prediction. The aerodynamic shape is parameterized and used as input to the decoder, which learns an end-to-end mapping between the shape and the flow fields. Compared with the CFD method, the TransCFD was evaluated to have a mean absolute error (MAE) of less than 1%, increase the speed by three orders of magnitude, and perform very well in generalization capability. The method simplifies the input requirements compared to most existing methods. Although the object of this work is a two-dimensional airfoil, the setup of this scheme is very general. TransCFD is promising for rapid aerodynamic performance evaluation, with potential applications in accelerating the aerodynamic design.
Keywords: Transformer; Flow field prediction; Computational fluid dynamics
