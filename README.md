# Prediction-Stock-Market-Project (Archived)
## 说明 Note

本repo是先前关于股价预测的一个归档整理，方便日后查阅，团队所得的对应代码供参考,运行需详细调试

This repository is an archived collection of previous work on stock price prediction for future reference. The corresponding code is provided for reference while execution is NOT exactly guaranteed.

## 目录结构（Structure Overview）

```
data/
		HSI_5min.txt
        HSI_Daily.txt
Pred_GRU/
		GRU for 5min_data.py
		GRU for daily_data.py
		README.md
Pred_RandomForest/
		RF_for_5min_date.py
		RF_for_daily_date.py
		README.md
Preprocessing/
		data process_GRU.py
		data process_ML.py
		New_preprocess_data.py
Results/
		Result.html
		Strategies.md
Trade_Analysis/
		Analysis.ipynb
		Trade.py
LICENSE.txt
README.md
```

### 文件夹及文件说明（Folder & File Descriptions）

- **data/**
	- 存放原始数据文件，如HSI_5分钟线.txt。
	- Stores raw data files, e.g., HSI_5min_line.txt.
- **Pred_GRU/**
	- 基于GRU模型的预测相关代码和说明文档。
	- GRU-based prediction scripts and documentation.
- **Pred_RandomForest/**
	- 随机森林模型预测相关代码和说明文档。
	- Random Forest prediction scripts and documentation.
- **Preprocessing/**
	- 数据预处理脚本，包括针对不同模型的数据处理。
	- Data preprocessing scripts for different models.
- **Results/**
	- 存放结果文件和策略说明。
	- Stores result files and strategy documentation.
- **Trade_Analysis/**
	- 交易分析相关脚本和Jupyter笔记本。
	- Trade analysis scripts and Jupyter notebook.
- **Requirements**
    - 供用户所安装使用到的环境配置。
    - The environment configuration for users to install and use.
