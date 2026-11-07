# 遠距課程學生成績預測與學習行為分析

這是我將過去參與撰寫的教育大數據研究程式整理成可閱讀、可測試的個人作品集。主要任務包含預測課程成績（迴歸）、預測學生是否及格（二元分類），以及解釋學習行為特徵對預測的貢獻。

## 研究資料

研究使用 110–112 學年的遠距課程學習紀錄，成果展海報記載 131 門課程、9,629 筆紀錄。資料欄位涵蓋平台造訪、停留時間、影音觀看、作業與測驗繳交、討論互動與出缺席等學習行為。

為保護資料隱私，這個公開儲存庫不包含學生原始資料。請使用已取得合法授權、且具有 `課程成績` 或 `及格與否` 目標欄位的 CSV 進行實驗。

## 程式內容

| 程式 | 功能 |
| --- | --- |
| `experiments/mlp.py` | PyTorch 多層感知器（MLP）及格分類、驗證集提前停止、分類門檻選取、置換特徵重要性 |
| `logistic_regression.py` | 原始邏輯迴歸分類實驗 |
| `experiments/regression.py` | 隨機森林、梯度提升、AdaBoost、XGBoost、LightGBM、CatBoost、Ridge、Lasso 與 SVR 的成績迴歸及參數搜尋 |
| `experiments/ensemble.py` | 多分類器組合融合（CFT）及交叉驗證 |
| `experiments/shap_analysis.py` | 使用 XGBoost 與 SHAP 分析及格分類的特徵貢獻 |
| `experiments/common.py` | 共用資料讀取、欄位篩選、評估指標與結果輸出 |
| `Multilayer Perceptron.py` | 舊版 MLP 檔名相容入口 |
| `tests/` | 共用流程與 MLP 測試 |

專案只收錄我實際參與撰寫或後續整理的實驗程式。各模組為整理後的實作版本，與當年的展示版本可能不同。

## 安裝

建議使用 Python 3.10–3.12：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

將已獲授權的資料自行放在 `data/test.csv`。

## 執行

```bash
python -m experiments.mlp --data data/test.csv --output results/mlp
python -m experiments.regression --data data/test.csv --output results/regression
python -m experiments.ensemble --data data/test.csv --output results/ensemble
python -m experiments.shap_analysis --data data/test.csv --output results/shap
```

快速測試：

```bash
python -m experiments.mlp --data data/test.csv --epochs 20 --patience 5 --importance-repeats 0
python -m experiments.regression --data data/test.csv --models ridge random_forest --cv 3 --n-iter 3
python -m experiments.ensemble --data data/test.csv --cv 3 --n-iter 3
python -m experiments.shap_analysis --data data/test.csv --max-samples 100
python -m unittest discover -s tests -v
```

MLP 輸出模型權重、前處理器、訓練紀錄、混淆矩陣指標、ROC 曲線與置換特徵重要性；迴歸與融合模型輸出最佳模型和測試指標；SHAP 輸出特徵重要性表格及視覺化。結果預設儲存於 `results/`，不會加入版本控制。

## 實驗評估

- 缺失值填補、標準化與模型參數搜尋僅使用訓練資料；測試集保留到最後評估。
- 及格分類排除 `課程成績`、`分數等級`；成績迴歸排除 `及格與否`、`分數等級`，兩者皆排除 `cou_code`。
- MLP 保留 100→50 的隱藏層與 BatchNorm、Dropout，加入訓練集類別加權、獨立驗證集與提前停止；分類門檻由驗證資料選擇。
- 迴歸以交叉驗證 R² 進行模型選擇；分類報告 Macro-F1、Balanced Accuracy、各類召回率及 ROC-AUC。
- 置換重要性及 SHAP 用來說明模型使用特徵的方式，不能直接視為因果影響。

目前資料缺少可核對的學生唯一識別碼與統一預測時間點，實驗使用紀錄層級隨機切分。這些結果尚不能代表跨學生、跨學期或課程早期預警的泛化能力。

**海報所列的歷史數字屬於當時展示結果，不能當作整理後程式重新執行的效能。** 請以正式重新訓練得到的測試集指標為準。
