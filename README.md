# 利用遠距課程數據預測學生成績與及格情形

這個專案以遠距課程的學生學習行為資料，研究兩項任務：預測課程最終成績（迴歸），以及預測學生是否及格（二元分類）。另以特徵解釋方法分析各項學習行為對分類模型預測的影響。

## 研究背景與資料

2024 年中部五校教育大數據微學程聯合成果展海報記載，研究使用 110–112 學年、131 門課程、9,629 筆紀錄，涵蓋平台造訪次數、停留時間、影音觀看、作業與測驗提交、討論參與、出缺席等欄位。不同實驗使用不同的欄位篩選方式，因此程式接受包含 `課程成績` 與 `及格與否` 的本地 CSV 檔，並由程式執行前處理。

原始學生資料未附於本公開儲存庫。請自行取得具有合法使用權限的資料，並確認符合研究倫理與隱私規定。

## 專案內容

| 檔案 | 用途 |
| --- | --- |
| `experiments/regression.py` | 比較 Random Forest、Gradient Boosting、AdaBoost、XGBoost、LightGBM、CatBoost、Ridge、Lasso 與 SVR，使用交叉驗證搜尋參數並預測成績 |
| `experiments/ensemble.py` | 整合決策樹、隨機森林、AdaBoost、梯度提升、KNN、邏輯迴歸、SVM、XGBoost 與 LightGBM，以軟投票進行及格分類 |
| `experiments/shap_analysis.py` | 以 XGBoost 預測及格與否，輸出 SHAP 特徵重要性、摘要圖、瀑布圖及依賴圖 |
| `experiments/common.py` | 共用 CSV 載入、欄位處理、模型評估與 JSON 輸出 |
| `logistic_regression.py`、`教育大數據MLP版.py` | 原先已公開的歷史實驗程式，保留以維持原始研究紀錄 |

`experiments/` 為後續整理的可重現版本，與 2024 年海報當時使用的實驗版本及結果可能不同。原有程式未在本次重整中覆寫。

## 安裝及執行

建議使用 Python 3.10–3.12，在儲存庫根目錄執行：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m experiments.regression --data data/test.csv --output results/regression
python -m experiments.ensemble --data data/test.csv --output results/ensemble
python -m experiments.shap_analysis --data data/test.csv --output results/shap
```

可以先用較小的搜尋範圍驗證執行：

```bash
python -m experiments.regression --data data/test.csv --models ridge random_forest --cv 3 --n-iter 3
python -m experiments.ensemble --data data/test.csv --cv 3 --n-iter 3
python -m experiments.shap_analysis --data data/test.csv --max-samples 100
```

迴歸輸出 `cv_results.csv`、`test_metrics.json`、`best_model.joblib`；融合模型輸出分類評估與模型檔；SHAP 分析輸出分類評估、`feature_importance.csv` 和三種圖表。所有輸出預設存入已忽略的 `results/` 目錄。

## 評估設計

- 在訓練集內執行缺失值填補、標準化與交叉驗證搜尋；測試集僅用於最後評估。
- 迴歸模型以訓練資料的交叉驗證 R² 選型，再報告獨立測試集的 R²、MAE、MSE、RMSE 與 MAPE。
- 融合模型以訓練資料的交叉驗證 Macro-F1 調整 KNN 參數；測試集報告 Accuracy、Balanced Accuracy、Macro-F1、每類召回率及 ROC-AUC。
- 為避免直接目標洩漏，分類時排除 `課程成績`、`分數等級`；迴歸時排除 `及格與否`、`分數等級`。兩個任務均排除 `cou_code`。
- XGBoost／SHAP 的訓練樣本以類別權重減少及格／不及格比例失衡的影響，SHAP 圖使用留出的測試資料產生。

由於目前資料缺少可核對的學生唯一識別碼，以及統一的預測時間點，這個版本仍採紀錄層級的隨機切分，不能證明對全新學生、跨學期或預警情境的泛化能力。部分出席、作業和其他課程期末資訊也可能在實際早期預測時尚未取得。

**請勿直接將成果展海報的 100%、97.12%、96.50%、R² 0.6121 等數字當作這份整理版程式重新執行的結果。** 正式效能以執行後產生的測試指標為準。

## 團隊與貢獻

本研究為團隊合作專案，指導老師林俊淵及組員謝東霖、傅澤雨、洪紹棋、張秉暉、吳豐宇。這個儲存庫由吳豐宇維護；為個人想記錄其參與經歷和編成經驗所撰寫，僅上傳自身有操作項目。其餘實驗與貢獻依團隊原始研究紀錄歸屬。

海報與歷史程式僅作研究背景參照；整理版的模型與驗證流程已作必要調整，以利閱讀與重現。
