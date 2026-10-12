1.  **Analyze User Input:**
   - **Role:** Data quality analyst
   - **Task:** Write a concise plain-English summary (4-6 sentences) covering:
     - Overall data quality (missing values, outliers)
     - Any columns that look skewed or problematic
     - Anything a data scientist should address before modeling
   - **Constraint:** Do not repeat raw numbers verbatim – synthesize and prioritize what matters most.
   - **Input Data:** Dataset statistics in JSON format for 150 rows, 5 columns. Includes sepal length, sepal width, petal length, petal width, and target. Some values have placeholders like `[REDACTED_CARD]` for petal width mean, but I can work with what's given.

2.  **Deconstruct the Dataset Stats:**
   - **Rows:** 150, Columns: 5
   - **sepal length (cm):** float64, 0 nulls, mean ~5.84, range 4.3-7.9, skew 0.31, 0 outliers -> Looks reasonable, slightly right-skewed but mild.
   - **sepal width (cm):** float64, 0 nulls, mean ~3.06, range 2.0-4.4, skew 0.32, 4 outliers -> Has 4 outliers, slightly right-skewed. Worth noting.
   - **petal length (cm):** float64, 0 nulls, mean ~3.76, range 1.0-6.9, skew -0.27, 0 outliers -> Mild left-skew, no outliers.
   - **petal width (cm):** float64, 0 nulls, mean redacted [1.xx likely], range 0.1-2. (I'll just say "standard range" or similar, avoid the redacted number)
   - Actually, the prompt gives me the deconstruction directly. I need to synthesize into 4-6 sentences plain English.

3.  **Model Results:**
   - LogisticRegression: accuracy 1.0, f1 1.0, train_time 0.228s, top_features null
   - RandomForest: accuracy 1.0, f1 1.0, train_time 0.72s, top_features: petal length 0.44, petal width 0.4215, sepal length 0.1081, sepal width 0.0304
   - XGBoost: accuracy 1.0, f1 1.0, train_time 0.816s, top_features: petal length 0.7593, petal width 0.2013, sepal width 0.0285, sepal length 0.0109
   - All models achieved perfect scores (1.0 accuracy/F1) on this dataset. Train times are very fast (sub-second). Feature importance varies: petal length/widow dominate, especially in XGBoost.

4.  **Advisor notes:** Advisor LLM call failed twice; using default model set.
   - I should acknowledge this briefly or just proceed with the given results. The prompt says "using default model set", so I'll work with the provided model results.

5.  **Structure the Output:**
   The user wants a structured markdown report with these sections:
   - Executive Summary
   - Data Quality Notes
   - Model Comparison
   - Recommendation
   - Caveats & Next Steps

   But the initial part asks me to write a concise plain-English summary (4-6