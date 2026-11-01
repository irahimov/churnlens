# ChurnLens

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

> Churn və propensity modelləri ilə "kimə zəng edək?" sualına cavab. Zəng siyahısının iqtisadiyyatı əsas göstəricidir, AUC deyil.

🎯 **Əsas nəticə:** Telco-da ən yuxarı 10%-ə zəng edəndə churn edənlərin **73.7%**-i tutulur (lift **2.78×**).

## Problem

ChurnLens bankda müştəri itkisinin (churn) qarşısını almaq üçün qurulub. Nümunə ssenari: bankda 100,000 müştəri var və hamısına tək-tək zəng etmək mümkün deyil, çünki resurs məhduddur. Model hər müştəriyə churn ehtimalı skoru (churn probability score, $p \in [0, 1]$) təyin edir və müştəriləri risk sırasına düzür. Resurslar (zəng mərkəzi, xüsusi kampaniyalar) yalnız ən yuxarı risk qrupuna, məsələn top 10%-ə yönəldilir.

Churn analitikasında geniş yayılmış yanlış fikir budur: "Bankı tərk etməyəcək müştərilərə resurs ayırmırıq, yalnız gedəcəklərə fokuslanırıq." Bu yanaşma natamamdır. Müştərilər üç qrupa bölünür:

1. **Sadiq müştərilər (Safe / Loyal):** Nə etsən də bankda qalacaqlar. Bunlara zəng edib endirim və ya hədiyyə təklif etmək bankın pulunu boşa xərcləməkdir.
2. **İtirilmiş müştərilər (Lost Causes):** Artıq qərarını verib, başqa banka keçib və ya xidmətdən tam narazıdır. Nə təklif etsən də qalmayacaq. Bunlara zəng etmək operatorun vaxtını boşa xərcləməkdir.
3. **Tərəddüddə olanlar / risklilər (Persuadables / At-Risk):** ChurnLens-in ovlamalı olduğu kütlə budur. Churn ehtimalı yüksəkdir ($p > 0.6$), amma doğru vaxtda doğru təklif (Next Best Offer / Product) edilsə, bankda qalacaqlar.

ChurnLens-in əsas məqsədi məhz bu üçüncü qrupu ayırmaqdır.

### Xərc balansı

- **Yanlış müsbət (False Positive, Type I):** Müştəri qalacaqdı, amma ona zəng edildi. Bankın xərci: operatorun 2 dəqiqəlik zəng vaxtı + 10 AZN-lik kiçik bonus və ya endirim (nümunə rəqəmlər). Maliyyət aşağıdır.
- **Yanlış mənfi (False Negative, Type II):** Müştəri gedəcəkdi, amma model onu tapmadı. Bankın xərci: müştərinin depozitləri, kredit faizləri və gələcək bütün mənfəəti, yəni CLV (Customer Lifetime Value), tamamilə itirilir. Maliyyət yüksəkdir.

Buna görə modelin uğuru yalnız AUC ilə deyil, bu xərc balansı ilə ölçüləcək.

Data: Telco (Kaggle, blastchar/telco-customer-churn) və Bank Marketing (UCI). Repo-da data yoxdur, data/raw/ altına qoyulmalıdır.
Necə işlədilir: pip install -r requirements.txt, sonra notebook-ları sırayla işlət.
Metod və nəticələr: Faza 1–8-in markdown nəticələri, əsas cədvəl (Telco və Bank müqayisəsi).
Məhdudiyyətlər: snapshot data, proxy bölmə, fərziyyəli biznes hesabı (save rate, CLV = aylıq × 12), propensity ≠ uplift, PSI eşikləri sənaye təcrübəsidir.
Nə deyil: production sistemi deyil, deployment yoxdur, real bank datası yoxdur.

## Necə işlədilir

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m churnlens.train      # model, scored CSV, top-k cədvəli
PYTHONPATH=src uvicorn churnlens.api:app       # API: GET /health, POST /score
pytest                                         # testlər
python sql/run_queries.py                      # DuckDB sorğuları
streamlit run app/dashboard.py                 # dashboard
```

`requirements.txt` birbaşa asılılıqları, `requirements.lock.txt` tam pinli siyahını saxlayır.

## Nəticələr (Telco, 5-fold stratified CV, out-of-fold)

- ROC-AUC: 0.8419 ± 0.0120
- Model B feature-ları: `tenure`, `MonthlyCharges` və 11 kateqorial sütun (`Contract`, `PaymentMethod`, `PaperlessBilling` daxil)

| Top | Zəng sayı | Precision | Recall | Lift | Value lift | Gözlənilən dəyər (AZN) |
|---|---|---|---|---|---|---|
| 5% | 352 | 0.801 | 0.151 | 3.02 | 3.54 | 80,719 |
| 10% | 704 | 0.737 | 0.278 | 2.78 | 3.18 | 150,109 |
| 20% | 1,408 | 0.666 | 0.501 | 2.51 | 2.79 | 263,807 |

## Dəyər və fərziyyələr

- **Dəyər (Telco):** `MonthlyCharges × 12` illik gəlirdir, lifetime value deyil. Müddət, endirim və mənfəət marjası nəzərə alınmayıb.
- **Gözlənilən dəyər:** top-k üzrə `Σ (p × dəyər × SAVE_RATE − CALL_COST)`. `SAVE_RATE = 0.30` və `CALL_COST = 10 AZN` fərziyyədir, sınaqla ölçülməlidir. Nəticəyə görə tune edilməyib.
- **Value lift:** top-k-nın tutduğu churn-çülərin dəyər payı, təsadüfi seçimdə gözlənilən pay (`fraction`) ilə bölünür. Təsadüfi pay təxminidir, çünki dəyər müştərilər arasında bərabər paylanmır.
- **Bank dəyəri (planlanıb):** `max(balance, 0)` proxy-si. Mənfi balans overdraft sayılır və 0 qəbul olunur. Bu illik gəlir deyil, anlıq vəziyyətdir.

## Məhdudiyyətlər

- Telco snapshot-dır, zaman sütunu yoxdur. Out-of-time test mümkün deyil.
- Proxy bölmə (`tenure`) üzrə Faza 8-də kalibrasiya pozulması göstərilib: yeni müştərilərdə ehtimallar aşağı qiymətləndirilir. Bu seqmentdə ehtimal dəyəri xərc hesabında istifadə olunmamalıdır.
- Public data istifadə olunub, real bank nəticəsi deyil. Xərc və save rate fərziyyədir.
- Propensity ≠ uplift: yüksək ehtimallı müştərilərin bir hissəsi zəng olmadan da churn etməyəcək. Real təsir A/B test ilə ölçülməlidir.
- `TotalCharges` modeldə yoxdur, çünki `tenure × MonthlyCharges` ilə təxminən eynidir.

## Nə deyil

Production sistemi deyil: deployment, auth, monitorinq pipeline-ı və model registry yoxdur. API yalnız lokal işlətmək üçündür.
