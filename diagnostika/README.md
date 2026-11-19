# diagnostika — երկփուլ դերմատոսկոպիկ վերլուծություն

> **Հետազոտական նախատիպ է, ոչ թե բժշկական ախտորոշիչ սարք։** Արդյունքը աջակցում է, բայց չի փոխարինում
> բժշկի դատողությանը։ Շեմը և բոլոր կլինիկական արժեքները պետք է վերանայվեն կլինիցիստի կողմից։

## 1. Նպատակը

Համակարգը դերմատոսկոպիկ նկարի (ամբողջ պատկեր կամ բժշկի ընտրած հատված) համար տալիս է.

1. **բարորակ / չարորակ** գնահատական (փուլ 1, չորս մոդելի անսամբլ);
2. **բարորակ ենթատիպի բաշխում** (փուլ 2)՝ `bdb`, `cb`, `db`, `srjd` և, ըստ կոնֆիգուրացիայի, `other_benign`;
3. ուղեգրման (referral) դրոշ, «անորոշ» դրոշ և մոդելների համաձայնության թիվ։

Վեբ ինտերֆեյսը (`app/`) թույլ է տալիս բժշկին ընտրել հատված, գրանցել սեփական ախտորոշումը և ներբեռնել
CSV մատյան։ Պատկերները մշակվում են միայն հիշողության մեջ և չեն պահվում։

## 2. Երկփուլ սխեման

```
          պատկեր  (ամբողջ կամ ROI → crop → 224×224, ImageNet նորմալացում)
                    │
                    ▼
 ┌────────────────────────────────────────────────────────────┐
 │ ՓՈՒԼ 1 — բարորակ vs չարորակ                                  │
 │ ResNet50 · DenseNet121 · EfficientNet-B7 · ViT-B/16 (timm)  │
 │ p_malignant = միջին չորս մոդելի P(malignant)                 │
 └────────────────────────────────────────────────────────────┘
                    │                          │
        p_malignant ≥ շեմ (validation)?         │
          այո → ՈՒՂԵԳՐՈՒՄ (referral)           │
                    │                          │
                    ▼                          ▼
 ┌────────────────────────────────────────────────────────────┐
 │ ՓՈՒԼ 2 — բարորակ ենթատիպ (նույն չորս ճարտարապետությունը)      │
 │ bdb · cb · db · srjd · [other_benign]                       │
 │ E = average  կամ  weighted (validation F1-ով)               │
 └────────────────────────────────────────────────────────────┘
                    │
                    ▼
 P(malignant) = p ;   P(benign_c) = (1 − p) · E_c      (գումարը = 1)
 referral = p ≥ շեմ ;  unclear = max(P) < 0.5 ;  agreement = 0…4
```

Երբ `referral = true`, ենթատիպի բաշխումը վերադարձվում է, բայց նշվում է որպես «միայն տեղեկատվական»։

## 3. Տվյալներ և մեջբերում

- **DERM12345** — 12 345 դերմատոսկոպիկ պատկեր, 1 627 հիվանդ, 40 ենթադաս։ Պաշտոնական բաժանում՝
  train 9 860 / test 2 485։
- Մեջբերում. *DERM12345*, Scientific Data, 2024. DOI: [10.1038/s41597-024-04104-3](https://doi.org/10.1038/s41597-024-04104-3).
  Պատկերների աղբյուր՝ [ISIC archive](https://www.isic-archive.com/).
- Տվյալները **չեն ներառված** ռեպոզիտորիայում (նաև՝ manifest-ները և կշիռները)։ Ներբեռնեք Kaggle-ից և
  ցույց տվեք `DIAG_RAW_DIR`-ով։
- Լիցենզիան ստուգեք նախքան կոմերցիոն օգտագործումը։

## 4. Ինչպես վերարտադրել (հրամանները հերթականությամբ)

Աշխատում է Python 3.11-ով (կախվածությունները ֆիքսված են `requirements.txt`-ում)։

```bash
cd diagnostika
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Kaggle-ի տվյալները (չեն պահվում ռեպոզիտորիայում). տեղը նշեք միջավայրի փոփոխականով
export DIAG_RAW_DIR=/path/to/DERM12345
# Kaggle-ից ներբեռնելու համար՝ նշեք միայն միջավայրում. KAGGLE_USERNAME, KAGGLE_KEY (երբեք ֆայլում)

python -m src.data.prepare --config config.yaml                 # manifest-ներ, dedup, հիվանդների բաժանում
python -m src.stage1_binary.train --config config.yaml          # 4 մոդել, բարորակ/չարորակ
python -m src.stage2_subtype.train --config config.yaml         # 4 մոդել, ենթատիպ
python -m src.stage1_binary.threshold --config config.yaml      # շեմ՝ միայն validation-ով
# ՔԱՅԼ: կլինիցիստը վերանայում է շեմը և target_sensitivity-ը, ապա միայն հետո՝
python -m src.evaluation.evaluate --config config.yaml          # ՄԵԿ անգամ, test-ի վրա
pytest                                                          # միավորի և ինտեգրման թեստեր
python -m app.main                                              # http://127.0.0.1:8000
```

`evaluate`-ը հրաժարվում է աշխատել, եթե նույն կոնֆիգով և manifest-ով արդեն կա արդյունք (`--force-rerun`-ով
կարելի է շրջանցել, բայց դա պետք է փաստաթղթավորվի)։ Ուսուցման, շեմի և գնահատման քայլերը ստուգում են manifest-ի SHA-256-ը և հրաժարվում են աշխատել, եթե այն չի համընկնում։

## 5. Որտեղ են կշիռները

`weights/` թղթապանակը (`*.pt`) և `runs/` թղթապանակը **չեն** կոմիտվում։ Կշիռները ստեղծվում են
վերևի քայլերով կամ տեղափոխվում են արտաքին պահոց՝
[կշիռների տեղը — TODO: հղումը կավելացվի](#)։
Ծրագիրը սկսելիս, եթե չկան՝ տրվում է հստակ սխալ՝ թվարկելով բացակայող ֆայլերը։

## 6. Արդյունքներ

**Այս տարբերակում արդյունքներ դեռ չկան.** DERM12345-ը և նախապես վերապատրաստված (ImageNet) կշիռները
ներբեռնվում են Kaggle-ից և Hugging Face-ից, և այս միջավայրից դրանք հասանելի չեն եղել։ Ստորև աղյուսակը
լրացվում է `src.evaluation.evaluate`-ի արդյունքից (`reports/results.md`, `runs/evaluation/<run_id>/metrics.json`)։

| Մոդել / անսամբլ | Զգայունություն | Մասնավորություն | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| ResNet50 | — | — | — | — |
| DenseNet121 | — | — | — | — |
| EfficientNet-B7 | — | — | — | — |
| ViT-B/16 | — | — | — | — |
| **Անսամբլ (average)** | — | — | — | — |

Շեմը (`threshold`), ընտրված validation-ով 0.90 զգայունության նպատակով, — **դեռ չի հաշվարկվել**։
Ենթատիպի (փուլ 2) և կասկադի աղյուսակները նույնպես դեռ չկան։

Ամեն գնահատականի կողքին նշվում է 95% CI (հիվանդներով bootstrap, 1000 վերանմուշ)։
`srjd`-ն (Spitz/Reed) test-ում ունի **n = 4** պատկեր, ուստի նրա գնահատականները անկայուն են։

## 7. Սահմանափակումներ

- **Հետազոտական նախատիպ է, ոչ բժշկական սարք։** Չունի կլինիկական վավերացում և CE/FDA հավանություն։
- Միայն մեկ տվյալների բազա (DERM12345)։
- Բնակչություն. ըստ տվյալների բազայի նկարագրության՝ թուրքական բնակչություն, ուստի չի ընդհանրացվում այլ
  ռասայական, տարիքային կամ սարքավորումային պայմաններում։
- **Արտաքին վավերացում չկա.** Թեստը նույն տվյալների բազայի պաշտոնական test-բաժինն է։
- `srjd`-ի test-ում միայն 4 պատկեր է։
- `indeterminate` malignancy-ով պատկերները (58) բացառված են թե՛ ուսուցումից, թե՛ գնահատումից։
- ROI կտրվածքները դուրս են մոդելների վերապատրաստման բաշխումից (ամբողջ նկարներով են վերապատրաստված)։
  Փոքր ROI-ի դեպքում համակարգը զգուշացում է տալիս։
- Շեմը ընտրվում է **միայն validation-ով**. այն պետք է վերանայվի կլինիցիստի կողմից։ Test-ը երբեք
  չի օգտագործվում շեմի ընտրության համար։
- Հիվանդների մակարդակի բաժանումը և պատկերների մակարդակի dedup-ը կիրառված են, բայց տվյալների
  ստուգված հայտնի սխալները (օրինակ՝ նույն հիվանդի պատկերներ տարբեր բաժիններում) միշտ չէ, որ կարելի է
  ամբողջությամբ բացառել։

## 8. Լիցենզիա

Կոդը՝ հետազոտական նպատակով։ DERM12345-ի և ISIC-ի պատկերների լիցենզիաները չեն վերաբաշխվում այստեղ.
դրանք ստուգեք նախքան որևէ կոմերցիոն կամ վերաբաշխման օգտագործում։ [TODO: ընտրել կոդի լիցենզիան]

## 9. Որոշումների գրանցամատյան (Decisions)

Սա այն դեպքերի ցանկն է, որտեղ աղբյուրները (`Analysis-of-CNN-and-ViT/derm_research/`, Colab-ի
նախագիծ, առաջադրանքի նշումները) չեն համընկնում, և թե ինչ է ընտրվել։

### 9.1. Հայտնաբերված անհամապատասխանություններ

| # | Աղբյուր | Անհամապատասխանություն | Ընտրություն / ինչ արված է |
|---|---|---|---|
| 1 | `DERM12345_TwoStage_Colab.ipynb` | Նոթատետրը ներբեռնված չէր. ո՛չ ռեպոզիտորիայում, ո՛չ բոլոր ճյուղերում, ո՛չ միջավայրում չգտնվեց։ | Փուլ 1-ի և 2-ի դիզայնը վերցված է առաջադրանքի նշումներից։ Նոթատետրի հետ համեմատությունը կատարված չէ։ **Բաց։** |
| 2 | Colab և `derm_training.py`, 02A–02D | Colab-ը օգտագործում է `torchvision`, իսկ 02A–02D-ն ու `derm_training.py`-ը՝ `timm`։ | `timm` (համապատասխանում է գոյություն ունեցող 02A–02D checkpoint-ներին)։ `torchvision`-ը օգտագործվում է միայն ձևափոխությունների համար։ |
| 3 | Առաջադրանքի նշումներ / 02D | ViT-ի անունը. նշումներում՝ `vit_b_16`, 02D-ում՝ `vit_base_patch16_224` (timm)։ | Կոնֆիգում `vit_b_16` կոնֆիգուրացիայի բանալի է, timm-ի ճարտարապետությունը՝ `vit_base_patch16_224`։ |
| 4 | 02A–02D (BATCH_SIZE) | Առաջադրանքի նշումներ՝ 32/32/4/16 (ResNet50/DenseNet121/EfficientNet-B7/ViT)։ Notebook-ներ՝ 8/8/2/4։ | Օգտագործված են **առաջադրանքի նշումների** արժեքները։ Եթե GPU հիշողությունը չի բավականացնում, փոխեք `training.models`-ում և գրանցեք։ |
| 5 | `README.md` (Analysis) և 02A–02D | Fine-tune epochs՝ 8 (README-ն «up to 8»), առաջադրանքի նշումներ՝ 12։ | 12 (առաջադրանքի նշումներ)։ |
| 6 | `derm_training.py`, README | Checkpoint ընտրությունը՝ validation **macro F1** (բոլոր չորս դասերի համար)։ Առաջադրանքի նշումներ՝ փուլ 1-ում՝ **ROC-AUC**։ | Փուլ 1՝ ROC-AUC, փուլ 2՝ macro-F1 (ինչպես նշումներում)։ |
| 7 | `derm_training.py` | Augmentation. Հետազոտությունը՝ horizontal + **vertical** flip, rotate 15, ColorJitter(0.08, 0.08, 0.05)։ Նշումները նշում են միայն «flip»։ | Պահված է և horizontal, և vertical flip (ինչպես հետազոտությունում)։ `augment.vertical_flip` կոնֆիգով կարելի է անջատել։ |
| 8 | `derm_training.py` | Class weights՝ 1/√n, **նորմալացված mean = 1**։ Նշումները չեն նշում նորմալացումը։ | Նորմալացված է mean = 1 (բեռնվածության մասշտաբը չի փոխվում հարաբերական կշիռները)։ |
| 9 | `01_Preprocessing.ipynb` | Validation բաժանում. Հետազոտությունը՝ `GroupShuffleSplit` seed-ների որոնմամբ (42…1041)՝ մինչև չորս դասի առկայությունը։ Նշումները՝ stratified, patient-grouped, seed 42։ | `StratifiedGroupKFold(n_splits=5, shuffle, seed 42)`, առաջին fold-ը՝ validation (≈20%)։ Ստուգվում է, որ վալիդացիայում բոլոր դասերը կան, առանց seed-ների որոնման։ |
| 10 | `01_Preprocessing.ipynb` | Տվյալների շրջանակը. Հետազոտությունը պահում է **միայն 4 դաս** և չի օգտագործում malignancy-ն։ Նշումները՝ binary + other_benign + indeterminate բացառում + dedup։ | Ընդլայնված է նշումների համաձայն։ Հետազոտության 4-դասանի արդյունքները չեն համեմատվում ուղղակիորեն։ |
| 11 | `01_Preprocessing.ipynb` | Հետազոտությունը **assert** է անում, որ train և test-ի հիվանդները չեն համընկնում։ Նշումներում՝ ~21 պատկեր / 3 հիվանդ համընկնում է։ | Հետազոտության assert-ը ամբողջ տվյալների վրա կձախողվեր, եթե համընկնում լինի։ Նշումների համաձայն՝ train-ի այդ պատկերները հեռացվում են և գրանցվում `patient_overlap_excluded.csv`-ում։ **Ստուգել Kaggle տարբերակով.** |
| 12 | `01_Preprocessing.ipynb` vs `README` | Տվյալների թղթապանակ՝ README-ն՝ `data/DERM12345/`, կոդը՝ `derma_data`։ | Կոնֆիգում՝ `data.raw_dir: data/raw/DERM12345`, կամ `DIAG_RAW_DIR` միջավայր։ |
| 13 | `01_Preprocessing.ipynb` | Մետատվյալների սխեման. հետազոտությունը՝ `image_id`, `patient_id`, `label`, `split`։ Նշումներ՝ `isic_id`/`image_id`, `study_patient_id`/`patient_id`, `malignancy`, `split`։ | Կոնֆիգում՝ թեկնածու սյունակների ցուցակներ (առաջինը գտնվածը)։ **Kaggle տարբերակը ստուգված չէ** (ցանցը հասանելի չէր)։ |
| 14 | `01_Preprocessing.ipynb` | Պատկերները գտնվում են stem-ով (մեծատառի անտեսումով), ինչպես նշումներում։ Dedup-ը ավելացված է։ | Dedup-ը ավելացված է (SHA-256 և pHash)։ Stem-ների կրկնությունը վերածվում է սխալի։ |
| 15 | Առաջադրանքի նշումներ (`srjd` test) | `srjd`-ն ունի 4 test պատկեր. հետազոտության README-ն արդեն նշում է անկայունությունը։ | Գնահատումը զգուշացնում է, նաև bootstrap CI-ով։ |

### 9.2. Փաստացի թվեր քայլ 1-ի համար (`data.expected` ↔ `manifest_summary.json`)

**Չի կատարվել** — DERM12345 տվյալները հասանելի չեն եղել։ Գործարկելուց հետո համեմատության աղյուսակը
գրվում է `data/manifests/manifest_summary.json`-ում (բաժին `expected_vs_actual`) և արտապատկերվում
`prepare`-ի վերջում։ Սպասվող թվերը՝ 12 345 / 9 860 / 2 485 / 1 627 / 40 / 58 indeterminate / 21 պատկեր / 3 հիվանդ։

### 9.3. Բաց որոշումներ (պահանջում են Ձեր որոշումը)

1. **Colab նոթատետրի** ներբեռնումը — եթե այն պարունակում է այլ հիպերպարամետրեր, թվարկեք, և համեմատությունը կկատարվի։
2. **Kaggle տարբերակը** — որ տարբերակն է համապատասխանում պաշտոնական բաժանմանը և ինչ է սյունակների իրական անունը։
3. **Batch size-երը** (9.1 #4) — ընդունել առաջադրանքի նշումների, թե՞ հետազոտության արժեքները (GPU հիշողության պատճառով)։
4. **Fine-tune epochs** (9.1 #5) — 12 թե 8։
5. **`target_sensitivity`** — 0.90 լռելյայն է, պետք է հաստատվի կլինիցիստի կողմից։
6. **`include_other_benign`** — լռելյայն՝ `true` (Colab-ի պես)։
7. **Ensemble mode** — `average` թե `weighted` (լռելյայն՝ average)։
8. **Կոդի լիցենզիան** և **կշիռների արտաքին հղումը** (README §5, §8)։

## Summary in English

**diagnostika** is a research prototype for two-stage analysis of dermatoscopic lesions on DERM12345.

- **Stage 1** — benign vs malignant. Four timm architectures (ResNet50, DenseNet121, EfficientNet-B7, ViT-B/16),
  averaged. The threshold is the largest value whose **validation** sensitivity reaches the target (default 0.90).
  It must be reviewed by a clinician. The test set is never used to choose it.
- **Stage 2** — benign subtype (bdb, cb, db, srjd, and optionally other_benign), same four architectures, selected
  by validation macro-F1. Ensemble: `average` or `weighted` (per-class validation F1).
- **Output**: `P(malignant)` and `P(benign_c) = (1 − p) · E_c`, which always sums to 1, plus referral, unclear and
  agreement flags. When referral is set, the subtype distribution is marked "for information only".
- **Data**: `src/data/prepare.py` builds manifests. It excludes indeterminate rows, removes train images of test
  patients, deduplicates (SHA-256 exact, pHash near with threshold 4, keeping the test copy), and makes a
  patient-grouped stratified validation split. Manifests are hashed; every downstream step verifies the hash.
- **Evaluation**: run once on the test manifest. It reports sensitivity, specificity, ROC-AUC, PR-AUC, subtype
  macro-F1, per-class recall, end-to-end cascade accuracy, and 95% patient-level bootstrap intervals.
- **App**: FastAPI backend (`POST /api/analyze`, ROI in normalized coordinates) and one static Armenian page with
  no external calls. A doctor can log their own diagnosis and download a CSV. Images are not stored.

**Status in this repository:** the code, configuration, tests and this README are complete. The data (DERM12345),
the pretrained ImageNet weights and any trained weights were **not available** in the environment where this was
built, so **no metrics, threshold or results are reported**. The discrepancies between the research sources and
the specification are listed in §9.1; the Colab notebook named in the task was not found.

**Reproduce**: see §4 (commands in order). Tests: `pytest` (dedup, patient leakage, threshold, ensemble, API).

**What was verified without the real data:** `pytest` (31 tests) passes. The whole chain (prepare → both training
stages → threshold → evaluation → API) was run once on a small synthetic DERM12345-shaped dataset with tiny
backbones (1 epoch, 64 px); this checks plumbing only, not model quality. The `weighted` ensemble mode was run the
same way. The browser UI was not rendered or tested at 360 px width in this environment.
