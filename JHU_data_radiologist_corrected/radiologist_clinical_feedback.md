# Radiologist Clinical Feedback & Mask Reuse Master Guide

**Dataset Directory:** `03_Datasets/JHU_data_radiologist_corrected`  
**Purpose:** Comprehensive translation, contextualization, and analysis of expert radiologist corrections and annotations for the BDMAP dataset (`BDMAP_*` scans).

---

## Executive Summary

The radiologist evaluation of the BDMAP dataset reveals two critical insights:
1. **Multi-Phase Mask Reusability:** Several BDMAP scans correspond to sequential contrast/imaging phases of the same patients. Radiologist-corrected masks for key scans can be directly propagated to sibling scans (e.g., `113` $\rightarrow$ `115`/`116`, `131` $\rightarrow$ `132`/`133`, `136` $\rightarrow$ `135`).
2. **Systematic Model Misclassifications & Pitfalls:** Standard automated models (e.g., baseline TotalSegmentator or preliminary MedNeXt) exhibit consistent failure modes:
   * **Spatial Confusion:** Misidentifying distal rectum as proximal esophagus.
   * **Vascular Density Overlap:** Misclassifying major venous structures (Superior Mesenteric Vein, Right Iliac Vein) as small intestine due to similar Hounsfield Unit (HU) attenuation.
   * **Anatomical Boundary Blurring:** Misidentifying liver tissue as stomach when fat planes are indistinct.
   * **Lesions & Surgical Changes:** Missed bowel lesions (e.g., cecal carcinoma/soft tissue mass in `BDMAP_00394224`) and unaccounted post-surgical anatomy.

---

## 1. Multi-Phase Mask Propagation Matrix

Based on `notes.md` and `note2.png`, the following annotation propagation rules apply across multi-phase scans of the same patient:

| Source Corrected Mask | Derived / Sibling Scans | Radiologist Instructions & Notes |
| :--- | :--- | :--- |
| **`BDMAP_00242113.seg.nrrd`** | `BDMAP_00242114` *(slight edit)*<br>`BDMAP_00242115`<br>`BDMAP_00242116` | `114` was slightly modified based on `113`. Scans `115` and `116` can both directly reuse `113`'s annotation mask. |
| **`BDMAP_00242131.seg.nrrd`** | `BDMAP_00242132`<br>`BDMAP_00242133` | Scans `131`, `132`, and `133` can all reuse the annotation mask from `131`. |
| **`BDMAP_00242136.seg.nrrd`** | `BDMAP_00242135` | Scan `135` can reuse scan `136`'s contour (different scan phase of the same patient). |

---

## 2. Detailed Per-Scan Clinical Annotations

### `BDMAP_00242131` (Source Image: `note1.png`)
* **Original Notes:**
  1. 直肠总识别到食管，或许需要限制一下区域
  2. 将肠系膜上静脉识别为小肠
  3. 大片的小肠漏识别，完全没有道理，有的小肠内还有阳性造影剂
* **Clinical Breakdown:**
  * **Rectum $\leftrightarrow$ Esophagus Confusion:** The model mislabels rectum as esophagus. Spatial bounding (z-axis ROI limits) is needed.
  * **Vessel Misclassification:** Superior Mesenteric Vein (SMV) is misidentified as small bowel.
  * **Severe Under-segmentation:** Massive missed regions of small bowel despite clear positive oral contrast in some intestinal loops.

---

### `BDMAP_00242133` (Source Image: `note2.png`)
* **Original Notes:**
  1. 结肠错误识别到小肠（左上腹）
  2. 乙状结肠及回盲部识别不全
* **Clinical Breakdown:**
  * **Colon $\leftrightarrow$ Small Bowel Confusion:** Colon in the left upper quadrant is misclassified as small bowel.
  * **Incomplete Segments:** Sub-optimal / incomplete identification of the sigmoid colon and ileocecal region.

---

### `BDMAP_00242134` (Source Image: `note2.png`)
* **Original Notes:**
  1. 直肠总会识别到食管
  2. 食管上段往往会漏识别，原因未知
  3. 小肠的识别总是不全，似乎和密度（肠腔内是否有造影剂等）关系不大
* **Clinical Breakdown:**
  * **Rectum $\leftrightarrow$ Esophagus Confusion:** Rectum erroneously identified as esophagus.
  * **Upper Esophagus Dropout:** Proximal upper esophagus is frequently missed/omitted by the segmenter.
  * **Small Bowel Incompleteness:** Small intestine segmentation remains consistently incomplete regardless of intraluminal contrast density.

---

### `BDMAP_00242136` (Source Image: `note2.png`)
* **Original Notes:**
  1. Colon: 右半结肠及回盲部走形变异、乙状结肠冗长
  2. 降结肠与乙状结肠连接处表现欠佳，原因未知
  3. 结肠肝区及降结肠错误标记，与默认肠道走形有关，常规在该位置，但患者解剖变异
  4. 小肠表现较差，和部分肠道未充盈有关，有些或许与走形零散有关
  5. 胃主要是和肝脏间隙不清楚的时候会错误识别部分肝脏
* **Clinical Breakdown:**
  * **Anatomical Course Variations:** Patient displays an abnormal gut course in the right colon and ileocecal region, plus a redundant/elongated sigmoid colon.
  * **Tension / Transition Zones:** Descending-to-sigmoid colon junction exhibits poor segmentation.
  * **Anatomical Mislabeling:** Hepatic flexure and descending colon are mislabeled because standard atlas/priors expect typical bowel paths, but this patient has significant anatomical variations.
  * **Bowel Distension Failure:** Small bowel segmentation is poor due to un-distended / collapsed bowel loops and fragmented course.
  * **Liver $\leftrightarrow$ Stomach Boundary Failure:** Parts of the liver are mislabeled as stomach whenever the visceral fat plane separating them is indistinct.

---

### `BDMAP_00394224` (Source Image: `notes3_for224.png`)
* **Original Notes:**
  1. 未勾画肠道病灶，盲肠似见软组织密度，需要结合多期增强扫描图像并根据 baseline 及近期图像判断结肠 CA 真实病灶位置
  2. 患者直肠做过手术，报告并未提及。回盲瓣处软组织病灶需对比前片
  3. 标记总体靠谱，食管上段未标注，回盲部标注有少许误差
  4. 将右侧髂静脉识别为小肠，可能 CT 值接近
* **Clinical Breakdown:**
  * **Uncontoured Intestinal Lesion / Malignancy:** Cecum displays a soft-tissue density (suspected Colon Cancer / CA). Needs multi-phase contrast CT comparison, baseline, and prior scans to confirm true tumor extent.
  * **Post-Surgical Anatomy & Lesion:** Patient underwent prior rectal surgery (unmentioned in standard clinical report). Soft tissue lesion near the ileocecal valve requires baseline image comparison.
  * **Esophagus & Ileocecal Boundary Errors:** Upper esophagus was left un-annotated; minor errors around the ileocecal junction.
  * **Iliac Vein Misclassification:** Right iliac vein misidentified as small bowel due to overlapping CT Hounsfield Unit (HU) attenuation values.

---

## 3. Recommended Action Plan & Implementation Steps

1. **Mask Propagation Pipeline**:
   * Create symlinks or copy masks according to the propagation table:
     * `BDMAP_00242113.seg.nrrd` $\rightarrow$ `BDMAP_00242115.seg.nrrd`, `BDMAP_00242116.seg.nrrd`
     * `BDMAP_00242131.seg.nrrd` $\rightarrow$ `BDMAP_00242132.seg.nrrd`, `BDMAP_00242133.seg.nrrd`
     * `BDMAP_00242136.seg.nrrd` $\rightarrow$ `BDMAP_00242135.seg.nrrd`
2. **Model Training & Loss Engineering**:
   * **Spatial Constraints:** Apply bounding box / z-coordinate ROI limits to penalize distant organ swaps (e.g. Rectum in lower pelvis labeled as Esophagus).
   * **Asymmetric Partial Cross-Entropy (`AsymmetricPDCELoss`) & Ignore Masks:** For ambiguous boundary regions (liver-stomach interface, vascular boundaries), assign `ignore` class labels so the loss function does not penalize soft predictions.
3. **Data Quality & Topology Verification**:
   * Flag cases with major anatomical variants or post-surgical alterations (`BDMAP_00394224`) for specialized validation or exclusion from standard baseline training splits.
